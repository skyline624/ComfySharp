using System.Buffers.Binary;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LokrLinearDispatchTests
{
    private static JsonDocument Fixture()
    {
        var compressed=ClipReferenceTests.Resource("lokr-linear-dispatch.reference.json.gz");
        Assert.Equal("e8ec49c3c53daedfc42a7a3689d56abce2586fffa9be63b436a65b04487cf43c",Convert.ToHexStringLower(SHA256.HashData(compressed)));
        using var input=new MemoryStream(compressed);using var gzip=new GZipStream(input,CompressionMode.Decompress);using var output=new MemoryStream();gzip.CopyTo(output);
        var raw=output.ToArray();Assert.Equal("2f903ae234d820cf5244b3ae0244fd7a932f16256dd1cdf82da6cb2bdf53b7fa",Convert.ToHexStringLower(SHA256.HashData(raw)));
        return JsonDocument.Parse(raw);
    }
    private static Tensor Read(JsonElement value,JsonElement root)
    {
        var sha=value.GetProperty("sha256").GetString()!;
        var bytes=Convert.FromBase64String(root.GetProperty("payloads").GetProperty(sha).GetString()!);
        Assert.Equal(sha,Convert.ToHexStringLower(SHA256.HashData(bytes)));
        var values=new float[bytes.Length/4];for(int i=0;i<values.Length;i++)values[i]=BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i*4,4));
        return tensor(values,value.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray());
    }
    private static void Near(Tensor actual,JsonElement value,JsonElement root)
    {
        using var expected=Read(value,root);Assert.Equal(expected.shape,actual.shape);
        Assert.True(actual.isfinite().all().item<bool>());
        Assert.True(allclose(actual,expected,atol:3e-5,rtol:3e-5));
    }
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)]
    public void Leaf_metadata_preserves_source_mm_bmm_paths_and_exact_frozen_reload(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;int threads=get_num_threads();set_num_threads(1);
        try
        {
            using var scope=NewDisposeScope();using var noGrad=no_grad();using var fixture=Fixture();var root=fixture.RootElement;var row=root.GetProperty("cases")[index];
            Assert.Equal(3e-5,root.GetProperty("absoluteTolerance").GetDouble());Assert.Equal(3e-5,root.GetProperty("relativeTolerance").GetDouble());
            using var input=Read(row.GetProperty("input"),root);using var first=Read(row.GetProperty("first"),root);using var second=Read(row.GetProperty("second"),root);
            using var baseOutput=zeros(row.GetProperty("training").GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray());
            using var owner=new TrainableLokrPatch(first,second,1);
            using var trained=owner.ApplyBypass(input,baseOutput,null,1,0);
            Assert.False(trained.requires_grad);Assert.True(owner.NamedParameters["lokr_w1"].requires_grad);
            Near(trained,row.GetProperty("training"),root);
            using var loaded=LoraWeightPatch.FromLokr(new Dictionary<string,Tensor>{{"lokr_w1",first},{"lokr_w2",second}},alpha:1);
            using var inference=loaded.ApplyBypass(input,baseOutput);
            Near(inference,row.GetProperty("inference"),root);
            foreach(var parameter in owner.Parameters)parameter.requires_grad_(false);
            using var frozen=owner.ApplyBypass(input,baseOutput,null,1,0);
            Assert.Equal(inference.bytes.ToArray(),frozen.bytes.ToArray());
            Near(frozen,row.GetProperty("frozenOwners"),root);

            // The source profiler records mm for a trainable first side, bmm for
            // its frozen equivalent. Explicit native forms verify that dispatch.
            using var grouped=LokrBypassMath.Group(input,first.shape[1],0);
            using var hidden=nn.functional.linear(grouped,second).transpose(-1,-2);
            Assert.False(hidden.is_contiguous());
            var shape=hidden.shape.Take(checked((int)hidden.dim()-1)).Append(first.shape[0]).ToArray();
            using var trainableFirst=first.clone().requires_grad_();
            using var trainCross=nn.functional.linear(hidden,trainableFirst);
            using var folded=mm(hidden.reshape(-1,hidden.shape[^1]),first.t()).reshape(shape);
            Assert.Equal(folded.bytes.ToArray(),trainCross.bytes.ToArray());
            Near(trainCross,row.GetProperty("cross")[0],root);
            using var frozenCross=nn.functional.linear(hidden,first);
            using var batches=hidden.reshape(-1,hidden.shape[^2],hidden.shape[^1]);
            using var batched=bmm(batches,first.t().expand(batches.shape[0],first.shape[1],first.shape[0])).reshape(shape);
            Assert.Equal(batched.bytes.ToArray(),frozenCross.bytes.ToArray());
            Near(frozenCross,row.GetProperty("cross")[1],root);
            Assert.Equal(1,row.GetProperty("profiles")[0].GetProperty("operators").GetProperty("aten::mm").GetInt32());
            Assert.Equal(1,row.GetProperty("profiles")[1].GetProperty("operators").GetProperty("aten::bmm").GetInt32());
        }
        finally{set_num_threads(threads);}
        Assert.Equal(before,Tensor.TotalCount);
    }
}
