using System.Buffers.Binary;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LokrBypassLoadingTests
{
    public static IEnumerable<object[]> Cases=>Enumerable.Range(0,22).Select(i=>new object[]{i});
    private static JsonDocument Fixture()
    {
        var packed=ClipReferenceTests.Resource("lokr-bypass-loading.reference.json.gz");
        Assert.Equal("be9918b7cd2ebfb413f387bbecb0eaf8b44b4254b8da8eb5fab5cc889a13fa23",Convert.ToHexStringLower(SHA256.HashData(packed)));
        using var stream=new MemoryStream(packed);using var gzip=new GZipStream(stream,CompressionMode.Decompress);using var output=new MemoryStream();gzip.CopyTo(output);
        var raw=output.ToArray();Assert.Equal("2b18d476cfe48f06d9fc2cbd9f1002bbc658428cf75284496fc298ed9fc147e8",Convert.ToHexStringLower(SHA256.HashData(raw)));return JsonDocument.Parse(raw);
    }
    private static Tensor Read(JsonElement value,JsonElement root)
    {
        var sha=value.GetProperty("sha256").GetString()!;var bytes=Convert.FromBase64String(root.GetProperty("payloads").GetProperty(sha).GetString()!);
        Assert.Equal(sha,Convert.ToHexStringLower(SHA256.HashData(bytes)));var values=new float[bytes.Length/4];
        for(int i=0;i<values.Length;i++)values[i]=BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i*4,4));
        return tensor(values,value.GetProperty("shape").EnumerateArray().Select(n=>n.GetInt64()).ToArray());
    }
    [Theory] [MemberData(nameof(Cases))]
    public void File_loading_and_addition_preserve_source_bypass_geometry(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        string path=Path.Combine(Path.GetTempPath(),"comfysharp-lokr-bypass-"+Guid.NewGuid().ToString("N")+".safetensors");
        try
        {
            using var scope=NewDisposeScope();using var fixture=Fixture();var root=fixture.RootElement;var row=root.GetProperty("cases")[index];string name=row.GetProperty("name").GetString()!;
            Assert.Equal(3e-5,root.GetProperty("absoluteTolerance").GetDouble());Assert.Equal(3e-5,root.GetProperty("relativeTolerance").GetDouble());
            var values=row.GetProperty("factors").EnumerateObject().ToDictionary(p=>"layer."+p.Name,p=>Read(p.Value,root));
            values.Add("layer.dora_scale",Read(row.GetProperty("dora"),root));values.Add("layer.alpha",tensor(row.GetProperty("alpha").GetSingle()));
            using(var output=File.Create(path))SafeTensorWriter.Write(output,values);
            using var file=new SafeTensorFile(path);using var native=new NativeLoraTensorSource(values);
            var target=row.GetProperty("target").EnumerateArray().Select(n=>n.GetInt64()).ToArray();LoraAlias[] aliases=[new("layer",new("model","layer.weight",target))];
            using var input=Read(row.GetProperty("input"),root);using var baseOutput=Read(row.GetProperty("base"),root);var original=baseOutput.bytes.ToArray();
            if(row.GetProperty("error").ValueKind!=JsonValueKind.Null&&name!="error-output-spatial")
            {
                Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(file,aliases,mode:LoraLoadMode.Bypass));
                Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(native,aliases,mode:LoraLoadMode.Bypass));
            }
            else
            {
                var plan=LoraFileLoader.Inspect(file,aliases,mode:LoraLoadMode.Bypass);Assert.Equal(LoraLoadMode.Bypass,plan.Mode);Assert.Empty(plan.UnclaimedKeys);
                Assert.Throws<NotSupportedException>(()=>LoraFileLoader.Inspect(file,aliases,maxResidentFactorBytes:plan.ResidentFactorBytes-1,mode:LoraLoadMode.Bypass));
                using var loaded=LoraFileLoader.Load(file,plan,new Dictionary<string,double>{{"model",row.GetProperty("strength").GetDouble()}});
                using var fromNative=LoraFileLoader.Load(native,LoraFileLoader.Inspect(native,aliases,mode:LoraLoadMode.Bypass),new Dictionary<string,double>{{"model",row.GetProperty("strength").GetDouble()}});
                file.Dispose();native.Dispose();foreach(var value in values.Values)value.fill_(0);
                Assert.Throws<InvalidOperationException>(()=>{using var ignored=loaded.Apply("model","layer.weight",zeros(target));});
                if(name=="error-output-spatial")
                    Assert.Throws<System.Runtime.InteropServices.ExternalException>(()=>{using var ignored=loaded.ApplyBypass("model","layer.weight",input,baseOutput);});
                else
                {
                    using var actual=loaded.ApplyBypass("model","layer.weight",input,baseOutput,row.GetProperty("stride").GetInt64(),row.GetProperty("padding").GetInt64());
                    using var again=fromNative.ApplyBypass("model","layer.weight",input,baseOutput);
                    using var expected=Read(row.GetProperty("output"),root);Assert.Equal(expected.shape,actual.shape);
                    Assert.True(actual.isfinite().all().item<bool>());Assert.True(allclose(actual,expected,atol:3e-5,rtol:3e-5));Assert.Equal(actual.bytes.ToArray(),again.bytes.ToArray());
                }
                Assert.Throws<OperationCanceledException>(()=>loaded.ApplyBypass("model","layer.weight",input,baseOutput,cancellationToken:new(true)));
                Assert.Equal(original,baseOutput.bytes.ToArray());
            }
        }
        finally{File.Delete(path);}
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(4)] [InlineData(6)] [InlineData(8)] [InlineData(9)] [InlineData(10)] [InlineData(11)] [InlineData(12)] [InlineData(13)] [InlineData(14)]
    public void Bypass_does_not_relax_weight_reconstruction_validation(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var fixture=Fixture();var root=fixture.RootElement;var row=root.GetProperty("cases")[index];
            // No ignored DoRA scale: the refusal must come from weight geometry.
            var values=row.GetProperty("factors").EnumerateObject().ToDictionary(p=>"layer."+p.Name,p=>Read(p.Value,root));using var source=new NativeLoraTensorSource(values);
            var target=row.GetProperty("target").EnumerateArray().Select(n=>n.GetInt64()).ToArray();LoraAlias[] aliases=[new("layer",new("model","layer.weight",target))];
            Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(source,aliases));
            var plan=LoraFileLoader.Inspect(source,aliases,mode:LoraLoadMode.Bypass);Assert.Equal(LoraLoadMode.Bypass,plan.Mode);
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
