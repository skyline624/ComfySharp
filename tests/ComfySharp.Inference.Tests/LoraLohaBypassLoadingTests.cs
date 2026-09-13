using System.Buffers.Binary;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraLohaBypassLoadingTests
{
    public static IEnumerable<object[]> Cases=>Enumerable.Range(0,96).Select(i=>new object[]{i});
    private static JsonDocument Fixture()
    {
        var packed=ClipReferenceTests.Resource("lora-loha-bypass-loading.reference.json.gz");
        Assert.Equal("429a2b55aaafd0c39dca04630ad62d185628acfc42e3127b3c2cb151d257c8e8",Convert.ToHexStringLower(SHA256.HashData(packed)));
        using var stream=new MemoryStream(packed);using var gzip=new GZipStream(stream,CompressionMode.Decompress);using var output=new MemoryStream();gzip.CopyTo(output);
        var raw=output.ToArray();Assert.Equal("713b1afe5d4be8936682564182574090a3dc411a3caa6e7648efff6b7ad9cdbb",Convert.ToHexStringLower(SHA256.HashData(raw)));return JsonDocument.Parse(raw);
    }
    private static Tensor Read(JsonElement value,JsonElement root)
    {
        var sha=value.GetProperty("sha256").GetString()!;var bytes=Convert.FromBase64String(root.GetProperty("payloads").GetProperty(sha).GetString()!);
        Assert.Equal(sha,Convert.ToHexStringLower(SHA256.HashData(bytes)));var values=new float[bytes.Length/4];
        for(int i=0;i<values.Length;i++)values[i]=BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i*4,4));
        return tensor(values,value.GetProperty("shape").EnumerateArray().Select(n=>n.GetInt64()).ToArray());
    }
    [Theory] [MemberData(nameof(Cases))]
    public void Loaded_operator_chains_and_broadcasts_match_frozen_source(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        string path=Path.Combine(Path.GetTempPath(),"comfysharp-adapter-bypass-"+Guid.NewGuid().ToString("N")+".safetensors");
        try
        {
            using var scope=NewDisposeScope();using var fixture=Fixture();var root=fixture.RootElement;var row=root.GetProperty("cases")[index];string name=row.GetProperty("name").GetString()!;
            Assert.Equal(3e-5,root.GetProperty("absoluteTolerance").GetDouble());Assert.Equal(3e-5,root.GetProperty("relativeTolerance").GetDouble());
            var dtype=row.GetProperty("dtype").GetString() switch{"torch.float16"=>ScalarType.Float16,"torch.bfloat16"=>ScalarType.BFloat16,_=>ScalarType.Float32};
            var values=row.GetProperty("factors").EnumerateObject().ToDictionary(p=>"layer."+p.Name,p=>Read(p.Value,root).to_type(dtype));
            using var plain=new NativeLoraTensorSource(values);
            values.Add("layer.dora_scale",Read(row.GetProperty("dora"),root));values.Add("layer.alpha",tensor(row.GetProperty("alpha").GetSingle()));
            using(var output=File.Create(path))SafeTensorWriter.Write(output,values);
            using var file=new SafeTensorFile(path);using var native=new NativeLoraTensorSource(values);
            var target=row.GetProperty("target").EnumerateArray().Select(n=>n.GetInt64()).ToArray();LoraAlias[] aliases=[new("layer",new("model","layer.weight",target))];
            using var input=Read(row.GetProperty("input"),root);using var baseOutput=Read(row.GetProperty("base"),root);var original=baseOutput.bytes.ToArray();
            if(row.GetProperty("error").ValueKind!=JsonValueKind.Null&&name!="error-spatial")
            {
                Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(file,aliases,mode:LoraLoadMode.Bypass));
                Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(native,aliases,mode:LoraLoadMode.Bypass));
            }
            else
            {
                if(name.StartsWith("broadcast",StringComparison.Ordinal)||name.StartsWith("sequential-spatial",StringComparison.Ordinal)||name is "linear-mid" or "mid-conv-1" or "mid-conv-3")
                    Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(plain,aliases));
                var plan=LoraFileLoader.Inspect(file,aliases,mode:LoraLoadMode.Bypass);Assert.Empty(plan.UnclaimedKeys);
                Assert.Throws<NotSupportedException>(()=>LoraFileLoader.Inspect(file,aliases,maxResidentFactorBytes:plan.ResidentFactorBytes-1,mode:LoraLoadMode.Bypass));
                var strength=new Dictionary<string,double>{{"model",row.GetProperty("strength").GetDouble()}};
                using var loaded=LoraFileLoader.Load(file,plan,strength);
                using var fromNative=LoraFileLoader.Load(native,LoraFileLoader.Inspect(native,aliases,mode:LoraLoadMode.Bypass),strength);
                file.Dispose();native.Dispose();plain.Dispose();foreach(var value in values.Values)value.fill_(0);
                Assert.Throws<InvalidOperationException>(()=>{using var ignored=loaded.Apply("model","layer.weight",zeros(target));});
                if(name=="error-spatial")
                    Assert.Throws<System.Runtime.InteropServices.ExternalException>(()=>{using var ignored=loaded.ApplyBypass("model","layer.weight",input,baseOutput);});
                else
                {
                    using var actual=loaded.ApplyBypass("model","layer.weight",input,baseOutput);
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
}
