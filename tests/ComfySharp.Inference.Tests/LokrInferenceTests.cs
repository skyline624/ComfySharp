using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LokrInferenceTests
{
    public static IEnumerable<object[]> Cases => Enumerable.Range(0,150).Select(i=>new object[]{i});
    private static JsonDocument Fixture()
    {
        var bytes=ClipReferenceTests.Resource("lokr-inference.reference.json");
        Assert.Equal("92aff7aef2282cd22e51971e33526a9cdf7300b1667e5bb72d5a345923c3fcff",Convert.ToHexStringLower(SHA256.HashData(bytes)));
        return JsonDocument.Parse(bytes);
    }
    private static Tensor Read(JsonElement e)=>tensor(e.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray(),
        e.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray());
    private static void Near(Tensor actual,JsonElement expected)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()),actual.shape);
        var values=expected.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray();var observed=actual.data<float>().ToArray();
        for(int i=0;i<values.Length;i++)Assert.True(float.IsFinite(observed[i])&&Math.Abs(observed[i]-values[i])<=3e-5+3e-5*Math.Abs(values[i]),$"[{i}]: {observed[i]} vs {values[i]}");
    }
    [Theory] [MemberData(nameof(Cases))]
    public void File_and_native_snapshots_follow_frozen_weight_inference(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        string path=Path.Combine(Path.GetTempPath(),"comfysharp-lokr-"+Guid.NewGuid().ToString("N")+".safetensors");
        try
        {
            using var scope=NewDisposeScope();using var json=Fixture();var row=json.RootElement.GetProperty("cases")[index];
            Assert.Equal(3e-5,json.RootElement.GetProperty("absoluteTolerance").GetDouble());
            Assert.Equal(3e-5,json.RootElement.GetProperty("relativeTolerance").GetDouble());
            var dtype=row.GetProperty("dtype").GetString() switch {"torch.float16"=>ScalarType.Float16,"torch.bfloat16"=>ScalarType.BFloat16,_=>ScalarType.Float32};
            var values=row.GetProperty("factors").EnumerateObject().ToDictionary(p=>"layer."+p.Name,p=>Read(p.Value).to_type(p.Name=="alpha"?ScalarType.Float32:dtype));
            using(var output=File.Create(path))SafeTensorWriter.Write(output,values);
            using var file=new SafeTensorFile(path);using var native=new NativeLoraTensorSource(values);
            using var weight=Read(row.GetProperty("weight"));var original=weight.bytes.ToArray();
            LoraAlias[] aliases=[new("layer",new("model","layer.weight",weight.shape))];
            var plan=LoraFileLoader.Inspect(file,aliases);var nativePlan=LoraFileLoader.Inspect(native,aliases);
            var keys=Assert.Single(plan.Bindings).Lokr;Assert.NotNull(keys);
            Assert.Equal(row.GetProperty("loadedKeys").EnumerateArray().Select(v=>v.GetString()),keys!.Values.Order(StringComparer.Ordinal));
            Assert.Empty(plan.UnclaimedKeys);
            Assert.Equal(values.Where(p=>p.Key!="layer.alpha").Sum(p=>p.Value.numel()*4),plan.ResidentFactorBytes);
            Assert.Throws<NotSupportedException>(()=>LoraFileLoader.Inspect(file,aliases,maxResidentFactorBytes:plan.ResidentFactorBytes-1));
            double strength=row.GetProperty("strength").GetDouble();var strengths=new Dictionary<string,double>{{"model",strength}};
            using var loaded=LoraFileLoader.Load(file,plan,strengths);using var fromNative=LoraFileLoader.Load(native,nativePlan,strengths);
            double? alpha=row.GetProperty("alpha").ValueKind==JsonValueKind.Null?null:row.GetProperty("alpha").GetDouble();
            using var patch=LoraWeightPatch.FromLokr(values.Where(p=>p.Key.StartsWith("layer.lokr_",StringComparison.Ordinal)).ToDictionary(p=>p.Key[6..],p=>p.Value),strength,alpha,values.GetValueOrDefault("layer.dora_scale"));
            using var retained=patch.Retain();using var moved=patch.To(CPU);patch.Dispose();
            file.Dispose();native.Dispose();foreach(var value in values.Values)value.fill_(0);
            if(row.GetProperty("sourceErrors").GetArrayLength()>0)
            {
                // Source logs this native failure and returns unchanged weights. Do not
                // present a silently skipped adapter as successful ComfySharp inference.
                Near(weight,row.GetProperty("output"));long current=Tensor.TotalCount;
                var error=Assert.ThrowsAny<Exception>(()=>{using var ignored=loaded.Apply("model","layer.weight",weight);});
                Assert.Contains("view size is not compatible",error.Message);
                Assert.Equal(current,Tensor.TotalCount);
                error=Assert.ThrowsAny<Exception>(()=>{using var ignored=moved.Apply(weight);});
                Assert.Contains("view size is not compatible",error.Message);
                Assert.Equal(current,Tensor.TotalCount);
            }
            else
            {
                using var actual=loaded.Apply("model","layer.weight",weight);using var again=fromNative.Apply("model","layer.weight",weight);
                Near(actual,row.GetProperty("output"));Assert.Equal(actual.bytes.ToArray(),again.bytes.ToArray());
                using var frozen=moved.Apply(weight);using var retainedResult=retained.Apply(weight);
                Near(frozen,row.GetProperty("output"));Assert.Equal(frozen.bytes.ToArray(),retainedResult.bytes.ToArray());
            }
            Assert.Equal(original,weight.bytes.ToArray());
            Assert.Throws<OperationCanceledException>(()=>retained.Apply(weight,new(true)));
            Assert.Throws<ObjectDisposedException>(()=>patch.Retain());
        }
        finally{File.Delete(path);}
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Theory] [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void Provider_alias_and_difference_order_matches_source_loader(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var json=Fixture();var row=json.RootElement.GetProperty("loaderCases")[index];
            var values=row.GetProperty("tensors").EnumerateObject().ToDictionary(p=>p.Name,p=>Read(p.Value));
            using var source=new NativeLoraTensorSource(values);using var weight=Read(row.GetProperty("weight"));
            var aliases=row.GetProperty("aliases").EnumerateArray().Select(v=>new LoraAlias(v.GetString()!,new("model","layer.weight",weight.shape))).ToArray();
            var plan=LoraFileLoader.Inspect(source,aliases);var binding=Assert.Single(plan.Bindings);
            Assert.Equal(row.GetProperty("selectedKind").GetString(),binding.Difference is not null?"diff":binding.Lokr is not null?"lokr":binding.Loha is not null?"loha":"lora");
            Assert.Empty(plan.UnclaimedKeys);using var adapters=LoraFileLoader.Load(source,plan);
            using var result=adapters.Apply("model","layer.weight",weight);Near(result,row.GetProperty("output"));
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Fact]
    public void Invalid_factors_and_cancellation_release_all_owned_resources()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            var factors=new Dictionary<string,Tensor>{{"lokr_w1",ones(2,2)},{"lokr_w2",ones(3,2)}};
            LoraAlias[] aliases=[new("layer",new("model","layer.weight",new long[]{6,4}))];
            using var source=new NativeLoraTensorSource(factors.ToDictionary(p=>"layer."+p.Key,p=>p.Value));
            var plan=LoraFileLoader.Inspect(source,aliases);long current=Tensor.TotalCount;
            Assert.Throws<OperationCanceledException>(()=>LoraFileLoader.Load(source,plan,cancellationToken:new(true)));
            Assert.Equal(current,Tensor.TotalCount);
            using var missing=new NativeLoraTensorSource(new Dictionary<string,Tensor>{{"layer.lokr_w1",factors["lokr_w1"]}});
            Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(missing,aliases));
            Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(source,[new("layer",new("model","layer.weight",new long[]{7,4}))]));
            factors["lokr_w2"].fill_(float.NaN);current=Tensor.TotalCount;
            Assert.Throws<InvalidDataException>(()=>LoraWeightPatch.FromLokr(factors));Assert.Equal(current,Tensor.TotalCount);
            factors["lokr_w2"].fill_(1);factors["unexpected"]=ones(1);
            Assert.Throws<ArgumentException>(()=>LoraWeightPatch.FromLokr(factors));
            Assert.Throws<ArgumentOutOfRangeException>(()=>LoraWeightPatch.FromLokr(factors,alpha:double.NaN));
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
