using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LohaInferenceTests
{
    public static IEnumerable<object[]> Cases => Enumerable.Range(0,48).Select(i=>new object[]{i});
    private static JsonDocument Fixture()
    {
        var bytes=ClipReferenceTests.Resource("loha-inference.reference.json");
        Assert.Equal("d5d66d9add1dc642159f64f56864e8cea08522d47af61b24cf5be3b88d1cd426",Convert.ToHexStringLower(SHA256.HashData(bytes)));
        return JsonDocument.Parse(bytes);
    }
    private static Tensor Read(JsonElement e)=>tensor(e.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray(),
        e.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray());
    private static void Near(Tensor actual,JsonElement expected,string name)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()),actual.shape);
        var values=expected.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray();var observed=actual.data<float>().ToArray();
        for(int i=0;i<values.Length;i++)Assert.True(float.IsFinite(observed[i])&&Math.Abs(observed[i]-values[i])<=3e-5+3e-5*Math.Abs(values[i]),
            $"{name}[{i}]: {observed[i]} vs {values[i]}");
    }
    [Theory] [MemberData(nameof(Cases))]
    public void File_native_snapshot_and_bypass_match_frozen_LoHaAdapter(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        string path=Path.Combine(Path.GetTempPath(),"comfysharp-loha-inference-"+Guid.NewGuid().ToString("N")+".safetensors");
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
            Assert.NotNull(Assert.Single(plan.Bindings).Loha);
            Assert.Equal(values.Where(p=>p.Key!="layer.alpha").Sum(p=>p.Value.numel()*4),plan.ResidentFactorBytes);
            Assert.Throws<NotSupportedException>(()=>LoraFileLoader.Inspect(file,aliases,maxResidentFactorBytes:plan.ResidentFactorBytes-1));
            var claimed=plan.Bindings.SelectMany(b=>new[]{b.Up!,b.Down!,b.Loha!.W2A,b.Loha.W2B,b.Loha.T1,b.Loha.T2}.OfType<string>()).Order(StringComparer.Ordinal);
            Assert.Equal(row.GetProperty("loadedKeys").EnumerateArray().Select(v=>v.GetString()),claimed);
            double strength=row.GetProperty("strength").GetDouble();var strengths=new Dictionary<string,double>{{"model",strength}};
            using var loaded=LoraFileLoader.Load(file,plan,strengths);using var fromNative=LoraFileLoader.Load(native,nativePlan,strengths);
            file.Dispose();native.Dispose();foreach(var v in values.Values)v.fill_(0);
            using var actual=loaded.Apply("model","layer.weight",weight);using var again=fromNative.Apply("model","layer.weight",weight);
            Near(actual,row.GetProperty("output"),"weight");Assert.Equal(actual.bytes.ToArray(),again.bytes.ToArray());Assert.Equal(original,weight.bytes.ToArray());
            // Fresh inputs come from the independent source record; owner must freeze and retain every factor.
            var factors=row.GetProperty("factors").EnumerateObject().ToDictionary(p=>p.Name,p=>Read(p.Value));
            double? alpha=row.GetProperty("alpha").ValueKind==JsonValueKind.Null?null:row.GetProperty("alpha").GetDouble();
            using var patch=LoraWeightPatch.FromLoha(factors["hada_w1_a"],factors["hada_w1_b"],factors["hada_w2_a"],factors["hada_w2_b"],
                strength,alpha,factors.GetValueOrDefault("hada_t1"),factors.GetValueOrDefault("hada_t2"),factors.GetValueOrDefault("dora_scale"));
            using var retained=patch.Retain();using var moved=patch.To(CPU);patch.Dispose();foreach(var v in factors.Values)v.fill_(0);
            using var frozen=moved.Apply(weight);Near(frozen,row.GetProperty("output"),"snapshot");
            using var input=Read(row.GetProperty("input")).requires_grad_();
            bool conv=row.GetProperty("kind").GetString() is "conv2d" or "tucker2d";
            var baseOutput=conv?nn.functional.conv2d(input,weight,strides:new long[]{2,2},padding:new long[]{1,1}):nn.functional.linear(input,weight);
            using var bypass=retained.ApplyBypass(input,baseOutput,conv?new long[]{2,2}:null,conv?2:1,conv?1:0);
            Near(bypass,row.GetProperty("bypass"),"bypass");Near(baseOutput,row.GetProperty("baseOutput"),"borrowed base");
            bypass.square().mean().backward();using var gradient=input.grad;Assert.NotNull(gradient);Near(gradient!,row.GetProperty("inputGradient"),"activation gradient");
            Assert.Throws<OperationCanceledException>(()=>retained.Apply(weight,new(true)));
            Assert.Throws<ObjectDisposedException>(()=>patch.Retain());
        }
        finally{File.Delete(path);}
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Theory] [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void Provider_alias_and_difference_overwrites_match_actual_source_loader(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var json=Fixture();var row=json.RootElement.GetProperty("loaderCases")[index];
            var values=row.GetProperty("tensors").EnumerateObject().ToDictionary(p=>p.Name,p=>Read(p.Value));
            using var source=new NativeLoraTensorSource(values);using var weight=Read(row.GetProperty("weight"));
            var aliases=row.GetProperty("aliases").EnumerateArray().Select(v=>new LoraAlias(v.GetString()!,new("model","layer.weight",weight.shape))).ToArray();
            var plan=LoraFileLoader.Inspect(source,aliases);var binding=Assert.Single(plan.Bindings);
            Assert.Equal(row.GetProperty("selectedKind").GetString(),binding.Difference is not null?"diff":binding.Loha is not null?"loha":"lora");
            Assert.Empty(plan.UnclaimedKeys);using var adapters=LoraFileLoader.Load(source,plan);
            using var result=adapters.Apply("model","layer.weight",weight);Near(result,row.GetProperty("output"),"source loader");
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Missing_cores_geometry_nonfinite_and_precedence_are_explicit()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            var values=new Dictionary<string,Tensor>{{"layer.hada_w1_a",ones(3,2)},{"layer.hada_w1_b",ones(2,5)},
                {"layer.hada_w2_a",ones(3,1)},{"layer.hada_w2_b",ones(1,5)}};
            LoraAlias[] aliases=[new("layer",new("model","layer.weight",new long[]{3,5}))];
            using(var missing=new NativeLoraTensorSource(values.Where(p=>p.Key!="layer.hada_w2_b").ToDictionary()))
                Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(missing,aliases));
            values["layer.hada_t1"]=ones(2,2,2,2);
            using(var missing=new NativeLoraTensorSource(values))Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(missing,aliases));
            values.Remove("layer.hada_t1");values["layer.hada_t2"]=ones(2,2,2,2);
            using(var orphan=new NativeLoraTensorSource(values))
            {
                Assert.Throws<NotSupportedException>(()=>LoraFileLoader.Inspect(orphan,aliases));
                Assert.Equal(new[]{"layer.hada_t2"},LoraFileLoader.Inspect(orphan,aliases,allowUnclaimedKeys:true).UnclaimedKeys);
            }
            values.Remove("layer.hada_t2");values["layer.lora_up.weight"]=ones(3,1);values["layer.lora_down.weight"]=ones(1,5);
            using(var mixed=new NativeLoraTensorSource(values))
            {
                var plan=LoraFileLoader.Inspect(mixed,aliases);Assert.NotNull(Assert.Single(plan.Bindings).Loha);Assert.Empty(plan.UnclaimedKeys);Assert.Empty(plan.ShadowedPrefixes);
            }
            values.Remove("layer.lora_up.weight");values.Remove("layer.lora_down.weight");values["layer.diff"]=full(new long[]{3,5},.2f);
            using(var mixed=new NativeLoraTensorSource(values))
            {
                var plan=LoraFileLoader.Inspect(mixed,aliases);Assert.Equal("layer.diff",Assert.Single(plan.Bindings).Difference);Assert.Equal(60,plan.ResidentFactorBytes);
            }
            values.Remove("layer.diff");values["layer.hada_w2_b"].fill_(float.NaN);
            using(var invalid=new NativeLoraTensorSource(values))
            {
                var plan=LoraFileLoader.Inspect(invalid,aliases);long current=Tensor.TotalCount;
                Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Load(invalid,plan));Assert.Equal(current,Tensor.TotalCount);
            }
            Assert.Throws<ArgumentException>(()=>LoraWeightPatch.FromLoha(ones(3,2),ones(1,5),ones(3,1),ones(1,5)));
            Assert.Throws<ArgumentException>(()=>LoraWeightPatch.FromLoha(ones(2,3),ones(2,5),ones(2,3),ones(2,5),t1:ones(2,2,3),t2:ones(2,2,3)));
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
