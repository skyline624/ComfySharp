using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdLohaResumeTests
{
    private static JsonDocument Fixture()
    {
        byte[] compressed=ClipReferenceTests.Resource("loha-resume.reference.json.gz");
        Assert.Equal("943f420356848ea494810d22361db25b360bb94a7d5c79600cdcd6fbbc198f74",Convert.ToHexStringLower(SHA256.HashData(compressed)));
        using var input=new MemoryStream(compressed);using var gzip=new GZipStream(input,CompressionMode.Decompress);using var output=new MemoryStream();gzip.CopyTo(output);
        var bytes=output.ToArray();Assert.Equal("8c3cacc79a7f02432c08c1726ba0a9a104e83d38d22dfec9aea0e67113579b92",Convert.ToHexStringLower(SHA256.HashData(bytes)));
        return JsonDocument.Parse(bytes);
    }
    private static Tensor Read(JsonElement e)
    {
        var dtype=e.GetProperty("dtype").GetString() switch {"torch.float16"=>ScalarType.Float16,"torch.bfloat16"=>ScalarType.BFloat16,"torch.float64"=>ScalarType.Float64,_=>ScalarType.Float32};
        return tensor(e.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray(),e.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray()).to_type(dtype);
    }
    private static IReadOnlyDictionary<string,Tensor> Named(TrainableWeightPatch patch)=>patch switch
    {
        TrainableLohaPatch loha=>loha.NamedParameters,
        TrainableLoraPatch lora=>new Dictionary<string,Tensor>{{"alpha",lora.AlphaParameter!},{"lora_up.weight",lora.Up},{"lora_down.weight",lora.Down}},
        TrainableDifferencePatch diff=>new Dictionary<string,Tensor>{{"bias",diff.Difference}},
        _=>throw new NotSupportedException()
    };
    [Theory] [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void All_mixed_factory_parameters_and_rng_match_frozen_resume(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var json=Fixture();var row=json.RootElement.GetProperty("cases")[index];bool linear=row.GetProperty("linearProjection").GetBoolean();
            Assert.Equal(3e-5,json.RootElement.GetProperty("absoluteTolerance").GetDouble());Assert.Equal(3e-5,json.RootElement.GetProperty("relativeTolerance").GetDouble());
            var config=new SdUnetConfig(32,16,linear?SdAttentionHeadMode.FixedSize:SdAttentionHeadMode.FixedCount,linear?8:4,linear);
            var values=row.GetProperty("existing").EnumerateObject().ToDictionary(p=>p.Name,p=>Read(p.Value));
            using var source=new NativeLoraTensorSource(values);string algorithm=row.GetProperty("algorithm").GetString()!;
            long bytes=row.GetProperty("parameterBytes").GetInt64();long borrowed=Tensor.TotalCount;
            Assert.Throws<NotSupportedException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,bytes-1,existing:source,algorithm:algorithm));Assert.Equal(borrowed,Tensor.TotalCount);
            using var resumed=new SdTrainableAdapterSet(config,2,317,CPU,bytes,existing:source,algorithm:algorithm);
            source.Dispose();foreach(var v in values.Values){v.fill_(99);v.Dispose();}
            Assert.Equal(row.GetProperty("resumed").EnumerateArray().Select(v=>v.GetString()),resumed.ResumedTargets);
            Assert.Equal(bytes,resumed.ParameterBytes);Assert.Equal(row.GetProperty("parameterCount").GetInt32(),resumed.Patches.Values.Sum(p=>p.Parameters.Count));
            Assert.Equal(row.GetProperty("randomStateSha256").GetString(),resumed.InitialCpuRandomStateSha256);
            Assert.Equal(row.GetProperty("targets").EnumerateObject().Select(p=>p.Name),resumed.Patches.Keys);
            foreach(var (name,patch) in resumed.Patches)
            {
                var expected=row.GetProperty("targets").GetProperty(name);var named=Named(patch);
                Assert.Equal(expected.EnumerateObject().Select(p=>p.Name),named.Keys);
                foreach(var (key,value) in named)
                {
                    var record=expected.GetProperty(key);Assert.Equal(record.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()),value.shape);
                    Assert.True(value.requires_grad);Assert.Equal(ScalarType.Float32,value.dtype);
                    var wanted=record.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray();var actual=value.data<float>().ToArray();
                    if(resumed.ResumedTargets.Contains(name))Assert.Equal(wanted,actual);
                    else for(int i=0;i<wanted.Length;i++)
                        if(!float.IsFinite(actual[i])||Math.Abs(actual[i]-wanted[i])>3e-5+3e-5*Math.Abs(wanted[i]))Assert.Fail($"{name}/{key}[{i}] {actual[i]} vs {wanted[i]}");
                }
            }
            Assert.IsType<TrainableLoraPatch>(resumed.Patches["time_embed.0.weight"]); // Training's first provider wins.
            Assert.Contains("diffusion_model.time_embed.0.hada_w1_a",resumed.IgnoredExistingKeys);
            Assert.Contains("diffusion_model.out.0.diff",resumed.IgnoredExistingKeys);Assert.Contains("diffusion_model.out.2.diff_b",resumed.IgnoredExistingKeys);
            Assert.Contains("diffusion_model.out.2.alpha",resumed.IgnoredExistingKeys);
            Assert.Equal(1,Assert.IsType<TrainableLohaPatch>(resumed.Patches["out.2.weight"]).NamedParameters["alpha"].item<float>());
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Resumed_Tucker_and_plain_targets_train_then_reload_with_frozen_inference(bool tucker)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;int threads=get_num_threads();set_num_threads(1);
        try
        {
            using var scope=NewDisposeScope();var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var seedPatch=tucker?new TrainableLohaPatch(ones([2,4])*.01,ones([2,32])*.02,ones([2,4])*.03,ones([3,32])*.04,
                t1:ones([2,2,3,3])*.03,t2:ones([2,3,3,3])*.02)
                :new TrainableLohaPatch(ones([4,2])*.01,ones([2,288])*.02,ones([4,3])*.03,ones([3,288])*.04);
            using var state=LoraTrainingState.Capture(new Dictionary<string,TrainableLohaPatch>{{"out.2.weight",seedPatch}},ScalarType.Float32);
            using var source=new NativeLoraTensorSource(state.Tensors);using var resumed=new SdTrainableAdapterSet(config,2,317,CPU,existing:source,algorithm:"LoHa");
            source.Dispose();state.Dispose();seedPatch.Dispose();Assert.Equal(new[]{"out.2.weight"},resumed.ResumedTargets);
            using var bank=SdSyntheticInputs.CreateUnet(config);using var model=new SdUnet(bank);
            using var x=NativeMath.CpuNoise([1,4,8,8],511);using var context=NativeMath.CpuNoise([1,3,16],512);using var time=tensor(new[]{17.25f});
            using var baseline=model.Forward(x,time,context);using var optimizer=new LoraTrainingOptimizer(resumed.Patches.Values,"SGD",.01);
            using var first=model.ForwardForTraining(x,time,context,resumed.Patches);using var loss=first.square().mean();optimizer.Accumulate(loss);
            foreach(var patch in resumed.Patches.Values)foreach(var value in patch.Parameters)
            {
                using var gradient=value.grad;
                if(patch is TrainableLohaPatch h&&ReferenceEquals(value,h.NamedParameters["alpha"]))Assert.Null(gradient);
                else{Assert.NotNull(gradient);Assert.True(gradient!.isfinite().all().item<bool>());}
            }
            optimizer.Step();using var changed=model.ForwardForTraining(x,time,context,resumed.Patches);Assert.NotEqual(first.bytes.ToArray(),changed.bytes.ToArray());
            using var snapshot=LoraTrainingState.Capture(resumed.Patches,ScalarType.Float32);resumed.Dispose();
            using var native=new NativeLoraTensorSource(snapshot.Tensors);var plan=LoraFileLoader.Inspect(native,LoraModelAliases.ForUnet(config));using var loaded=LoraFileLoader.Load(native,plan);
            using var baked=loaded.ApplyTo(model);using var reloaded=baked.Forward(x,time,context);Assert.Equal(changed.bytes.ToArray(),reloaded.bytes.ToArray());
            using var unchanged=model.Forward(x,time,context);Assert.Equal(baseline.bytes.ToArray(),unchanged.bytes.ToArray());
        }
        finally{set_num_threads(threads);}
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Theory] [InlineData("missing")] [InlineData("core")] [InlineData("shape")] [InlineData("integer")] [InlineData("nonfinite")]
    public void Invalid_LoHa_resume_is_atomic_and_leaves_source_readable(string kind)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            const string prefix="diffusion_model.out.2";var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            var values=new Dictionary<string,Tensor>{{prefix+".hada_w1_a",ones(4,2)},{prefix+".hada_w1_b",ones(2,288)},
                {prefix+".hada_w2_a",ones(4,3)},{prefix+".hada_w2_b",ones(3,288)}};
            if(kind=="missing")values.Remove(prefix+".hada_w2_b");
            if(kind=="core")values[prefix+".hada_t1"]=ones(2,2,3,3);
            if(kind=="shape")values[prefix+".hada_w2_b"]=ones(3,12);
            if(kind=="integer")values[prefix+".hada_w2_b"]=ones(3,288,dtype:ScalarType.Int32);
            if(kind=="nonfinite")values[prefix+".hada_w2_b"].fill_(float.NaN);
            using var source=new NativeLoraTensorSource(values);long borrowed=Tensor.TotalCount;
            if(kind=="nonfinite")Assert.Throws<ArgumentException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,existing:source));
            else Assert.Throws<InvalidDataException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,existing:source));
            Assert.Equal(borrowed,Tensor.TotalCount);
            Assert.Throws<OperationCanceledException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,existing:source,cancellationToken:new(true)));
            using var stillReadable=source.ReadTensor(prefix+".hada_w1_a");Assert.False(stillReadable.IsInvalid);
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
