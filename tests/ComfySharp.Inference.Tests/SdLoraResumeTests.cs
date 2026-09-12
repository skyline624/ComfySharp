using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdLoraResumeTests
{
    [Fact]
    public void Filename_counter_follows_source_sentinel_split_unicode_and_unbounded_integer_rules()
    {
        using var resource=typeof(SdLoraResumeTests).Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora-resume-steps.reference.json")!;
        using var json=JsonDocument.Parse(resource);
        foreach(var row in json.RootElement.GetProperty("cases").EnumerateArray())
        {
            string name=row.GetProperty("name").GetString()!;
            if(row.TryGetProperty("error",out _)) Assert.Throws<FormatException>(()=>SdTrainingResumeSteps.Parse(name));
            else Assert.Equal(row.GetProperty("steps").GetString(),SdTrainingResumeSteps.Parse(name).ToString(System.Globalization.CultureInfo.InvariantCulture));
        }
    }
    private static JsonDocument Fixture()
    {
        using var stream=typeof(SdLoraResumeTests).Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora-resume.reference.json")!;
        Assert.Equal("47bb97f556dbfa0b3e6613e2e2be8613dcc643d3e9d906add15ee58e94268086",Convert.ToHexStringLower(SHA256.HashData(stream)));
        stream.Position=0; return JsonDocument.Parse(stream);
    }
    private static Tensor Read(JsonElement value)
    {
        var dtype=value.GetProperty("dtype").GetString() switch {"torch.float16"=>ScalarType.Float16,"torch.bfloat16"=>ScalarType.BFloat16,"torch.int64"=>ScalarType.Int64,_=>ScalarType.Float32};
        return tensor(value.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray(),
            value.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray()).to_type(dtype);
    }
    private static string Digest(SdTrainableAdapterSet set)
    {
        using var hash=IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach(var (name,patch) in set.Patches)
            for(int i=0;i<patch.Parameters.Count;i++)
            { hash.AppendData(Encoding.UTF8.GetBytes(name+"/"+i+"\0")); hash.AppendData(patch.Parameters[i].bytes); }
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void Complete_parameter_initialization_and_rng_match_source_resume(int index)
    {
        NativeRuntimeBootstrap.Initialize(); long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var json=Fixture(); var row=json.RootElement.GetProperty("cases")[index]; bool linear=row.GetProperty("linearProjection").GetBoolean();
            var config=new SdUnetConfig(32,16,linear?SdAttentionHeadMode.FixedSize:SdAttentionHeadMode.FixedCount,linear?8:4,linear);
            var values=row.GetProperty("existing").EnumerateObject().ToDictionary(p=>p.Name,p=>Read(p.Value));
            using var source=new NativeLoraTensorSource(values);
            long bytes=row.GetProperty("parameterBytes").GetInt64();
            Assert.Throws<NotSupportedException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,bytes-1,existing:source));
            using var resumed=new SdTrainableAdapterSet(config,2,317,CPU,bytes,existing:source);
            foreach(var value in values.Values)value.Dispose(); source.Dispose();
            Assert.Equal(7,resumed.ResumedTargets.Count); Assert.Equal(bytes,resumed.ParameterBytes);
            Assert.Equal(row.GetProperty("allParameterSha256").GetString(),Digest(resumed));
            Assert.Equal(row.GetProperty("randomStateSha256").GetString(),resumed.InitialCpuRandomStateSha256);
            Assert.Equal(1250,resumed.Patches.Values.Sum(p=>p.Parameters.Count));
            Assert.Contains("diffusion_model.out.0.diff",resumed.IgnoredExistingKeys);
            Assert.Contains("diffusion_model.out.2.diff_b",resumed.IgnoredExistingKeys);
            Assert.Contains("diffusion_model.time_embed.0.alpha",resumed.IgnoredExistingKeys);
            foreach(var pair in row.GetProperty("resumed").EnumerateObject())
            {
                var patch=Assert.IsType<TrainableLoraPatch>(resumed.Patches[pair.Name]);
                foreach(var (name,value) in new[]{("alpha",patch.AlphaParameter!),("lora_up.weight",patch.Up),("lora_down.weight",patch.Down)})
                {
                    var expected=pair.Value.GetProperty(name); Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()),value.shape);
                    Assert.Equal(expected.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()),value.data<float>().ToArray());
                    Assert.True(value.requires_grad);
                }
            }
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void Resumed_adapters_train_and_reload_without_mutating_base(bool bypass)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;int threads=get_num_threads();set_num_threads(1);
        try
        {
            using var scope=NewDisposeScope(); using var json=Fixture();var row=json.RootElement.GetProperty("cases")[0];
            var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var source=new NativeLoraTensorSource(row.GetProperty("existing").EnumerateObject().ToDictionary(p=>p.Name,p=>Read(p.Value)));
            using var resumed=new SdTrainableAdapterSet(config,2,317,CPU,existing:source); source.Dispose();string initial=Digest(resumed);
            using var bank=SdSyntheticInputs.CreateUnet(config);using var model=new SdUnet(bank);
            using var input=NativeMath.CpuNoise([1,4,8,8],511);using var context=NativeMath.CpuNoise([1,3,16],512);using var time=tensor(new[]{17.25f});
            using var baseline=model.Forward(input,time,context);using var optimizer=new LoraTrainingOptimizer(resumed.Patches.Values,"SGD",.01);
            for(int i=0;i<2;i++)
            {
                using var iteration=NewDisposeScope();using var prediction=model.ForwardForTraining(input,time,context,resumed.Patches,bypassMode:bypass);
                using var loss=prediction.square().mean();optimizer.Accumulate(loss);
                foreach(var patch in resumed.Patches.Values)foreach(var value in patch.Parameters)
                { using var gradient=value.grad;Assert.NotNull(gradient);Assert.True(gradient!.isfinite().all().item<bool>()); }
                optimizer.Step();
            }
            Assert.NotEqual(initial,Digest(resumed));using var trained=model.ForwardForTraining(input,time,context,resumed.Patches,bypassMode:bypass);
            using var state=LoraTrainingState.Capture(resumed.Patches,ScalarType.Float32);resumed.Dispose();
            using var snapshot=new NativeLoraTensorSource(state.Tensors);var plan=LoraFileLoader.Inspect(snapshot,LoraModelAliases.ForUnet(config));
            using var adapter=LoraFileLoader.Load(snapshot,plan);using var loaded=bypass?adapter.ApplyBypassTo(model):adapter.ApplyTo(model);
            using var predictionAfter=loaded.Forward(input,time,context);using var error=(trained-predictionAfter).abs().max();Assert.InRange(error.item<float>(),0,3e-5);
            using var unchanged=model.Forward(input,time,context);Assert.Equal(baseline.bytes.ToArray(),unchanged.bytes.ToArray());
        }
        finally{set_num_threads(threads);}
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Theory]
    [InlineData("missing")] [InlineData("shape")] [InlineData("mid")] [InlineData("other")] [InlineData("alpha")] [InlineData("nonfinite")]
    public void Invalid_resume_is_atomic_and_preserves_the_source(string kind)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);const string prefix="diffusion_model.out.2";
            var values=new Dictionary<string,Tensor>{{prefix+".lora_up.weight",ones([4,1])},{prefix+".lora_down.weight",ones([1,288])}};
            if(kind=="missing")values.Remove(prefix+".lora_down.weight");
            if(kind=="shape")values[prefix+".lora_down.weight"]=ones([1,12]);
            if(kind=="mid")values[prefix+".lora_mid.weight"]=ones([1,1,1,1]);
            if(kind=="other"){values.Clear();values[prefix+".hada_w1_a"]=ones([4,1]);}
            if(kind=="alpha")values[prefix+".weight.alpha"]=ones([2]);
            if(kind=="nonfinite")values[prefix+".weight.alpha"]=tensor(float.NaN);
            using var source=new NativeLoraTensorSource(values);
            if(kind is "mid" or "other")Assert.Throws<NotSupportedException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,existing:source));
            else Assert.Throws<InvalidDataException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,existing:source));
            Assert.Throws<OperationCanceledException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,cancellationToken:new(true),existing:source));
            using var stillReadable=source.ReadTensor(values.Keys.First());Assert.False(stillReadable.IsInvalid);
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Orphan_supplemental_keys_do_not_claim_an_adapter_algorithm()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            var values=new[]{".lokr_w1_b",".lokr_w2_b",".lokr_t2",".rescale",".oft_blocks"}
                .ToDictionary(s=>"diffusion_model.out.2"+s,s=>ones([1,1]));
            using var source=new NativeLoraTensorSource(values);
            using var resumed=new SdTrainableAdapterSet(config,2,317,CPU,existing:source);
            using var fresh=new SdTrainableAdapterSet(config,2,317,CPU);
            Assert.Empty(resumed.ResumedTargets);Assert.Equal(5,resumed.IgnoredExistingKeys.Count);
            Assert.Equal(Digest(fresh),Digest(resumed));Assert.Equal(fresh.InitialCpuRandomStateSha256,resumed.InitialCpuRandomStateSha256);
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
