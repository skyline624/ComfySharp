using System.Buffers.Binary;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdLokrResumeTests
{
    private static JsonDocument Fixture()
    {
        byte[] compressed=ClipReferenceTests.Resource("lokr-resume.reference.json.gz");
        Assert.Equal("fa21a17ee89d236f671c7bf7e9e4e0e7e635b74d1f13957631ad780222967f3c",Convert.ToHexStringLower(SHA256.HashData(compressed)));
        using var input=new MemoryStream(compressed);using var gzip=new GZipStream(input,CompressionMode.Decompress);using var output=new MemoryStream();gzip.CopyTo(output);
        var bytes=output.ToArray();Assert.Equal("c274f36962f53f14d745a23de19ea817f857d7dc532857f7e454f01485a4e470",Convert.ToHexStringLower(SHA256.HashData(bytes)));
        return JsonDocument.Parse(bytes);
    }
    private static float[] Values(JsonElement record,JsonElement root)
    {
        string hash=record.GetProperty("sha256").GetString()!;var bytes=Convert.FromBase64String(root.GetProperty("payloads").GetProperty(hash).GetString()!);
        Assert.Equal(hash,Convert.ToHexStringLower(SHA256.HashData(bytes)));var values=new float[bytes.Length/4];
        for(int i=0;i<values.Length;i++)values[i]=BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i*4,4));return values;
    }
    private static Tensor Read(JsonElement e,JsonElement root)
    {
        var dtype=e.GetProperty("dtype").GetString() switch {"torch.float16"=>ScalarType.Float16,"torch.bfloat16"=>ScalarType.BFloat16,"torch.float64"=>ScalarType.Float64,_=>ScalarType.Float32};
        return tensor(Values(e,root),e.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray()).to_type(dtype);
    }
    private static IReadOnlyDictionary<string,Tensor> Named(TrainableWeightPatch patch)=>patch switch
    {
        TrainableLokrPatch lokr=>lokr.NamedParameters,
        TrainableLohaPatch loha=>loha.NamedParameters,
        TrainableLoraPatch lora=>new Dictionary<string,Tensor>{{"alpha",lora.AlphaParameter!},{"lora_up.weight",lora.Up},{"lora_down.weight",lora.Down}},
        TrainableDifferencePatch diff=>new Dictionary<string,Tensor>{{"bias",diff.Difference}},
        _=>throw new NotSupportedException()
    };
    [Theory] [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void Complete_mixed_factory_parameters_and_rng_match_frozen_source(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        string path=Path.Combine(Path.GetTempPath(),"comfysharp-lokr-resume-"+Guid.NewGuid().ToString("N")+".safetensors");
        try
        {
            using var scope=NewDisposeScope();using var json=Fixture();var root=json.RootElement;var row=root.GetProperty("cases")[index];
            Assert.Equal("base64-f32-little-endian",root.GetProperty("payloadEncoding").GetString());
            Assert.Equal(3e-5,root.GetProperty("absoluteTolerance").GetDouble());Assert.Equal(3e-5,root.GetProperty("relativeTolerance").GetDouble());
            bool linear=row.GetProperty("linearProjection").GetBoolean();var config=new SdUnetConfig(32,16,linear?SdAttentionHeadMode.FixedSize:SdAttentionHeadMode.FixedCount,linear?8:4,linear);
            var values=row.GetProperty("existing").EnumerateObject().ToDictionary(p=>p.Name,p=>Read(p.Value,root));
            using(var output=File.Create(path))SafeTensorWriter.Write(output,values);
            using var source=new SafeTensorFile(path);string algorithm=row.GetProperty("algorithm").GetString()!;
            long bytes=row.GetProperty("parameterBytes").GetInt64();long borrowed=Tensor.TotalCount;
            Assert.Throws<NotSupportedException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,bytes-1,existing:source,algorithm:algorithm));Assert.Equal(borrowed,Tensor.TotalCount);
            using var global=manual_seed(117);using var prior=global.get_state();
            using var resumed=new SdTrainableAdapterSet(config,2,317,CPU,bytes,existing:source,algorithm:algorithm);using var after=global.get_state();
            Assert.Equal(prior.bytes.ToArray(),after.bytes.ToArray());source.Dispose();foreach(var v in values.Values){v.fill_(99);v.Dispose();}
            Assert.Equal(row.GetProperty("resumed").EnumerateArray().Select(v=>v.GetString()),resumed.ResumedTargets);
            Assert.Equal(bytes,resumed.ParameterBytes);Assert.Equal(row.GetProperty("parameterCount").GetInt32(),resumed.Patches.Values.Sum(p=>p.Parameters.Count));
            Assert.Equal(row.GetProperty("randomStateSha256").GetString(),resumed.InitialCpuRandomStateSha256);
            Assert.Equal(row.GetProperty("targets").EnumerateObject().Select(p=>p.Name),resumed.Patches.Keys);
            foreach(var(name,patch)in resumed.Patches)
            {
                var expected=row.GetProperty("targets").GetProperty(name);var named=Named(patch);Assert.Equal(expected.EnumerateObject().Select(p=>p.Name),named.Keys);
                foreach(var(key,value)in named)
                {
                    var record=expected.GetProperty(key);Assert.Equal(record.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()),value.shape);
                    Assert.True(value.requires_grad);Assert.Equal(ScalarType.Float32,value.dtype);var wanted=Values(record,root);var actual=value.data<float>().ToArray();
                    if(resumed.ResumedTargets.Contains(name))Assert.Equal(wanted,actual);
                    else for(int i=0;i<wanted.Length;i++)Assert.True(float.IsFinite(actual[i])&&Math.Abs(actual[i]-wanted[i])<=3e-5+3e-5*Math.Abs(wanted[i]),$"{name}/{key}[{i}] {actual[i]} vs {wanted[i]}");
                }
            }
            Assert.IsType<TrainableLoraPatch>(resumed.Patches["time_embed.0.weight"]);
            Assert.IsType<TrainableLohaPatch>(resumed.Patches["time_embed.2.weight"]);
            Assert.Contains("diffusion_model.time_embed.0.lokr_w1",resumed.IgnoredExistingKeys);
            Assert.Contains("diffusion_model.out.0.diff",resumed.IgnoredExistingKeys);Assert.Contains("diffusion_model.out.2.diff_b",resumed.IgnoredExistingKeys);
            Assert.Contains("diffusion_model.out.2.alpha",resumed.IgnoredExistingKeys);Assert.Contains("diffusion_model.out.2.lokr_w1_b",resumed.IgnoredExistingKeys);
            Assert.Contains("diffusion_model.out.2.lokr_t2",resumed.IgnoredExistingKeys);
            Assert.Equal(1,Assert.IsType<TrainableLokrPatch>(resumed.Patches["out.2.weight"]).NamedParameters["alpha"].item<float>());
        }
        finally{File.Delete(path);}
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Theory] [InlineData("direct")] [InlineData("first")] [InlineData("both")] [InlineData("tucker")]
    public void Resumed_targets_train_and_export_without_changing_base(string kind)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;int threads=get_num_threads();set_num_threads(1);
        try
        {
            using var scope=NewDisposeScope();var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var seedPatch=kind switch
            {
                "first"=>new TrainableLokrPatch(null,ones([2,16,3,3])*.01,w1a:ones([2,1])*.02,w1b:ones([1,2])*.03),
                "both"=>new TrainableLokrPatch(null,null,w1a:ones([2,2])*.02,w1b:ones([2,2])*.03,w2a:ones([2,3])*.04,w2b:ones([3,144])*.01),
                "tucker"=>new TrainableLokrPatch(ones([4,32])*.01,null,w2a:ones([2,1])*.02,w2b:ones([3,1])*.03,t2:ones([2,3,3,3])*.04),
                _=>new TrainableLokrPatch(ones([2,2])*.01,ones([2,16,3,3])*.02)
            };
            using var state=LoraTrainingState.Capture(new Dictionary<string,TrainableLokrPatch>{{"out.2.weight",seedPatch}},ScalarType.Float32);
            using var source=new NativeLoraTensorSource(state.Tensors);using var resumed=new SdTrainableAdapterSet(config,2,317,CPU,existing:source,algorithm:"LoKr");
            source.Dispose();state.Dispose();seedPatch.Dispose();Assert.Equal(new[]{"out.2.weight"},resumed.ResumedTargets);
            using var bank=SdSyntheticInputs.CreateUnet(config);using var model=new SdUnet(bank);
            using var x=NativeMath.CpuNoise([1,4,8,8],511);using var context=NativeMath.CpuNoise([1,3,16],512);using var time=tensor(new[]{17.25f});
            using var baseline=model.Forward(x,time,context);using var optimizer=new LoraTrainingOptimizer(resumed.Patches.Values,"SGD",.01);
            using var first=model.ForwardForTraining(x,time,context,resumed.Patches);using var loss=first.square().mean();optimizer.Accumulate(loss);
            foreach(var patch in resumed.Patches.Values)foreach(var value in patch.Parameters)
            {
                using var gradient=value.grad;
                bool directAlpha=patch is TrainableLokrPatch k&&k.NamedParameters.ContainsKey("lokr_w1")&&k.NamedParameters.ContainsKey("lokr_w2")&&ReferenceEquals(value,k.NamedParameters["alpha"]);
                if(directAlpha)Assert.Null(gradient);else{Assert.NotNull(gradient);Assert.True(gradient!.isfinite().all().item<bool>());}
            }
            optimizer.Step();using var changed=model.ForwardForTraining(x,time,context,resumed.Patches);Assert.NotEqual(first.bytes.ToArray(),changed.bytes.ToArray());
            using var snapshot=LoraTrainingState.Capture(resumed.Patches,ScalarType.Float32);resumed.Dispose();
            using var native=new NativeLoraTensorSource(snapshot.Tensors);var plan=LoraFileLoader.Inspect(native,LoraModelAliases.ForUnet(config));using var loaded=LoraFileLoader.Load(native,plan);
            using var baked=loaded.ApplyTo(model);using var reloaded=baked.Forward(x,time,context);
            if(kind=="both")Assert.NotEqual(changed.bytes.ToArray(),reloaded.bytes.ToArray()); // Source scales twice in training, once in inference.
            else Assert.True(allclose(changed,reloaded,rtol:3e-5,atol:3e-5));
            if(kind=="direct")Assert.Equal(changed.bytes.ToArray(),reloaded.bytes.ToArray());
            using var unchanged=model.Forward(x,time,context);Assert.Equal(baseline.bytes.ToArray(),unchanged.bytes.ToArray());
        }
        finally{set_num_threads(threads);}
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Theory] [InlineData("missing")] [InlineData("core")] [InlineData("shape")] [InlineData("integer")] [InlineData("nonfinite")]
    public void Invalid_resume_is_atomic_and_preserves_borrowed_source(string kind)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            const string prefix="diffusion_model.out.2";var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            var values=new Dictionary<string,Tensor>{{prefix+".lokr_w1",ones(2,2)},{prefix+".lokr_w2",ones(2,16,3,3)}};
            if(kind=="missing"){values.Remove(prefix+".lokr_w1");values[prefix+".lokr_w1_a"]=ones(2,2);}
            if(kind=="core"){values.Remove(prefix+".lokr_w2");values[prefix+".lokr_w2_a"]=ones(1,2);values[prefix+".lokr_w2_b"]=ones(1,16);values[prefix+".lokr_t2"]=ones(2,1,3,3);}
            if(kind=="shape")values[prefix+".lokr_w2"]=ones(2,17,3,3);
            if(kind=="integer")values[prefix+".lokr_w2"]=ones(2,16,3,3,dtype:ScalarType.Int32);
            if(kind=="nonfinite")values[prefix+".lokr_w2"].fill_(float.NaN);
            using var source=new NativeLoraTensorSource(values);long borrowed=Tensor.TotalCount;
            if(kind=="nonfinite")Assert.Throws<ArgumentException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,existing:source));
            else Assert.Throws<InvalidDataException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,existing:source));
            Assert.Equal(borrowed,Tensor.TotalCount);
            Assert.Throws<OperationCanceledException>(()=>new SdTrainableAdapterSet(config,2,317,CPU,existing:source,cancellationToken:new(true)));
            using var readable=source.ReadTensor(values.Keys.First());Assert.False(readable.IsInvalid);
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
