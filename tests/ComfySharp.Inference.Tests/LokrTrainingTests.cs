using System.Security.Cryptography;
using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LokrTrainingTests
{
    private static JsonDocument Factory()
    {
        var compressed=ClipReferenceTests.Resource("lokr-factory.reference.json.gz");
        Assert.Equal("ab6d955a8c35e59d2c47ba2aab2768799e75861482dabf316047d590588bfa5a",Convert.ToHexStringLower(SHA256.HashData(compressed)));
        using var input=new MemoryStream(compressed);using var gzip=new GZipStream(input,CompressionMode.Decompress);using var decoded=new MemoryStream();gzip.CopyTo(decoded);
        var bytes=decoded.ToArray();Assert.Equal("216fe9c36089b5e474fec5200a64cc35f40e9a0f2de6b1e3ea7545356349f2b9",Convert.ToHexStringLower(SHA256.HashData(bytes)));
        return JsonDocument.Parse(bytes);
    }
    [Fact]
    public void Factorization_preserves_source_divisors_and_rank_limits()
    {
        using var doc=Factory();var rows=doc.RootElement.GetProperty("factorizations");Assert.Equal(72,rows.GetArrayLength());
        foreach(var row in rows.EnumerateArray())
        {
            var pair=TrainableLokrPatch.Factorize(row.GetProperty("dimension").GetInt64(),row.GetProperty("factor").GetInt32());
            Assert.Equal(row.GetProperty("result")[0].GetInt64(),pair.Small);Assert.Equal(row.GetProperty("result")[1].GetInt64(),pair.Large);
        }
    }
    [Theory] [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void All_686_factory_targets_and_rng_match_source(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var fixture=Factory();var root=fixture.RootElement;var row=root.GetProperty("cases")[index];bool linear=row.GetProperty("linear").GetBoolean();
            Assert.Equal("base64-f32-little-endian",root.GetProperty("payloadEncoding").GetString());
            var config=new SdUnetConfig(32,16,linear?SdAttentionHeadMode.FixedSize:SdAttentionHeadMode.FixedCount,linear?8:4,linear);
            int rank=row.GetProperty("rank").GetInt32();long bytes=row.GetProperty("parameterBytes").GetInt64();
            Assert.Throws<NotSupportedException>(()=>new SdTrainableAdapterSet(config,rank,317,CPU,bytes-1,algorithm:"LoKr"));
            using var global=manual_seed(117);using var prior=global.get_state();
            using var adapters=new SdTrainableAdapterSet(config,rank,317,CPU,bytes,algorithm:"LoKr");using var after=global.get_state();
            Assert.Equal(prior.bytes.ToArray(),after.bytes.ToArray());Assert.Equal(row.GetProperty("randomStateSha256").GetString(),adapters.InitialCpuRandomStateSha256);
            Assert.Equal(686,adapters.Patches.Count);Assert.Equal(1250,adapters.Patches.Values.Sum(p=>p.Parameters.Count));Assert.Equal(bytes,adapters.ParameterBytes);
            Assert.Equal(row.GetProperty("targets").EnumerateObject().Select(p=>p.Name),adapters.Patches.Keys);
            foreach(var(name,patch)in adapters.Patches)
            {
                var expected=row.GetProperty("targets").GetProperty(name);
                IReadOnlyDictionary<string,Tensor> parameters=patch is TrainableLokrPatch lokr?lokr.NamedParameters:
                    new Dictionary<string,Tensor>{{"bias",Assert.IsType<TrainableDifferencePatch>(patch).Difference}};
                Assert.Equal(expected.EnumerateObject().Select(p=>p.Name),parameters.Keys);
                foreach(var(key,value)in parameters)
                {
                    var e=expected.GetProperty(key);Assert.Equal(e.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()),value.shape);
                    string hash=e.GetProperty("sha256").GetString()!;byte[] raw=Convert.FromBase64String(root.GetProperty("payloads").GetProperty(hash).GetString()!);
                    Assert.Equal(hash,Convert.ToHexStringLower(SHA256.HashData(raw)));Assert.Equal(value.numel()*4,raw.Length);
                    var observed=value.data<float>().ToArray();for(int i=0;i<observed.Length;i++)
                    {
                        float v=BinaryPrimitives.ReadSingleLittleEndian(raw.AsSpan(i*4,4));
                        Assert.True(float.IsFinite(observed[i])&&Math.Abs(observed[i]-v)<=3e-5+3e-5*Math.Abs(v),$"{name}/{key}[{i}]");
                    }
                }
            }
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Theory] [InlineData(false)] [InlineData(true)]
    public void Unet_reconstruction_keeps_base_and_trains_factors_without_substituting_bypass(bool rebuilt)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;int threads=get_num_threads();set_num_threads(1);
        try
        {
            using var scope=NewDisposeScope();var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var bank=SdSyntheticInputs.CreateUnet(config);using var model=new SdUnet(bank);
            using var patch=new TrainableLokrPatch(rebuilt?null:ones([2,2])*.1,ones([2,16,3,3])*.1,1.75,
                rebuilt?ones([2,1])*.2:null,rebuilt?ones([1,2])*.3:null);
            var patches=new Dictionary<string,TrainableWeightPatch>{{"out.2.weight",patch}};
            using var input=NativeMath.CpuNoise([1,4,8,8],511);using var context=NativeMath.CpuNoise([1,3,16],512);using var time=tensor(new[]{17.25f});
            using var baseline=model.Forward(input,time,context);using var optimizer=new LoraTrainingOptimizer(patches.Values,"SGD",.01);
            using var prediction=model.ForwardForTraining(input,time,context,patches);using var loss=prediction.square().mean();optimizer.Accumulate(loss);
            foreach(var(name,value)in patch.NamedParameters)
            {
                using var gradient=value.grad;if(name=="alpha"&&!rebuilt)Assert.Null(gradient);
                else{Assert.NotNull(gradient);Assert.True(gradient!.isfinite().all().item<bool>());Assert.True(gradient.abs().sum().item<float>()>0);}
            }
            optimizer.Step();using var changed=model.ForwardForTraining(input,time,context,patches);Assert.NotEqual(prediction.bytes.ToArray(),changed.bytes.ToArray());
            Assert.Throws<NotSupportedException>(()=>model.ForwardForTraining(input,time,context,patches,bypassMode:true));
            using var unchanged=model.Forward(input,time,context);Assert.Equal(baseline.bytes.ToArray(),unchanged.bytes.ToArray());
        }
        finally{set_num_threads(threads);}
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Fact]
    public void Invalid_geometry_and_cancellation_leave_no_native_resources()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            Assert.Throws<ArgumentException>(()=>new TrainableLokrPatch(null,ones([2,3])));
            Assert.Throws<ArgumentException>(()=>new TrainableLokrPatch(null,ones([2,3]),w1a:ones([2,4]),w1b:ones([3,2])));
            using var patch=new TrainableLokrPatch(ones([2,2]),ones([3,4]));
            Assert.Throws<ArgumentException>(()=>patch.Apply(ones([3,4])));
            Assert.Throws<OperationCanceledException>(()=>patch.Apply(ones([6,8]),new(true)));
            Assert.Throws<ArgumentException>(()=>new LoraTrainingOptimizer(new[]{patch,patch},"SGD",.01));
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
    public static IEnumerable<object[]> Cases=>Enumerable.Range(0,40).Select(i=>new object[]{i});
    private static JsonDocument Fixture()
    {
        var bytes=ClipReferenceTests.Resource("lokr-training.reference.json");
        Assert.Equal("46856bedcf23b203e3ac1344e700f164cd7ef4692d5aaa1ba226d54f0c7c6e75",Convert.ToHexStringLower(SHA256.HashData(bytes)));
        return JsonDocument.Parse(bytes);
    }
    private static Tensor Read(JsonElement e)=>tensor(e.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray(),e.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray());
    private static TrainableLokrPatch Create(JsonElement initial)
    {
        Tensor? Get(string name)=>initial.TryGetProperty(name,out var e)?Read(e):null;
        return new(Get("lokr_w1"),Get("lokr_w2"),initial.GetProperty("alpha").GetProperty("values")[0].GetSingle(),
            Get("lokr_w1_a"),Get("lokr_w1_b"),Get("lokr_w2_a"),Get("lokr_w2_b"),Get("lokr_t2"));
    }
    private static void Near(Tensor actual,JsonElement expected,string name)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()),actual.shape);
        var values=expected.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray();var observed=actual.data<float>().ToArray();
        for(int i=0;i<values.Length;i++)Assert.True(float.IsFinite(observed[i])&&Math.Abs(observed[i]-values[i])<=3e-5+3e-5*Math.Abs(values[i]),$"{name}[{i}]: {observed[i]} vs {values[i]}");
    }
    [Theory] [MemberData(nameof(Cases))]
    public void Reconstruction_alpha_gradients_and_updates_follow_frozen_LokrDiff(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var fixture=Fixture();var root=fixture.RootElement;Assert.Equal(3e-5,root.GetProperty("absoluteTolerance").GetDouble());Assert.Equal(3e-5,root.GetProperty("relativeTolerance").GetDouble());
            var row=root.GetProperty("cases")[index];var initial=row.GetProperty("initial");
            using var patch=Create(initial);using var retained=patch.Retain();patch.Dispose();Assert.Throws<ObjectDisposedException>(()=>patch.Parameters);
            Assert.Equal(initial.EnumerateObject().Select(p=>p.Name),retained.NamedParameters.Keys);
            using var weight=Read(row.GetProperty("weight"));byte[] original=weight.bytes.ToArray();using var target=Read(row.GetProperty("target"));
            if(row.GetProperty("error").ValueKind!=JsonValueKind.Null)
            {
                // Frozen Tucker output layout can make torch.kron reject a view.
                // Preserve the observed failure, never silently insert contiguous().
                var error=Assert.ThrowsAny<Exception>(()=>retained.Apply(weight));Assert.Contains("view size is not compatible",error.Message);
            }
            else
            {
                using var optimizer=new LoraTrainingOptimizer(new[]{retained},row.GetProperty("optimizer").GetString()!,.003);
                foreach(var step in row.GetProperty("steps").EnumerateArray())
                {
                    using var iteration=NewDisposeScope();using var output=retained.Apply(weight);Near(output,step.GetProperty("output"),"output");
                    using var loss=TrainingLoss.Calculate("MSE",output,target);Assert.InRange(Math.Abs(loss.item<float>()-step.GetProperty("loss").GetSingle()),0,3e-5);
                    optimizer.Accumulate(loss);
                    foreach(var(name,value) in retained.NamedParameters)
                    {
                        Assert.True(value.requires_grad);using var gradient=value.grad;var expected=step.GetProperty("gradients").GetProperty(name);
                        if(expected.ValueKind==JsonValueKind.Null)Assert.Null(gradient);else{Assert.NotNull(gradient);Near(gradient!,expected,"gradient/"+name);}
                    }
                    optimizer.Step();foreach(var(name,value)in retained.NamedParameters)Near(value,step.GetProperty("updated").GetProperty(name),"updated/"+name);
                }
            }
            Assert.Equal(original,weight.bytes.ToArray());
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Theory] [InlineData(0)] [InlineData(12)] [InlineData(20)] [InlineData(36)]
    public void Snapshot_and_safetensors_preserve_keys_dtypes_and_independent_ownership(int index)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        string path=Path.Combine(Path.GetTempPath(),"comfysharp-lokr-"+Guid.NewGuid().ToString("N")+".safetensors");
        try
        {
            using var scope=NewDisposeScope();using var fixture=Fixture();var initial=fixture.RootElement.GetProperty("cases")[index].GetProperty("initial");
            using var patch=Create(initial);var targets=new Dictionary<string,TrainableLokrPatch>{{"layer.weight",patch}};long bytes=patch.Parameters.Sum(p=>p.numel())*4;
            Assert.Throws<NotSupportedException>(()=>LoraTrainingState.Capture(targets,ScalarType.Float32,maxSnapshotBytes:bytes-1));
            Assert.Throws<NotSupportedException>(()=>LoraTrainingFile.SaveTargetsNew(path,targets,maxFactorBytes:bytes-1));Assert.False(File.Exists(path));
            using var state=LoraTrainingState.Capture(targets,ScalarType.Float32,maxSnapshotBytes:bytes);
            using var bf16=LoraTrainingState.Capture(targets,ScalarType.BFloat16,maxSnapshotBytes:bytes/2);
            LoraTrainingFile.SaveTargetsNew(path,targets,maxFactorBytes:bytes);Assert.Throws<IOException>(()=>LoraTrainingFile.SaveTargetsNew(path,targets));
            using(var noGrad=no_grad())foreach(var value in patch.Parameters)value.fill_(99);patch.Dispose();
            using var file=new SafeTensorFile(path);
            foreach(var entry in initial.EnumerateObject())
            {
                string key="diffusion_model.layer."+entry.Name;using var value=file.ReadTensor(key);Near(value,entry.Value,key);
                Assert.Equal(state.Tensors[key].bytes.ToArray(),value.bytes.ToArray());using var cast=value.to_type(ScalarType.BFloat16);
                Assert.Equal(cast.bytes.ToArray(),bf16.Tensors[key].bytes.ToArray());Assert.False(state.Tensors[key].requires_grad);
            }
        }
        finally{if(File.Exists(path))File.Delete(path);}
        Assert.Equal(before,Tensor.TotalCount);
    }
}
