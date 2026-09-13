using System.Buffers.Binary;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LokrBypassTests
{
    public static IEnumerable<object[]> Cases=>Enumerable.Range(0,125).SelectMany(i=>new[]{new object[]{i,false},new object[]{i,true}});
    private static JsonDocument Fixture()
    {
        var compressed=ClipReferenceTests.Resource("lokr-bypass.reference.json.gz");
        Assert.Equal("1adbce39371b81dcd276d52debad4dc18b63a8794acaa88f8075486e0ad2017a",Convert.ToHexStringLower(SHA256.HashData(compressed)));
        using var input=new MemoryStream(compressed);using var gzip=new GZipStream(input,CompressionMode.Decompress);using var output=new MemoryStream();gzip.CopyTo(output);
        var bytes=output.ToArray();Assert.Equal("f0259fd9635ea6bc3833d74c7786915c33f8f35226c363bd053106af9c76dfe0",Convert.ToHexStringLower(SHA256.HashData(bytes)));return JsonDocument.Parse(bytes);
    }
    private static float[] Values(JsonElement e,JsonElement root)
    {
        string hash=e.GetProperty("sha256").GetString()!;var bytes=Convert.FromBase64String(root.GetProperty("payloads").GetProperty(hash).GetString()!);
        Assert.Equal(hash,Convert.ToHexStringLower(SHA256.HashData(bytes)));var values=new float[bytes.Length/4];
        for(int i=0;i<values.Length;i++)values[i]=BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i*4,4));return values;
    }
    private static Tensor Read(JsonElement e,JsonElement root)=>tensor(Values(e,root),e.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray());
    private static void Near(Tensor actual,JsonElement expected,JsonElement root,string label)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()),actual.shape);
        var wanted=Values(expected,root);var values=actual.data<float>().ToArray();
        for(int i=0;i<wanted.Length;i++)Assert.True(float.IsFinite(values[i])&&Math.Abs(values[i]-wanted[i])<=3e-5+3e-5*Math.Abs(wanted[i]),$"{label}[{i}]: {values[i]} vs {wanted[i]}");
    }
    private static void SourceError(Exception error,JsonElement source,bool training)
    {
        string type=source.GetProperty("type").GetString()!;
        if(type=="ZeroDivisionError")Assert.IsType<DivideByZeroException>(error);
        else if(training&&type=="ValueError")Assert.IsType<ArgumentException>(error);
        else Assert.True(error is ArgumentException or System.Runtime.InteropServices.ExternalException,$"Unexpected {error.GetType().FullName}: {error.Message}");
    }
    [Theory] [MemberData(nameof(Cases))]
    public void Grouped_operators_gradients_and_updates_match_frozen_h(int index,bool training)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var fixture=Fixture();var root=fixture.RootElement;var row=root.GetProperty("cases")[index];var expected=row.GetProperty(training?"training":"inference");
            Assert.Equal(3e-5,root.GetProperty("absoluteTolerance").GetDouble());Assert.Equal(3e-5,root.GetProperty("relativeTolerance").GetDouble());
            var factors=row.GetProperty("factors").EnumerateObject().ToDictionary(p=>p.Name,p=>Read(p.Value,root));
            using var input=Read(row.GetProperty("input"),root).requires_grad_();using var weight=Read(row.GetProperty("weight"),root);
            int dims=row.GetProperty("dims").GetInt32();long stride=row.GetProperty("stride").GetInt64(),padding=row.GetProperty("padding").GetInt64();
            long[]? kernel=dims==0?null:row.GetProperty("kernel").EnumerateArray().Select(v=>v.GetInt64()).ToArray();
            // Base uses only ordinary module operators; expected h is independent frozen Python AST.
            using var baseOutput=LokrBypassMath.Op(input,weight,dims,stride,padding);byte[] borrowed=baseOutput.bytes.ToArray();
            double? alpha=row.GetProperty("alpha").ValueKind==JsonValueKind.Null?null:row.GetProperty("alpha").GetDouble();
            var error=expected.GetProperty("error");
            if(training)
            {
                TrainableLokrPatch Create()=>new(factors.GetValueOrDefault("lokr_w1"),factors.GetValueOrDefault("lokr_w2"),alpha??1,
                    factors.GetValueOrDefault("lokr_w1_a"),factors.GetValueOrDefault("lokr_w1_b"),factors.GetValueOrDefault("lokr_w2_a"),factors.GetValueOrDefault("lokr_w2_b"),factors.GetValueOrDefault("lokr_t2"));
                if(error.ValueKind!=JsonValueKind.Null)
                {
                    long current=Tensor.TotalCount;
                    var failure=Assert.ThrowsAny<Exception>(()=>{using var owner=Create();using var ignored=owner.ApplyBypass(input,baseOutput,kernel,stride,padding);});
                    SourceError(failure,error,true);Assert.Equal(current,Tensor.TotalCount);
                }
                else
                {
                    using var patch=Create();using var retained=patch.Retain();patch.Dispose();foreach(var value in factors.Values)value.fill_(99);
                    using var optimizer=new LoraTrainingOptimizer(new[]{retained},"SGD",.003);
                    using var result=retained.ApplyBypass(input,baseOutput,kernel,stride,padding);Near(result,expected.GetProperty("output"),root,"train output");
                    using var loss=result.square().mean();optimizer.Accumulate(loss);Assert.InRange(Math.Abs(loss.item<float>()-expected.GetProperty("loss").GetSingle()),0,3e-5);
                    using var dx=input.grad;Assert.NotNull(dx);Near(dx!,expected.GetProperty("inputGradient"),root,"train input gradient");
                    foreach(var(name,value)in retained.NamedParameters)
                    {
                        Near(value,expected.GetProperty("initial").GetProperty(name),root,"initial "+name);
                        using var gradient=value.grad;var wanted=expected.GetProperty("gradients").GetProperty(name);
                        if(wanted.ValueKind==JsonValueKind.Null)Assert.Null(gradient);else{Assert.NotNull(gradient);Near(gradient!,wanted,root,"parameter gradient "+name);}
                    }
                    optimizer.Step();foreach(var(name,value)in retained.NamedParameters)Near(value,expected.GetProperty("updated").GetProperty(name),root,"updated "+name);
                }
            }
            else
            {
                double strength=row.GetProperty("strength").GetDouble();
                using var patch=LoraWeightPatch.FromLokr(factors.Where(p=>p.Key!="dora_scale").ToDictionary(),strength,alpha,factors["dora_scale"]);
                using var retained=patch.Retain();using var moved=patch.To(CPU);patch.Dispose();foreach(var value in factors.Values)value.fill_(99);
                if(error.ValueKind!=JsonValueKind.Null)
                {
                    long current=Tensor.TotalCount;var failure=Assert.ThrowsAny<Exception>(()=>{using var ignored=moved.ApplyBypass(input,baseOutput,kernel,stride,padding);});
                    SourceError(failure,error,false);Assert.Equal(current,Tensor.TotalCount);
                }
                else
                {
                    using var result=moved.ApplyBypass(input,baseOutput,kernel,stride,padding);Near(result,expected.GetProperty("output"),root,"inference output");
                    using var again=retained.ApplyBypass(input,baseOutput,kernel,stride,padding);Assert.Equal(result.bytes.ToArray(),again.bytes.ToArray());
                    result.square().mean().backward();using var dx=input.grad;Assert.NotNull(dx);Near(dx!,expected.GetProperty("inputGradient"),root,"inference input gradient");
                }
            }
            Assert.Equal(borrowed,baseOutput.bytes.ToArray());
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Fact]
    public void Unet_bypass_trains_linear_and_convolution_factors_without_materializing_weights()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;int threads=get_num_threads();set_num_threads(1);
        try
        {
            using var scope=NewDisposeScope();var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var bank=SdSyntheticInputs.CreateUnet(config);using var model=new SdUnet(bank);
            using var conv=new TrainableLokrPatch(zeros(2,2),ones(2,16,3,3)*.01);
            using var linear=new TrainableLokrPatch(zeros(2,2),ones(16,16)*.01);
            var patches=new Dictionary<string,TrainableWeightPatch>{{"out.2.weight",conv},{"input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",linear}};
            using var input=NativeMath.CpuNoise([1,4,8,8],511);using var context=NativeMath.CpuNoise([1,3,16],512);using var time=tensor(new[]{17.25f});
            using var baseline=model.Forward(input,time,context);using var optimizer=new LoraTrainingOptimizer(patches.Values,"SGD",.01);
            for(int step=0;step<2;step++)
            {
                using var iteration=NewDisposeScope();using var output=model.ForwardForTraining(input,time,context,patches,maxPatchedWeightBytes:0,bypassMode:true);
                if(step==0)Assert.Equal(baseline.bytes.ToArray(),output.bytes.ToArray());
                using var loss=output.square().mean();optimizer.Accumulate(loss);
                foreach(var patch in new[]{conv,linear})foreach(var(name,value)in patch.NamedParameters)
                {
                    using var gradient=value.grad;
                    if(name=="alpha")Assert.Null(gradient);
                    else{Assert.NotNull(gradient);Assert.True(gradient!.isfinite().all().item<bool>());if(step==1)Assert.True(gradient.abs().sum().item<float>()>0);}
                }
                optimizer.Step();
            }
            using var trained=model.ForwardForTraining(input,time,context,patches,maxPatchedWeightBytes:0,bypassMode:true);
            using var diagnostic=model.ForwardTrainingDiagnostic(input,time,context,patches,0,default,true,false);
            Assert.True(trained.requires_grad);Assert.False(diagnostic.requires_grad);
            Assert.True(allclose(trained,diagnostic,rtol:3e-5,atol:3e-5));
            using var reference=ComfySharp.RuntimeProbe.FrozenAdapterEvaluation.Run(patches.Values,
                ()=>model.ForwardTrainingDiagnostic(input,time,context,patches,0,default,true,false));
            Assert.All(patches.Values.SelectMany(p=>p.Parameters),p=>Assert.True(p.requires_grad));
            using var state=LoraTrainingState.Capture(patches,ScalarType.Float32);using var source=new NativeLoraTensorSource(state.Tensors);
            var plan=LoraFileLoader.Inspect(source,LoraModelAliases.ForUnet(config));using var frozen=LoraFileLoader.Load(source,plan);
            using var inference=frozen.ApplyBypassTo(model,maxPatchedWeightBytes:0);source.Dispose();state.Dispose();conv.Dispose();linear.Dispose();
            using var reloaded=inference.Forward(input,time,context);Assert.Equal(reference.bytes.ToArray(),reloaded.bytes.ToArray());
            Assert.True(allclose(trained,reloaded,rtol:3e-5,atol:3e-5));
            using var unchanged=model.Forward(input,time,context);Assert.Equal(baseline.bytes.ToArray(),unchanged.bytes.ToArray());
        }
        finally{set_num_threads(threads);}
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Fact]
    public void Spatial_second_factor_loads_for_bypass_and_does_not_claim_weight_reconstruction()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            var values=new Dictionary<string,Tensor>{{"layer.lokr_w1",ones(2,2)*.02},{"layer.lokr_w2_a",ones(3,1)*.03},{"layer.lokr_w2_b",ones(1,2,2,2)*.04}};
            using var source=new NativeLoraTensorSource(values);var plan=LoraFileLoader.Inspect(source,[new("layer",new("model","layer.weight",new long[]{6,4,2,2}))]);
            using var adapters=LoraFileLoader.Load(source,plan);using var weight=zeros(6,4,2,2);
            var error=Assert.ThrowsAny<Exception>(()=>{using var ignored=adapters.Apply("model","layer.weight",weight);});Assert.Contains("matrix",error.Message,StringComparison.OrdinalIgnoreCase);
            using var patch=LoraWeightPatch.FromLokr(values.ToDictionary(p=>p.Key[6..],p=>p.Value));
            using var input=ones(1,4,5,5);using var baseline=zeros(1,6,4,4);using var bypass=patch.ApplyBypass(input,baseline,new long[]{2,2});
            Assert.True(bypass.isfinite().all().item<bool>());Assert.True(bypass.abs().sum().item<float>()>0);
            Assert.Throws<OperationCanceledException>(()=>LokrBypassMath.Apply(input,baseline,values,cancellationToken:new(true)));
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
