using Xunit;
using static TorchSharp.torch;
namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraWeightPatchTests
{
    [Fact]
    public void Patched_unet_shares_untouched_storage_and_predicts_after_parent_disposal()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;int threads=get_num_threads();set_num_threads(1);
        try
        {
            using var scope=NewDisposeScope();
            var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var source=SdSyntheticInputs.CreateUnet(config);using var original=new SdUnet(source);
            var up=full(new long[]{4,1},.125f);var down=full(new long[]{1,32,3,3},.01f);
            using var patch=new LoraWeightPatch(up,down);up.fill_(100);down.fill_(100);
            var patches=new Dictionary<string,LoraWeightPatch>{{"out.2.weight",patch}};
            using var bank=source.WithLora(patches);using var changed=new SdUnet(bank);
            Assert.Equal(Address(source.GetTensor("time_embed.0.weight")),Address(bank.GetTensor("time_embed.0.weight")));
            Assert.NotEqual(Address(source.GetTensor("out.2.weight")),Address(bank.GetTensor("out.2.weight")));
            var x=ones(new long[]{1,4,4,5});var t=tensor(new[]{500f});var context=ones(new long[]{1,3,16});
            using var baseline=original.Forward(x,t,context);using var prediction=changed.Forward(x,t,context);
            Assert.False(baseline.data<float>().ToArray().SequenceEqual(prediction.data<float>().ToArray()));
            using var again=original.Forward(x,t,context);Assert.Equal(baseline.data<float>().ToArray(),again.data<float>().ToArray());
            using var direct=original.WithLora(patches);patch.Dispose();bank.Dispose();original.Dispose();source.Dispose();
            using var survived=direct.Forward(x,t,context);Assert.Equal(prediction.data<float>().ToArray(),survived.data<float>().ToArray());
        }
        finally{set_num_threads(threads);}
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Patched_clip_preserves_projection_optional_contract_and_native_lifetimes()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;int threads=get_num_threads();set_num_threads(1);
        try
        {
            using var scope=NewDisposeScope();
            var config=new ClipTextConfig(4,8,2,1,ClipActivation.QuickGelu);
            var owned=ClipWeightSchema.Describe(config).ToDictionary(p=>p.Key,p=>full(p.Value.ToArray(),.1f));
            using var bank=ClipWeightSet.FromOwnedTensors(config,owned);using var model=new ClipTextEncoder(bank);
            using var patch=new LoraWeightPatch(ones(new long[]{4,1}),ones(new long[]{1,4}),.125);
            var patches=new Dictionary<string,LoraWeightPatch>{{ClipWeightSchema.Projection,patch}};
            using var changedBank=bank.WithLora(patches);using var changed=model.WithLora(patches);
            var untouched=ClipWeightSchema.Describe(config).Keys.First(k=>k!=ClipWeightSchema.Projection);
            Assert.Equal(Address(bank.GetTensor(untouched)),Address(changedBank.GetTensor(untouched)));
            int[] tokens=Enumerable.Repeat(1,77).ToArray();tokens[0]=49406;tokens[3]=49407;
            using var baseline=model.Forward([tokens]);using var prediction=changed.Forward([tokens]);
            Assert.False(baseline.ProjectedPooled!.data<float>().ToArray().SequenceEqual(prediction.ProjectedPooled!.data<float>().ToArray()));
            using var originalAgain=model.Forward([tokens]);Assert.Equal(baseline.ProjectedPooled.data<float>().ToArray(),originalAgain.ProjectedPooled!.data<float>().ToArray());
            bank.Dispose();model.Dispose();patch.Dispose();changedBank.Dispose();
            using var survived=changed.Forward([tokens]);Assert.Equal(prediction.ProjectedPooled.data<float>().ToArray(),survived.ProjectedPooled!.data<float>().ToArray());
            var noProjection=ClipWeightSchema.Describe(config,false).ToDictionary(p=>p.Key,p=>zeros(p.Value.ToArray()));
            using var optional=ClipWeightSet.FromOwnedTensors(config,noProjection,false);
            using var copied=optional.WithLora(new Dictionary<string,LoraWeightPatch>());Assert.False(copied.HasProjection);
        }
        finally{set_num_threads(threads);}
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void Partial_patch_failure_unknown_targets_and_budget_leave_original_bank_usable(bool clip)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var valid=new LoraWeightPatch(ones(new long[]{2,1}),ones(new long[]{1,2}));
            using var invalid=new LoraWeightPatch(ones(new long[]{7,1}),ones(new long[]{1,7}));
            if(clip)
            {
                var config=new ClipTextConfig(2,3,1,1,ClipActivation.QuickGelu);
                using var bank=ClipWeightSet.FromOwnedTensors(config,ClipWeightSchema.Describe(config).ToDictionary(p=>p.Key,p=>ones(p.Value.ToArray())));
                string first="text_model.encoder.layers.0.self_attn.q_proj.weight",second="text_model.encoder.layers.0.self_attn.k_proj.weight";
                Check((p,b,t)=>bank.WithLora(p,b,t),first,second,valid,invalid);
                Assert.All(bank.GetTensor(first).data<float>().ToArray(),v=>Assert.Equal(1,v));
            }
            else
            {
                var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
                using var bank=SdSyntheticInputs.CreateUnet(config);
                using var good=new LoraWeightPatch(ones(new long[]{128,1}),ones(new long[]{1,32}));
                Check((p,b,t)=>bank.WithLora(p,b,t),"time_embed.0.weight","time_embed.2.weight",good,invalid);
                Assert.Equal(new long[]{128,32},bank.GetTensor("time_embed.0.weight").shape);
            }
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    private static void Check(Func<IReadOnlyDictionary<string,LoraWeightPatch>,long,CancellationToken,IDisposable> bake,
        string first,string second,LoraWeightPatch valid,LoraWeightPatch invalid)
    {
        var good=new Dictionary<string,LoraWeightPatch>{{first,valid}};
        Assert.Throws<NotSupportedException>(()=>bake(good,0,default));
        Assert.ThrowsAny<OperationCanceledException>(()=>bake(good,long.MaxValue,new(true)));
        Assert.Throws<InvalidDataException>(()=>bake(new Dictionary<string,LoraWeightPatch>{{"missing.weight",valid}},long.MaxValue,default));
        Assert.Throws<ArgumentException>(()=>bake(new Dictionary<string,LoraWeightPatch>{{first,valid},{second,invalid}},long.MaxValue,default));
        using var okay=bake(good,long.MaxValue,default);
    }
    private static unsafe nint Address(Tensor tensor)
    {fixed(byte* address=tensor.bytes)return(nint)address;}
}
