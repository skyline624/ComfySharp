using ComfySharp.RuntimeProbe;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class FrozenAdapterEvaluationTests
{
    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void Evaluation_preserves_values_and_restores_mixed_leaf_flags_on_success_or_failure(bool fail)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var first=ones(2,2);using var second=ones(3,3);using var owner=new TrainableLokrPatch(first,second,1);
            owner.NamedParameters["alpha"].requires_grad_(false);
            var leaves=owner.Parameters.ToArray();var flags=leaves.Select(p=>p.requires_grad).ToArray();var values=leaves.Select(p=>p.bytes.ToArray()).ToArray();
            using var enabled=set_grad_enabled(true);
            Tensor Evaluate()
            {
                Assert.All(leaves,p=>Assert.False(p.requires_grad));
                using var probe=ones(1,requires_grad:true);using var product=probe*2;Assert.False(product.requires_grad);
                if(fail)throw new OperationCanceledException("Diagnostic interrupted.");
                return leaves[0].clone();
            }
            if(fail)Assert.Throws<OperationCanceledException>(()=>FrozenAdapterEvaluation.Run([owner,owner],Evaluate));
            else{using var result=FrozenAdapterEvaluation.Run([owner,owner],Evaluate);Assert.Equal(values[0],result.bytes.ToArray());}
            Assert.Equal(flags,leaves.Select(p=>p.requires_grad));
            for(int i=0;i<leaves.Length;i++)Assert.Equal(values[i],leaves[i].bytes.ToArray());
            using var active=ones(1,requires_grad:true);using var activeProduct=active*2;Assert.True(activeProduct.requires_grad);
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
