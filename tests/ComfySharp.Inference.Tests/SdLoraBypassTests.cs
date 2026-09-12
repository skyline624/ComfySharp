using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdLoraBypassTests
{
    [Theory]
    [InlineData("time_embed.0", false)]
    [InlineData("input_blocks.1.1.transformer_blocks.0.attn1.to_q", false)]
    [InlineData("input_blocks.1.1.proj_in", false)]
    [InlineData("input_blocks.1.1.proj_in", true)]
    [InlineData("input_blocks.3.0.op", false)]
    [InlineData("out.2", false)]
    public void Layer_hooks_change_prediction_without_allocating_patched_weights_and_survive_clones(string prefix, bool linear)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope();
            var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, linear);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var original = new SdUnet(bank);
            var shape = UnetWeightSchema.Describe(config)[prefix + ".weight"];
            using var patch = new LoraWeightPatch(full(new long[] { shape[0], 2 }, .03f), full(new long[] { 2, shape.Skip(1).Aggregate(1L, (a,b) => a*b) }, .02f), alpha: 2);
            var patches = new Dictionary<string, LoraWeightPatch> { [prefix + ".weight"] = patch };
            Assert.Throws<NotSupportedException>(() => original.WithLora(patches, maxPatchedWeightBytes: 0));
            using var bypass = original.WithBypassLora(patches, maxPatchedWeightBytes: 0);
            using var retained = bypass.Retain(); using var moved = bypass.To(CPU); bypass.Dispose(); patch.Dispose();
            using var input = NativeMath.CpuNoise([1,4,8,8], 51); using var context = NativeMath.CpuNoise([1,3,16], 52); using var time = tensor(new[] { 17.25f });
            using var baseline = original.Forward(input, time, context); using var changed = retained.Forward(input, time, context);
            using var again = moved.Forward(input, time, context); using var unchanged = original.Forward(input, time, context);
            Assert.NotEqual(baseline.bytes.ToArray(), changed.bytes.ToArray());
            Assert.Equal(changed.bytes.ToArray(), again.bytes.ToArray()); Assert.Equal(baseline.bytes.ToArray(), unchanged.bytes.ToArray());
            Assert.Throws<OperationCanceledException>(() => retained.Forward(input, time, context, new(true)));
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Replacing_bypass_group_and_adding_regular_differences_follow_separate_paths()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope();
            var config = new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var original = new SdUnet(bank);
            using var first = new LoraWeightPatch(full(new long[] {4,1}, .1f), full(new long[] {1,288}, .1f));
            using var second = new LoraWeightPatch(full(new long[] {128,1}, .02f), full(new long[] {1,32}, .03f));
            using var diff = LoraWeightPatch.FromDifference(full(new long[] {4}, .01f));
            using var one = original.WithBypassLora(new Dictionary<string,LoraWeightPatch>{{"out.2.weight", first}}, 0);
            var newer = new Dictionary<string,LoraWeightPatch>{{"time_embed.0.weight", second}};
            using var replaced = one.WithBypassLora(newer, 0); using var expected = original.WithBypassLora(newer, 0);
            var regular = new Dictionary<string,LoraWeightPatch>{{"out.2.bias", diff}};
            Assert.Throws<NotSupportedException>(() => one.WithBypassLora(regular, 0));
            using var withDiff = one.WithBypassLora(regular, 16); using var ordinaryDiff = one.WithLora(regular, 16);
            using var input = NativeMath.CpuNoise([1,4,8,8],51); using var context = NativeMath.CpuNoise([1,3,16],52); using var time = tensor(new[]{17.25f});
            using var a = replaced.Forward(input,time,context); using var b = expected.Forward(input,time,context);
            using var c = withDiff.Forward(input,time,context); using var d = ordinaryDiff.Forward(input,time,context);
            Assert.Equal(a.bytes.ToArray(),b.bytes.ToArray()); Assert.Equal(c.bytes.ToArray(),d.bytes.ToArray());
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
