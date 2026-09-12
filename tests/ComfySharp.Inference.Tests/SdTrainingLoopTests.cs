using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdTrainingLoopTests
{
    [Theory]
    [InlineData(SdTrainingDatasetMode.Standard)]
    [InlineData(SdTrainingDatasetMode.MultiResolution)]
    [InlineData(SdTrainingDatasetMode.Buckets)]
    public void Selected_dataset_groups_train_and_release_without_mutating_the_base(SdTrainingDatasetMode mode)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope();
            var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank); using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
            using var first = NativeMath.CpuNoise([2, 4, 8, 8], 31); using var second = NativeMath.CpuNoise([1, 4, mode == SdTrainingDatasetMode.Standard ? 8 : 16, 8], 32);
            using var dataset = new SdTrainingDataset([first, second], mode == SdTrainingDatasetMode.Buckets);
            using var context = NativeMath.CpuNoise([3, 3, 16], 33);
            using var patch = new TrainableLoraPatch(ones([4, 1]) * .01, ones([1, 288]) * .01, 1);
            var patches = new Dictionary<string, TrainableLoraPatch> { ["out.2.weight"] = patch };
            using var input = first.narrow(0, 0, 1); using var condition = context.narrow(0, 0, 1); using var sigma = tensor(new[] { .5f });
            using var original = denoiser.Denoise(input, sigma, condition); var originalFactors = patch.Up.data<float>().ToArray();
            var events = new List<SdLoraTrainingProgress>();
            var result = SdLoraTrainingLoop.Run(denoiser, dataset, context, patches, new() { Seed = 41, Steps = 2, BatchSize = 3, AccumulationSteps = 2 }, events.Add);
            Assert.Equal(2, result.OptimizerSteps); Assert.Equal(4, result.Microbatches); Assert.Equal(4, result.Losses.Count);
            Assert.All(result.Losses, value => Assert.True(float.IsFinite(value) && value >= 0));
            Assert.Equal(new long[] { 0, 1, 1, 2 }, events.Select(e => e.OptimizerSteps));
            Assert.NotEqual(originalFactors, patch.Up.data<float>().ToArray()); Assert.Null(patch.Up.grad);
            using var unchanged = denoiser.Denoise(input, sigma, condition); Assert.Equal(original.data<float>().ToArray(), unchanged.data<float>().ToArray());
            Assert.Null(context.grad);
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Cancelling_before_the_first_update_discards_accumulation()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var cancel = new CancellationTokenSource();
        var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
        using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank); using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        using var dataset = new SdTrainingDataset([ones([2, 4, 8, 8])]);
        using var patch = new TrainableLoraPatch(ones([4, 1]) * .01, ones([1, 288]) * .01, 1);
        var original = patch.Up.data<float>().ToArray();
        Assert.Throws<OperationCanceledException>(() => SdLoraTrainingLoop.Run(denoiser, dataset, zeros([1, 3, 16]), new Dictionary<string, TrainableLoraPatch> { ["out.2.weight"] = patch },
            new() { Steps = 2, AccumulationSteps = 2 }, _ => cancel.Cancel(), cancel.Token));
        Assert.Equal(original, patch.Up.data<float>().ToArray()); Assert.Null(patch.Up.grad); Assert.Null(patch.Down.grad);
    }
}
