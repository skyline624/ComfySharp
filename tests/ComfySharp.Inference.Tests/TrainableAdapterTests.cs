using ComfySharp.Inference;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class TrainableAdapterTests
{
    [Fact]
    public void Difference_snapshots_are_owned_and_optimizer_retains_both_kinds()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            using var input = tensor(new[] { .25f, -.5f });
            using var diff = new TrainableDifferencePatch(input);
            input.fill_(99);
            using var lora = new TrainableLoraPatch(ones([2, 1]), zeros([1, 2]), 1);
            using var retained = diff.Retain();
            using var optimizer = new LoraTrainingOptimizer(new TrainableWeightPatch[] { lora, diff }, "SGD", .01);
            diff.Dispose();
            Assert.Equal(new[] { .25f, -.5f }, retained.Difference.data<float>().ToArray());
            using var value = retained.Apply(ones([2]));
            using var loss = value.square().mean() + lora.Up.sum() + lora.Down.sum();
            optimizer.Accumulate(loss); optimizer.Step();
            Assert.Equal(1, optimizer.CompletedSteps);
            Assert.NotEqual(new[] { .25f, -.5f }, retained.Difference.data<float>().ToArray());
            Assert.Null(retained.Difference.grad);
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Mixed_bias_norm_and_matrix_adapters_preserve_the_base_and_receive_gradients()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            using var lora = new TrainableLoraPatch(ones([4, 1]) * .01, zeros([1, 288]), 1);
            using var bias = new TrainableDifferencePatch(zeros([4]));
            using var norm = new TrainableDifferencePatch(zeros([32]));
            var patches = new Dictionary<string, TrainableWeightPatch> { ["out.2.weight"] = lora, ["out.2.bias"] = bias, ["out.0.weight"] = norm };
            using var input = NativeMath.CpuNoise([1, 4, 8, 8], 51); using var context = NativeMath.CpuNoise([1, 3, 16], 52); using var time = tensor(new[] { 17.25f });
            using var baseline = model.Forward(input, time, context);
            using var optimizer = new LoraTrainingOptimizer(patches.Values, "SGD", .01);
            using var predicted = model.ForwardForTraining(input, time, context, patches);
            Assert.Equal(baseline.data<float>().ToArray(), predicted.data<float>().ToArray());
            using var loss = (predicted - .1f).square().mean(); optimizer.Accumulate(loss);
            foreach (var parameter in new[] { lora.Down, bias.Difference, norm.Difference })
            {
                using var gradient = parameter.grad; Assert.NotNull(gradient); Assert.True(gradient!.abs().sum().item<float>() > 0);
            }
            optimizer.Step();
            using var changed = model.ForwardForTraining(input, time, context, patches);
            Assert.NotEqual(baseline.data<float>().ToArray(), changed.data<float>().ToArray());
            using var unchanged = model.Forward(input, time, context);
            Assert.Equal(baseline.data<float>().ToArray(), unchanged.data<float>().ToArray());
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Difference_rejects_invalid_shapes_nonfinite_data_and_duplicate_owners()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        Assert.Throws<ArgumentException>(() => new TrainableDifferencePatch(zeros([2, 2])));
        Assert.Throws<ArgumentException>(() => new TrainableDifferencePatch(tensor(new[] { float.NaN })));
        using var diff = new TrainableDifferencePatch(zeros([2]));
        Assert.Throws<ArgumentException>(() => diff.Apply(zeros([3])));
        Assert.Throws<ArgumentException>(() => new LoraTrainingOptimizer(new[] { diff, diff }, "SGD", .01));
        Assert.Throws<OperationCanceledException>(() => diff.Apply(zeros([2]), new(true)));
    }
}
