using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraOptimizerTests
{
    private static JsonDocument Corpus() => JsonDocument.Parse(typeof(LoraOptimizerTests).Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora-optimizers.reference.json")!);
    private static Tensor Read(JsonElement e) => tensor(e.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray(), e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray()).clone();
    private static void Compare(Tensor actual, JsonElement expected)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()), actual.shape);
        var values = expected.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var received = actual.data<float>().ToArray();
        for (int i = 0; i < values.Length; i++) Assert.InRange(Math.Abs((double)received[i] - values[i]), 0, 2e-6 + 2e-6 * Math.Abs(values[i]));
    }
    public static IEnumerable<object[]> Combinations() => from optimizer in new[] { "Adam", "AdamW", "SGD", "RMSprop" } from loss in new[] { "MSE", "L1", "Huber", "SmoothL1" } select new object[] { optimizer, loss };

    [Theory]
    [MemberData(nameof(Combinations))]
    public void Accumulated_updates_and_unused_parameters_match_source(string name, string lossName)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        using (var grad = set_grad_enabled(true))
        using (var corpus = Corpus())
        {
            var root = corpus.RootElement;
            var row = root.GetProperty("cases").EnumerateArray().Single(r => r.GetProperty("optimizer").GetString() == name && r.GetProperty("loss").GetString() == lossName);
            using var patch = new TrainableLoraPatch(Read(root.GetProperty("initialUp")), Read(root.GetProperty("initialDown")), 2.5);
            using var unused = new TrainableLoraPatch(full([1, 1], .875f), full([1, 1], .875f), 1);
            using var optimizer = new LoraTrainingOptimizer([patch, unused], name, .0125, 2);
            var weight = Read(root.GetProperty("base"));
            foreach (var batch in row.GetProperty("batches").EnumerateArray())
            {
                using var iteration = NewDisposeScope();
                var up = batch.GetProperty("trainUp").GetBoolean() ? patch.Up : patch.Up.detach();
                using var prediction = Read(batch.GetProperty("input")).matmul((weight + up.matmul(patch.Down) * 1.25).transpose(0, 1));
                using var loss = TrainingLoss.Calculate(lossName, prediction, Read(batch.GetProperty("target")));
                Assert.InRange(Math.Abs(loss.item<float>() - batch.GetProperty("loss").GetDouble()), 0, 2e-6);
                optimizer.Accumulate(loss);
                if (batch.GetProperty("updated").GetBoolean()) optimizer.Step();
                Compare(patch.Up, batch.GetProperty("up")); Compare(patch.Down, batch.GetProperty("down"));
                Assert.Equal(.875f, unused.Up.item<float>()); Assert.Null(unused.Up.grad);
            }
            Assert.Equal(3, optimizer.CompletedSteps); Assert.Equal(0, optimizer.PendingMicrobatches);
            Assert.Null(patch.Up.grad); Assert.Null(patch.Down.grad);
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory]
    [InlineData("MSE")]
    [InlineData("L1")]
    [InlineData("Huber")]
    [InlineData("SmoothL1")]
    public void Loss_boundary_gradients_match_source(string name)
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var grad = set_grad_enabled(true); using var corpus = Corpus();
        var row = corpus.RootElement.GetProperty("boundaries").EnumerateArray().Single(r => r.GetProperty("loss").GetString() == name);
        var prediction = tensor(new[] { -2f, -1f, -.25f, 0f, .25f, 1f, 2f }, requires_grad: true);
        using var loss = TrainingLoss.Calculate(name, prediction, zeros_like(prediction)); loss.backward();
        Assert.InRange(Math.Abs(loss.item<float>() - row.GetProperty("value").GetDouble()), 0, 2e-6);
        Compare(prediction.grad!, row.GetProperty("gradient"));
    }

    [Fact]
    public void Cancellation_discards_partial_gradients_and_retained_parameters_survive_the_caller()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        using (var grad = set_grad_enabled(true))
        {
            using var patch = new TrainableLoraPatch(ones([1, 1]), ones([1, 1]), 1);
            using var retained = patch.Retain(); using var optimizer = new LoraTrainingOptimizer([patch], "AdamW", .01, 2);
            patch.Dispose();
            optimizer.Accumulate(retained.Up.square().mean());
            Assert.Throws<InvalidOperationException>(() => optimizer.Step());
            Assert.Equal(1, optimizer.PendingMicrobatches);
            Assert.Throws<OperationCanceledException>(() => optimizer.Accumulate(retained.Up.square().mean(), new(true)));
            Assert.Equal(0, optimizer.PendingMicrobatches); Assert.Null(retained.Up.grad); Assert.Equal(1f, retained.Up.item<float>());
            optimizer.Accumulate(retained.Up.square().mean()); optimizer.Accumulate(retained.Up.square().mean());
            Assert.Throws<OperationCanceledException>(() => optimizer.Step(new(true)));
            Assert.Equal(0, optimizer.CompletedSteps); Assert.Equal(1f, retained.Up.item<float>());
            optimizer.Accumulate(retained.Up.square().mean()); optimizer.Accumulate(retained.Up.square().mean()); optimizer.Step();
            Assert.Equal(1, optimizer.CompletedSteps); Assert.True(retained.Up.item<float>() < 1);
            optimizer.Dispose(); Assert.Throws<ObjectDisposedException>(() => optimizer.Step());
            Assert.True(retained.Up.item<float>() > 0);
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Nonfinite_gradients_are_rejected_before_any_weight_update_and_reset_is_reusable()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var grad = set_grad_enabled(true);
        using var patch = new TrainableLoraPatch(zeros([1, 1]), ones([1, 1]), 1);
        using var optimizer = new LoraTrainingOptimizer([patch], "Adam", .01);
        optimizer.Accumulate(patch.Up.sqrt().sum() + patch.Down.square().sum());
        Assert.Throws<ArithmeticException>(() => optimizer.Step());
        Assert.Equal(0f, patch.Up.item<float>()); Assert.Equal(1f, patch.Down.item<float>());
        Assert.Equal(0, optimizer.PendingMicrobatches); Assert.Null(patch.Up.grad);
        optimizer.Accumulate((patch.Up + 1).square().sum()); optimizer.Step();
        Assert.Equal(1, optimizer.CompletedSteps); Assert.True(patch.Up.item<float>() < 0);
    }

    [Fact]
    public void Overflow_during_update_faults_the_session_instead_of_reusing_corrupted_state()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var grad = set_grad_enabled(true);
        using var patch = new TrainableLoraPatch(full([1, 1], -3e38f), ones([1, 1]), 1);
        using var optimizer = new LoraTrainingOptimizer([patch], "SGD", 1);
        optimizer.Accumulate(((patch.Up - patch.Up.detach()) * 3e38).sum());
        Assert.Throws<ArithmeticException>(() => optimizer.Step());
        Assert.Equal(0, optimizer.CompletedSteps); Assert.Null(patch.Up.grad);
        Assert.Throws<InvalidOperationException>(() => optimizer.ResetAccumulation());
        Assert.Throws<InvalidOperationException>(() => optimizer.Accumulate(patch.Down.sum()));
    }

    [Fact]
    public void Invalid_configuration_and_duplicate_owners_do_not_take_ownership()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        using var patch = new TrainableLoraPatch(ones([1, 1]), ones([1, 1]), 1); using var retained = patch.Retain();
        Assert.Throws<ArgumentException>(() => new LoraTrainingOptimizer([patch, retained], "Adam", .01));
        Assert.Throws<ArgumentException>(() => new LoraTrainingOptimizer([], "Adam", .01));
        Assert.Throws<ArgumentException>(() => new LoraTrainingOptimizer([patch], "unknown", .01));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoraTrainingOptimizer([patch], "Adam", double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoraTrainingOptimizer([patch], "Adam", .01, 0));
        Assert.Throws<ArgumentException>(() => TrainingLoss.Calculate("MSE", ones([2, 1]), ones([1])));
        Assert.Throws<ArgumentException>(() => TrainingLoss.Calculate("unknown", ones([1]), ones([1])));
        Assert.Equal(1f, patch.Up.item<float>());
    }
}
