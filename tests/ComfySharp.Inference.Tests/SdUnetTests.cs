using System.Security.Cryptography;
using System.Text;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdUnetTests : IDisposable
{
    private readonly int previousThreads;

    public SdUnetTests()
    {
        NativeRuntimeBootstrap.Initialize();
        previousThreads = get_num_threads();
        set_num_threads(1);
    }

    public void Dispose() => set_num_threads(previousThreads);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BorrowedContextStorageOffsetsAndStridesDoNotChangePrediction(bool linearProjection)
    {
        using var scope = NewDisposeScope();
        using var callerGradMode = set_grad_enabled(true);
        string id = linearProjection ? "sd2-reduced/square" : "sd15-reduced/square";
        var config = new SdUnetConfig(32, 16,
            linearProjection ? SdAttentionHeadMode.FixedSize : SdAttentionHeadMode.FixedCount,
            linearProjection ? 8 : 4, linearProjection);
        using var bank = SdSyntheticInputs.CreateUnet(config);
        using var model = new SdUnet(bank);
        var latent = tensor(SdSyntheticInputs.Values(id + "/latent", new long[] { 1, 4, 8, 8 }), new long[] { 1, 4, 8, 8 }).clone();
        var aligned = tensor(SdSyntheticInputs.Values(id + "/context", new long[] { 1, 3, 16 }), new long[] { 1, 3, 16 }).clone();
        var time = tensor(new[] { 0.125f });
        using var baseline = model.Forward(latent, time, aligned);
        var expected = baseline.data<float>().ToArray();
        var originalValues = aligned.data<float>().ToArray();
        for (int offset = 1; offset < 16; offset++)
        {
            using var caseScope = NewDisposeScope();
            var context = empty(48 + offset).narrow(0, offset, 48).reshape(1, 3, 16);
            context.copy_(aligned);
            context.requires_grad_(true);
            using var output = model.Forward(latent, time, context);
            Assert.Equal(expected, output.data<float>().ToArray());
            Assert.Equal(originalValues, context.data<float>().ToArray());
            Assert.True(context.requires_grad);
            Assert.False(output.requires_grad);
        }

        var strided = empty(49).narrow(0, 1, 48).reshape(1, 16, 3).transpose(1, 2);
        strided.copy_(aligned);
        Assert.False(strided.is_contiguous());
        using var stridedOutput = model.Forward(latent, time, strided);
        Assert.Equal(expected, stridedOutput.data<float>().ToArray());
        using var cancelled = new CancellationTokenSource();
        model.DiagnosticObserver = (_, _) => cancelled.Cancel();
        Assert.Throws<OperationCanceledException>(() => model.Forward(latent, time, strided, cancelled.Token));
        model.DiagnosticObserver = null;
        Assert.False(strided.IsInvalid);
        Assert.Equal(originalValues, strided.data<float>().ToArray());
        using var afterCancellation = model.Forward(latent, time, strided);
        Assert.Equal(expected, afterCancellation.data<float>().ToArray());
        Assert.True(is_grad_enabled());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompleteGraphHandlesOddRectanglesAndOutputsSurviveAllOwners(bool linearProjection)
    {
        using var callerGradMode = set_grad_enabled(true);
        Tensor output;
        Tensor parameter;
        float[] snapshot;
        using (var callerScope = NewDisposeScope())
        using (var bank = SyntheticWeights(linearProjection))
        using (var model = new SdUnet(bank))
        {
            parameter = bank.GetTensor("input_blocks.0.0.weight");
            var latent = Input(new long[] { 2, 4, 9, 11 }, requiresGrad: true);
            var context = Input(new long[] { 2, 5, 16 }, phase: 0.7f, requiresGrad: true);
            var times = tensor(new[] { 17.25f }, requires_grad: true);
            var latentBefore = latent.data<float>().ToArray();
            var contextBefore = context.data<float>().ToArray();
            var timesBefore = times.data<float>().ToArray();
            output = model.Forward(latent, times, context);
            Assert.Equal(new long[] { 2, 4, 9, 11 }, output.shape);
            Assert.Equal(ScalarType.Float32, output.dtype);
            Assert.Equal(TorchSharp.DeviceType.CPU, output.device_type);
            Assert.True(output.is_contiguous());
            Assert.False(output.requires_grad);
            Assert.True(is_grad_enabled());
            Assert.Equal(latentBefore, latent.data<float>().ToArray());
            Assert.Equal(contextBefore, context.data<float>().ToArray());
            Assert.Equal(timesBefore, times.data<float>().ToArray());
            snapshot = output.data<float>().ToArray();
            Assert.All(snapshot, value => Assert.True(float.IsFinite(value)));
            Assert.Contains(snapshot, value => value != 0);
        }
        Assert.True(parameter.IsInvalid);
        using (output) Assert.Equal(snapshot, output.data<float>().ToArray());
        Assert.True(output.IsInvalid);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RealPredictionDependsOnLatentTimeAndAllContextTokens(bool linearProjection)
    {
        using var scope = NewDisposeScope();
        using var bank = SyntheticWeights(linearProjection);
        using var model = new SdUnet(bank);
        var latent = Input(new long[] { 1, 4, 8, 10 });
        var times = tensor(new[] { 17.25f });
        var context = Input(new long[] { 1, 5, 16 }, phase: 0.7f);
        using var original = model.Forward(latent, times, context);
        using var latentChanged = model.Forward(latent + 0.125f, times, context);
        using var timeChanged = model.Forward(latent, tensor(new[] { 731.5f }), context);
        var otherContext = context.clone();
        otherContext.narrow(1, 4, 1).add_(0.4f);
        using var lastTokenChanged = model.Forward(latent, times, otherContext);
        using var shorterContext = model.Forward(latent, times, context.narrow(1, 0, 3));
        var expected = original.data<float>().ToArray();
        foreach (var changed in new[] { latentChanged, timeChanged, lastTokenChanged, shorterContext })
        {
            Assert.Equal(original.shape, changed.shape);
            var values = changed.data<float>().ToArray();
            Assert.All(values, value => Assert.True(float.IsFinite(value)));
            Assert.False(expected.SequenceEqual(values));
        }
    }

    [Fact]
    public void OneOrPerRowTimestepsAndDifferentContextLengthsAreSupported()
    {
        using var scope = NewDisposeScope();
        using var bank = SyntheticWeights();
        using var model = new SdUnet(bank);
        var latent = Input(new long[] { 2, 4, 8, 8 });
        var context = Input(new long[] { 2, 7, 16 });
        using var sharedTime = model.Forward(latent, tensor(new[] { 0.5f }), context);
        using var perRowTime = model.Forward(latent, tensor(new[] { 0.5f, 299.25f }), context);
        Assert.Equal(latent.shape, sharedTime.shape);
        Assert.Equal(latent.shape, perRowTime.shape);
        Assert.False(sharedTime.data<float>().ToArray().SequenceEqual(perRowTime.data<float>().ToArray()));
    }

    [Fact]
    public void RejectsInvalidTensorContractsWithoutChangingGradModeOrBreakingModel()
    {
        using var scope = NewDisposeScope();
        using var gradMode = set_grad_enabled(true);
        using var bank = SyntheticWeights();
        using var model = new SdUnet(bank);
        var latent = Input(new long[] { 1, 4, 8, 8 });
        var times = tensor(new[] { 12.5f });
        var context = Input(new long[] { 1, 5, 16 });
        Assert.Throws<ArgumentNullException>(() => model.Forward(null!, times, context));
        Assert.Throws<ArgumentException>(() => model.Forward(latent.to_type(ScalarType.Float64), times, context));
        Assert.Throws<ArgumentException>(() => model.Forward(latent, tensor(new long[] { 12 }), context));
        Assert.Throws<ArgumentException>(() => model.Forward(latent, times, context.to_type(ScalarType.Float16)));
        Assert.Throws<ArgumentException>(() => model.Forward(latent.squeeze(0), times, context));
        Assert.Throws<ArgumentException>(() => model.Forward(zeros(new long[] { 1, 9, 8, 8 }), times, context));
        Assert.Throws<ArgumentException>(() => model.Forward(zeros(new long[] { 0, 4, 8, 8 }), times, context));
        Assert.Throws<ArgumentException>(() => model.Forward(zeros(new long[] { 1, 4, 0, 8 }), times, context));
        Assert.Throws<ArgumentException>(() => model.Forward(latent, times.unsqueeze(0), context));
        Assert.Throws<ArgumentException>(() => model.Forward(latent, tensor(new[] { 1f, 2f }), context));
        Assert.Throws<ArgumentException>(() => model.Forward(latent, zeros(new long[] { 0 }), context));
        Assert.Throws<ArgumentException>(() => model.Forward(latent, times, zeros(new long[] { 1, 0, 16 })));
        Assert.Throws<ArgumentException>(() => model.Forward(latent, times, zeros(new long[] { 2, 5, 16 })));
        Assert.Throws<ArgumentException>(() => model.Forward(latent, times, zeros(new long[] { 1, 5, 17 })));
        var disposed = times.clone();
        disposed.Dispose();
        Assert.Throws<ObjectDisposedException>(() => model.Forward(latent, disposed, context));
        Assert.True(is_grad_enabled());
        using var result = model.Forward(latent, times, context);
        Assert.All(result.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
        Assert.True(is_grad_enabled());
    }

    [Fact]
    public void RetainedOwnerAndOutputSurviveCancellationAndErrorsAfterNativeAllocations()
    {
        using var scope = NewDisposeScope();
        using var gradMode = set_grad_enabled(true);
        using var bank = SyntheticWeights();
        using var original = new SdUnet(bank);
        using var retained = original.Retain();
        original.Dispose();
        bank.Dispose();
        var latent = Input(new long[] { 1, 4, 8, 8 });
        var times = tensor(new[] { 12.5f });
        var context = Input(new long[] { 1, 5, 16 });
        Assert.Throws<ObjectDisposedException>(() => original.Retain());
        Assert.Throws<ObjectDisposedException>(() => original.Forward(latent, times, context));
        Assert.Throws<OperationCanceledException>(() => retained.Forward(latent, times, context, new(true)));
        // The first source GroupNorm rejects this reduced-width degenerate case after
        // the timestep MLP and initial convolution have already allocated intermediates.
        Assert.Throws<ArgumentException>(() => retained.Forward(zeros(new long[] { 1, 4, 1, 1 }), times, context));
        Assert.True(is_grad_enabled());
        using var first = retained.Forward(latent, times, context);
        using var second = retained.Forward(latent, times, context);
        Assert.Equal(first.data<float>().ToArray(), second.data<float>().ToArray());
        first.Dispose();
        Assert.All(second.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
        retained.Dispose();
        Assert.False(second.IsInvalid);
    }

    [Fact]
    public async Task ForwardAndOwnerDisposalRaceHasOnlyTheTwoValidLifetimeOutcomes()
    {
        // Do not carry a thread-local DisposeScope across await; these input owners are explicit.
        using var latent = Input(new long[] { 1, 4, 8, 8 });
        using var times = tensor(new[] { 51.25f });
        using var context = Input(new long[] { 1, 5, 16 });
        for (int iteration = 0; iteration < 4; iteration++)
        {
            using var bank = SyntheticWeights();
            var parameter = bank.GetTensor("input_blocks.0.0.weight");
            var model = new SdUnet(bank);
            bank.Dispose(); // Only the model and any running Forward lease can now keep parameters alive.
            using var barrier = new Barrier(2);
            var running = Task.Run(() =>
            {
                Assert.True(barrier.SignalAndWait(TimeSpan.FromSeconds(20)));
                try { return model.Forward(latent, times, context); }
                catch (ObjectDisposedException error) when (error.ObjectName == typeof(SdUnet).FullName)
                { return null; } // Dispose acquired the model gate first; no internal disposed-bank error is allowed.
            });
            Assert.True(barrier.SignalAndWait(TimeSpan.FromSeconds(20)));
            model.Dispose();
            using var result = await running;
            if (result is not null)
            {
                Assert.Equal(latent.shape, result.shape);
                Assert.All(result.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
            }
            Assert.True(parameter.IsInvalid);
            Assert.Throws<ObjectDisposedException>(() => model.Retain());
        }
        // A separately retained owner still computes after all raced owners have gone.
        using var survivorBank = SyntheticWeights();
        using var survivor = new SdUnet(survivorBank);
        survivorBank.Dispose();
        using var output = survivor.Forward(latent, times, context);
        Assert.All(output.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
    }

    internal static Tensor Input(long[] shape, float phase = 0, bool requiresGrad = false)
    {
        using var scope = NewDisposeScope();
        int count = checked((int)shape.Aggregate(1L, (a, b) => a * b));
        var values = Enumerable.Range(0, count).Select(i => 0.2f * MathF.Sin(i * 0.17f + phase)).ToArray();
        return tensor(values, dtype: ScalarType.Float32, device: CPU, requires_grad: requiresGrad).reshape(shape).MoveToOuterDisposeScope();
    }

    // Nondegenerate structural/lifetime inputs only. Independent numerical references belong
    // to the frozen upstream laboratory; no outputs from this graph become golden arrays.
    internal static UnetWeightSet SyntheticWeights(bool linearProjection = false)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var config = new SdUnetConfig(32, 16,
            linearProjection ? SdAttentionHeadMode.FixedSize : SdAttentionHeadMode.FixedCount,
            linearProjection ? 8 : 4, linearProjection);
        var weights = new Dictionary<string, Tensor>();
        foreach (var (name, shape) in UnetWeightSchema.Describe(config))
        {
            byte[] digest = SHA256.HashData(Encoding.ASCII.GetBytes(name));
            int seed = BitConverter.ToInt32(digest, 0) & 0x7fffffff;
            int count = checked((int)shape.Aggregate(1L, (a, b) => a * b));
            var values = new float[count];
            bool normalizationWeight = name.EndsWith(".weight", StringComparison.Ordinal) && shape.Count == 1;
            for (int i = 0; i < count; i++)
                values[i] = (normalizationWeight ? 1f : 0f) + (((long)i * (1 + digest[4]) + seed) % 257 - 128) / 4096f;
            weights.Add(name, tensor(values, dtype: ScalarType.Float32, device: CPU).reshape(shape.ToArray()));
        }
        return UnetWeightSet.FromOwnedTensors(config, weights);
    }
}
