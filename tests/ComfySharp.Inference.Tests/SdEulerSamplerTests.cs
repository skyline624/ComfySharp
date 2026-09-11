using ComfySharp.Inference;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Trajectory ownership, cancellation and input contracts using a real reduced graph.
/// These behavior checks are not independent numerical references or model-family qualification.</summary>
[Collection("Classical VAE")]
public sealed class SdEulerSamplerTests : IDisposable
{
    private readonly int previousThreads;
    private static SdUnetConfig Config => new(32, 16, SdAttentionHeadMode.FixedCount, 4, false);

    public SdEulerSamplerTests()
    {
        NativeRuntimeBootstrap.Initialize();
        previousThreads = get_num_threads();
        set_num_threads(1);
    }

    [Theory]
    [InlineData(SdPredictionKind.Epsilon, SdGuidanceBatchMode.Separate)]
    [InlineData(SdPredictionKind.Velocity, SdGuidanceBatchMode.ConcatenateCompatible)]
    public void RetainedTrajectoryAndResultSurviveParentsAndScopesWithGradModeRestored(
        SdPredictionKind kind, SdGuidanceBatchMode batchMode)
    {
        using var callerGrad = set_grad_enabled(true);
        Tensor result, parameter;
        float[] snapshot;
        using (var scope = NewDisposeScope())
        using (var bank = SdSyntheticInputs.CreateUnet(Config))
        using (var model = new SdUnet(bank))
        using (var denoiser = new SdDenoiser(model, kind))
        using (var original = new SdEulerSampler(denoiser))
        using (var sampler = original.Retain())
        {
            parameter = bank.GetTensor("input_blocks.0.0.weight");
            original.Dispose();
            denoiser.Dispose();
            model.Dispose();
            bank.Dispose();
            Assert.False(parameter.IsInvalid);
            Assert.Equal(Config, sampler.Config);
            Assert.Equal(kind, sampler.PredictionKind);
            Assert.Throws<ObjectDisposedException>(() => original.Retain());
            var latent = Input("euler.life.latent", 2, 4, 4, 5).requires_grad_(true);
            var sigmas = tensor(new[] { 1.25f, 0.5f, 0.125f, 0f }, requires_grad: true);
            var positive = Input("euler.life.positive", 2, 2, 16).requires_grad_(true);
            var negative = Input("euler.life.negative", 2, 3, 16).requires_grad_(true);
            var inputs = new[] { latent, sigmas, positive, negative };
            var before = inputs.Select(Values).ToArray();
            var observedSteps = new List<long>();
            var observedSigmas = new List<float>();
            var borrowedStates = new List<Tensor>();
            sampler.DiagnosticObserver = (step, current, denoised, sigma) =>
            {
                observedSteps.Add(step);
                observedSigmas.Add(sigma.item<float>());
                if (step != 0) Assert.False(current.requires_grad);
                Assert.False(denoised.requires_grad);
                Assert.False(is_grad_enabled());
                if (step > 0) borrowedStates.Add(current);
            };
            result = sampler.Sample(latent, sigmas, positive, negative, new() { Scale = 3.5, BatchMode = batchMode });
            Assert.Equal(new long[] { 0, 1, 2 }, observedSteps);
            Assert.Equal(new[] { 1.25f, 0.5f, 0.125f }, observedSigmas);
            Assert.All(borrowedStates, state => Assert.True(state.IsInvalid));
            Assert.False(result.requires_grad);
            Assert.Equal(latent.shape, result.shape);
            Assert.Equal(ScalarType.Float32, result.dtype);
            Assert.Equal(DeviceType.CPU, result.device_type);
            Assert.True(is_grad_enabled());
            for (int i = 0; i < inputs.Length; i++) Assert.Equal(before[i], Values(inputs[i]));
            snapshot = Values(result);
        }
        Assert.True(parameter.IsInvalid);
        using (result) Assert.Equal(snapshot, Values(result));
        Assert.True(result.IsInvalid);
        Assert.True(is_grad_enabled());
    }

    [Fact]
    public void EqualAdjacentSigmasStillEvaluateTheModelAndSingleIntervalReturnsIndependentStorage()
    {
        using var scope = NewDisposeScope();
        using var bank = SdSyntheticInputs.CreateUnet(Config);
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        using var sampler = new SdEulerSampler(denoiser);
        var latent = Input("euler.steps.latent", 1, 4, 4, 5);
        var positive = Input("euler.steps.positive", 1, 3, 16);
        int calls = 0;
        sampler.DiagnosticObserver = (_, _, _, _) => calls++;
        using var plateau = sampler.Sample(latent, tensor(new[] { 1f, 1f, 0f }), positive, null);
        Assert.Equal(2, calls);
        calls = 0;
        using var single = sampler.Sample(latent, tensor(new[] { 1f, 0f }), positive, null);
        Assert.Equal(1, calls);
        var inputBefore = Values(latent);
        single.fill_(42);
        Assert.Equal(inputBefore, Values(latent));
        Assert.False(plateau.IsInvalid);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CancellationOrObserverFailureReleasesIntermediateStatesAndLeavesTheSamplerUsable(bool throwInstead)
    {
        using var scope = NewDisposeScope();
        using var callerGrad = set_grad_enabled(true);
        using var bank = SdSyntheticInputs.CreateUnet(Config);
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        using var sampler = new SdEulerSampler(denoiser);
        using var cancellation = new CancellationTokenSource();
        var latent = Input("euler.cancel.latent", 1, 4, 4, 5);
        var sigmas = tensor(new[] { 1f, 0.5f, 0.125f, 0f });
        var context = Input("euler.cancel.context", 1, 3, 16);
        var inputs = new[] { latent, sigmas, context };
        var before = inputs.Select(Values).ToArray();
        Tensor? releasedState = null, releasedDenoised = null;
        var calls = new List<long>();
        sampler.DiagnosticObserver = (step, current, denoised, _) =>
        {
            calls.Add(step);
            if (step != 1) return;
            releasedState = current;
            releasedDenoised = denoised;
            if (throwInstead) throw new TestObservationException();
            cancellation.Cancel();
        };
        if (throwInstead)
            Assert.Throws<TestObservationException>(() => sampler.Sample(latent, sigmas, context, null, cancellationToken: cancellation.Token));
        else
            Assert.Throws<OperationCanceledException>(() => sampler.Sample(latent, sigmas, context, null, cancellationToken: cancellation.Token));
        Assert.Equal(new long[] { 0, 1 }, calls);
        Assert.NotNull(releasedState);
        Assert.NotNull(releasedDenoised);
        Assert.True(releasedState.IsInvalid);
        Assert.True(releasedDenoised.IsInvalid);
        Assert.True(is_grad_enabled());
        for (int i = 0; i < inputs.Length; i++) Assert.Equal(before[i], Values(inputs[i]));
        sampler.DiagnosticObserver = null;
        using var later = sampler.Sample(latent, sigmas, context, null);
        Assert.Equal(latent.shape, later.shape);
        Assert.False(later.requires_grad);
    }

    [Fact]
    public void DisposingSamplerDuringActiveTrajectoryDoesNotDisposeItsRetainedOperation()
    {
        using var scope = NewDisposeScope();
        using var bank = SdSyntheticInputs.CreateUnet(Config);
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        using var sampler = new SdEulerSampler(denoiser);
        var parameter = bank.GetTensor("input_blocks.0.0.weight");
        bank.Dispose(); model.Dispose(); denoiser.Dispose();
        int calls = 0;
        sampler.DiagnosticObserver = (_, _, _, _) =>
        {
            sampler.Dispose();
            Assert.False(parameter.IsInvalid);
            calls++;
        };
        using var result = sampler.Sample(Input("euler.retain.latent", 1, 4, 4, 5),
            tensor(new[] { 1f, 0.5f, 0f }), Input("euler.retain.context", 1, 3, 16), null);
        Assert.Equal(2, calls);
        Assert.True(parameter.IsInvalid);
        Assert.Throws<ObjectDisposedException>(() => sampler.Retain());
        Assert.True(result.isfinite().all().item<bool>());
    }

    [Fact]
    public void InvalidSchedulesAndPreCancellationCannotReachDenoisingOrChangeCallerGradMode()
    {
        using var scope = NewDisposeScope();
        using var callerGrad = set_grad_enabled(true);
        using var bank = SdSyntheticInputs.CreateUnet(Config);
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        using var sampler = new SdEulerSampler(denoiser);
        var latent = Input("euler.invalid.latent", 1, 4, 4, 5);
        var context = Input("euler.invalid.context", 1, 3, 16);
        int calls = 0;
        sampler.DiagnosticObserver = (_, _, _, _) => calls++;
        Assert.Throws<OperationCanceledException>(() => sampler.Sample(null!, null!, null!, null, cancellationToken: new(true)));
        foreach (var invalid in new[]
        {
            tensor(Array.Empty<float>()), tensor(new[] { 0f }), tensor(new[] { 0f, 0f }),
            tensor(new[] { 1f, 0f, 0f }), tensor(new[] { 1f, -0.5f, 0f }), tensor(new[] { 0.5f, 1f, 0f }),
            tensor(new[] { 1f, 0.5f }), tensor(new[] { float.NaN, 0f }), tensor(new[] { float.PositiveInfinity, 0f }),
            tensor(new[] { 1f, 0f }).unsqueeze(0), tensor(new[] { 1.0, 0.0 }), tensor(1f)
        })
            Assert.Throws<ArgumentException>(() => sampler.Sample(latent, invalid, context, null));
        Assert.Throws<ArgumentNullException>(() => sampler.Sample(latent, null!, context, null));
        Assert.Equal(0, calls);
        Assert.True(is_grad_enabled());
    }

    [Fact]
    public void StridedSigmasPreserveTheirBackingStorageAndDisabledCallerGradMode()
    {
        using var scope = NewDisposeScope();
        using var callerGrad = set_grad_enabled(false);
        using var bank = SdSyntheticInputs.CreateUnet(Config);
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        using var sampler = new SdEulerSampler(denoiser);
        var backing = tensor(new[] { 1f, 99f, 0.5f, 99f, 0f, 99f });
        var sigmas = backing.as_strided(new long[] { 3 }, new long[] { 2 });
        Assert.False(sigmas.is_contiguous());
        var before = Values(backing);
        var observedSigmas = new List<float>();
        sampler.DiagnosticObserver = (_, _, _, sigma) => observedSigmas.Add(sigma.item<float>());
        using var result = sampler.Sample(Input("euler.strided.latent", 1, 4, 4, 5),
            sigmas, Input("euler.strided.context", 1, 3, 16), null);
        Assert.Equal(new[] { 1f, 0.5f }, observedSigmas);
        Assert.Equal(before, Values(backing));
        Assert.False(is_grad_enabled());
        Assert.False(result.requires_grad);
        Assert.True(result.isfinite().all().item<bool>());
    }

    [Fact]
    public void ContextAndGuidanceValidationPreserveExistingNearOneAndAbsentNegativeSemantics()
    {
        using var scope = NewDisposeScope();
        using var bank = SdSyntheticInputs.CreateUnet(Config);
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        using var sampler = new SdEulerSampler(denoiser);
        var latent = Input("euler.context.latent", 1, 4, 4, 5);
        var sigmas = tensor(new[] { 1f, 0f });
        var context = Input("euler.context.positive", 1, 3, 16);
        var invalidNegative = zeros(new long[] { 1, 0, 16 });
        using var omitted = sampler.Sample(latent, sigmas, context, invalidNegative, new() { Scale = 1 + 5e-10 });
        Assert.Throws<ArgumentException>(() => sampler.Sample(latent, sigmas, context, invalidNegative,
            new() { Scale = 1 + 5e-10, DisableScaleOneOptimization = true }));
        Assert.Throws<ArgumentException>(() => sampler.Sample(latent, sigmas, context.unsqueeze(0), null));
        Assert.Throws<ArgumentOutOfRangeException>(() => sampler.Sample(latent, sigmas, context, null, new() { Scale = double.NaN }));
        Assert.Throws<ArgumentOutOfRangeException>(() => sampler.Sample(latent, sigmas, context, null, new() { BatchMode = (SdGuidanceBatchMode)99 }));
        using var absent = sampler.Sample(latent, sigmas, context, null, new() { Scale = 7 });
        Assert.Equal(latent.shape, absent.shape);
    }

    private static Tensor Input(string name, params long[] shape) =>
        tensor(SdSyntheticInputs.Values(name, shape)).reshape(shape);

    private static float[] Values(Tensor value)
    {
        using var contiguous = value.contiguous();
        return contiguous.data<float>().ToArray();
    }

    private sealed class TestObservationException : Exception;
    public void Dispose() => set_num_threads(previousThreads);
}
