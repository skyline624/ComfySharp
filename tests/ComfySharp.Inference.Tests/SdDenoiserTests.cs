using ComfySharp.Inference;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Pipeline composition and ownership checks using a real synthetic U-Net.
/// These invariants are not independent source goldens or pretrained-model qualification.</summary>
[Collection("Classical VAE")]
public sealed class SdDenoiserTests : IDisposable
{
    private readonly int previousThreads;
    private static SdUnetConfig Config => new(32, 16, SdAttentionHeadMode.FixedCount, 4, false);

    public SdDenoiserTests()
    {
        NativeRuntimeBootstrap.Initialize();
        previousThreads = get_num_threads();
        set_num_threads(1);
    }

    [Theory]
    [InlineData(SdPredictionKind.Epsilon, false)]
    [InlineData(SdPredictionKind.Velocity, true)]
    public void DenoiseComposesScheduleInputScalingActualUnetAndPredictionConversion(SdPredictionKind kind, bool perRowSigma)
    {
        using var scope = NewDisposeScope();
        using var bank = SdSyntheticInputs.CreateUnet(Config);
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, kind);
        var latent = Input("composition.latent", 2, 4, 4, 5);
        var context = Input("composition.context", 2, 3, 16);
        var sigma = tensor(perRowSigma ? new[] { 0.25f, 1.5f } : new[] { 0.75f });
        using var scaled = SdSamplingMath.ScaleInput(latent, sigma);
        using var timestep = denoiser.Sampling.Timestep(sigma);
        var time = timestep.to_type(ScalarType.Float32).reshape(-1);
        using var prediction = model.Forward(scaled, time, context);
        using var expected = SdSamplingMath.Denoised(latent, prediction, sigma, kind);
        using var actual = denoiser.Denoise(latent, sigma, context);
        Equal(expected, actual);
    }

    [Theory]
    [InlineData(2, 3, 6)]
    [InlineData(3, 3, 3)]
    [InlineData(2, 10, 0)]
    public void GuidanceComposesSeparateOrWholeSequenceLcmBatchesAndFallsBack(int positiveLength, int negativeLength, int commonLength)
    {
        using var scope = NewDisposeScope();
        using var bank = SdSyntheticInputs.CreateUnet(Config);
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        var latent = Input("guidance.latent", 2, 4, 4, 5);
        var sigma = tensor(new[] { 0.25f, 1.5f });
        var positive = Input("guidance.positive", 2, positiveLength, 16);
        var negative = Input("guidance.negative", 2, negativeLength, 16);
        var snapshots = new[] { latent, sigma, positive, negative }.Select(Values).ToArray();
        const double scale = 3.5;
        using var positiveResult = denoiser.Denoise(latent, sigma, positive);
        using var negativeResult = denoiser.Denoise(latent, sigma, negative);
        using var expectedSeparate = SdSamplingMath.Guide(positiveResult, negativeResult, scale);
        using var separate = denoiser.DenoiseGuided(latent, sigma, positive, negative,
            new() { Scale = scale, BatchMode = SdGuidanceBatchMode.Separate });
        Equal(expectedSeparate, separate);
        using var concatenated = denoiser.DenoiseGuided(latent, sigma, positive, negative,
            new() { Scale = scale, BatchMode = SdGuidanceBatchMode.ConcatenateCompatible });

        Assert.Equal(commonLength != 0, SdDenoiser.TryCommonContextLength(positiveLength, negativeLength, out long actualCommon));
        Assert.Equal(commonLength, actualCommon);
        if (commonLength == 0)
        {
            // Exact equality checks that an incompatible pair takes the same
            // conservative two-call route rather than padding or truncation.
            Equal(separate, concatenated);
        }
        else
        {
            var positiveRepeated = positive.repeat(1, commonLength / positiveLength, 1);
            var negativeRepeated = negative.repeat(1, commonLength / negativeLength, 1);
            var contexts = cat(new[] { positiveRepeated, negativeRepeated }, 0);
            var latents = cat(new[] { latent, latent }, 0);
            var sigmas = cat(new[] { sigma, sigma }, 0);
            using var predictions = denoiser.Denoise(latents, sigmas, contexts);
            using var expected = SdSamplingMath.Guide(predictions.narrow(0, 0, 2), predictions.narrow(0, 2, 2), scale);
            Equal(expected, concatenated);
            // Separate and concatenated native batches have independent source
            // references: upstream itself can exceed the profile between them.
        }
        var inputs = new[] { latent, sigma, positive, negative };
        for (int i = 0; i < inputs.Length; i++) Assert.Equal(snapshots[i], Values(inputs[i]));
    }

    [Fact]
    public void NearOneOptimizationAndAbsentNegativeFollowGuideSemantics()
    {
        using var scope = NewDisposeScope();
        using var bank = SdSyntheticInputs.CreateUnet(Config);
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, SdPredictionKind.Velocity);
        var latent = Input("omission.latent", 1, 4, 4, 5);
        var sigma = tensor(0.75f);
        var positive = Input("omission.context", 1, 3, 16);
        var invalidNegative = zeros(new long[] { 1, 0, 16 });
        using var positiveResult = denoiser.Denoise(latent, sigma, positive);
        foreach (double scale in new[] { 1.0, 1.0 - 5e-10, 1.0 + 5e-10 })
        {
            using var expected = SdSamplingMath.Guide(positiveResult, null, scale);
            using var actual = denoiser.DenoiseGuided(latent, sigma, positive, invalidNegative,
                new() { Scale = scale, BatchMode = SdGuidanceBatchMode.ConcatenateCompatible });
            Equal(expected, actual);
            Assert.Throws<ArgumentException>(() => denoiser.DenoiseGuided(latent, sigma, positive, invalidNegative,
                new() { Scale = scale, DisableScaleOneOptimization = true }));
        }
        Assert.Throws<ArgumentException>(() => denoiser.DenoiseGuided(latent, sigma, positive, invalidNegative,
            new() { Scale = 1.0 + 2e-9 }));
        foreach (double scale in new[] { 0.0, 7.0, 1.0 + 2e-9 })
        {
            using var expected = SdSamplingMath.Guide(positiveResult, null, scale);
            using var actual = denoiser.DenoiseGuided(latent, sigma, positive, null,
                new() { Scale = scale, DisableScaleOneOptimization = true });
            Equal(expected, actual);
        }
    }

    [Fact]
    public void RetainedPipelineAndOutputsSurviveAllParentsAndCallerScopesWithoutMutatingInputs()
    {
        using var callerGrad = set_grad_enabled(true);
        Tensor output, guided, parameter;
        float[] snapshot, guidedSnapshot;
        using (var scope = NewDisposeScope())
        using (var bank = SdSyntheticInputs.CreateUnet(Config))
        using (var model = new SdUnet(bank))
        using (var original = new SdDenoiser(model, SdPredictionKind.Epsilon))
        using (var retained = original.Retain())
        {
            parameter = bank.GetTensor("input_blocks.0.0.weight");
            original.Dispose();
            model.Dispose();
            bank.Dispose();
            Assert.False(parameter.IsInvalid);
            Assert.Equal(Config, retained.Config);
            Assert.Equal(SdPredictionKind.Epsilon, retained.PredictionKind);
            var latent = Input("lifetime.latent", 1, 4, 4, 5).requires_grad_(true);
            var sigma = tensor(new[] { 0.75f }, requires_grad: true);
            var context = Input("lifetime.context", 1, 3, 16).requires_grad_(true);
            var before = new[] { latent, sigma, context }.Select(Values).ToArray();
            Assert.Throws<ObjectDisposedException>(() => original.Retain());
            Assert.Throws<ObjectDisposedException>(() => original.Denoise(latent, sigma, context));
            output = retained.Denoise(latent, sigma, context);
            guided = retained.DenoiseGuided(latent, sigma, context, null, new() { Scale = 2 });
            Assert.True(is_grad_enabled());
            foreach (var result in new[] { output, guided })
            {
                Assert.False(result.requires_grad);
                Assert.Equal(ScalarType.Float32, result.dtype);
                Assert.Equal(DeviceType.CPU, result.device_type);
                Assert.Equal(latent.shape, result.shape);
            }
            snapshot = Values(output);
            guidedSnapshot = Values(guided);
            var inputs = new[] { latent, sigma, context };
            for (int i = 0; i < inputs.Length; i++) Assert.Equal(before[i], Values(inputs[i]));
        }
        Assert.True(parameter.IsInvalid);
        using (output) Assert.Equal(snapshot, Values(output));
        Assert.True(output.IsInvalid);
        using (guided) Assert.Equal(guidedSnapshot, Values(guided));
        Assert.True(guided.IsInvalid);
        Assert.True(is_grad_enabled());
    }

    [Fact]
    public void CancellationAndInvalidSigmaOrContextLeavePipelineUsableAndRestoreGradMode()
    {
        using var scope = NewDisposeScope();
        using var callerGrad = set_grad_enabled(true);
        using var bank = SdSyntheticInputs.CreateUnet(Config);
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        var latent = Input("failure.latent", 2, 4, 4, 5);
        var sigma = tensor(new[] { 0.25f, 1.5f });
        var context = Input("failure.context", 2, 3, 16);
        var before = new[] { latent, sigma, context }.Select(Values).ToArray();
        Assert.Throws<OperationCanceledException>(() => denoiser.Denoise(latent, sigma, context, new(true)));
        Assert.Throws<OperationCanceledException>(() => denoiser.DenoiseGuided(latent, sigma, context, context, cancellationToken: new(true)));
        foreach (var invalid in new[]
                 {
                     tensor(new[] { -0.5f }), tensor(new[] { float.NaN }), tensor(new[] { float.PositiveInfinity }),
                     tensor(new[] { 0.1f, 0.2f, 0.3f }), zeros(new long[] { 0 }), sigma.unsqueeze(0), sigma.to_type(ScalarType.Float64)
                 })
        {
            Assert.Throws<ArgumentException>(() => denoiser.Denoise(latent, invalid, context));
            Assert.Throws<ArgumentException>(() => denoiser.DenoiseGuided(latent, invalid, context, context));
        }
        foreach (var invalid in new[]
                 {
                     zeros(new long[] { 1, 3, 16 }), zeros(new long[] { 2, 0, 16 }), zeros(new long[] { 2, 3, 17 }),
                     context.unsqueeze(0), context.to_type(ScalarType.Float64)
                 })
        {
            Assert.Throws<ArgumentException>(() => denoiser.Denoise(latent, sigma, invalid));
            Assert.Throws<ArgumentException>(() => denoiser.DenoiseGuided(latent, sigma, context, invalid));
        }
        Assert.Throws<ArgumentNullException>(() => denoiser.Denoise(latent, sigma, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => denoiser.DenoiseGuided(latent, sigma, context, context, new() { Scale = double.NaN }));
        Assert.Throws<ArgumentOutOfRangeException>(() => denoiser.DenoiseGuided(latent, sigma, context, context, new() { BatchMode = (SdGuidanceBatchMode)99 }));
        Assert.True(is_grad_enabled());
        using var first = denoiser.Denoise(latent, sigma, context);
        using var second = denoiser.Denoise(latent, sigma, context);
        Equal(first, second);
        first.Dispose();
        Assert.False(second.IsInvalid);
        var inputs = new[] { latent, sigma, context };
        for (int i = 0; i < inputs.Length; i++) Assert.Equal(before[i], Values(inputs[i]));
        Assert.True(is_grad_enabled());
    }

    private static Tensor Input(string name, params long[] shape) =>
        tensor(SdSyntheticInputs.Values(name, shape), shape, dtype: ScalarType.Float32, device: CPU);

    private static float[] Values(Tensor value)
    {
        using var contiguous = value.contiguous();
        return contiguous.data<float>().ToArray();
    }

    private static void Equal(Tensor expected, Tensor actual)
    {
        Assert.Equal(expected.shape, actual.shape);
        var values = Values(actual);
        Assert.All(values, value => Assert.True(float.IsFinite(value)));
        Assert.Equal(Values(expected), values);
    }

    public void Dispose() => set_num_threads(previousThreads);
}
