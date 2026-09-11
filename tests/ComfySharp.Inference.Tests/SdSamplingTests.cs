using ComfySharp.Inference;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdSamplingTests
{
    [Fact]
    public void EmptySchedulesScalarShapesAndClampedInfinityHaveDefinedBehavior()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var sampling = SdDiscreteSampling.Default;
        using var empty = sampling.Timestep(empty_like(tensor(Array.Empty<float>())));
        Assert.Equal(new long[] { 0 }, empty.shape);
        using var zero = sampling.Timestep(tensor(0f));
        Assert.Empty(zero.shape);
        Assert.Equal(0L, zero.item<long>());
        using var endpoints = sampling.Sigma(tensor(new[] { float.NegativeInfinity, float.PositiveInfinity }));
        using var ordinary = sampling.Sigma(tensor(new[] { 0f, 999f }));
        Assert.Equal(ordinary.data<float>().ToArray(), endpoints.data<float>().ToArray());
        Assert.Equal(999999999.9, sampling.PercentToSigma(double.NegativeInfinity));
        Assert.Equal(0, sampling.PercentToSigma(double.PositiveInfinity));
    }

    [Fact]
    public void MalformedSigmaAndMismatchedBatchesFailWithoutLeakingOrChangingGradMode()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var latent = ones(new long[] { 2, 4, 2, 3 });
        var wrongBatch = tensor(new[] { 1f, 2f, 3f });
        var negative = tensor(-1f);
        var nan = tensor(float.NaN);
        var doubleSigma = tensor(1.0, dtype: ScalarType.Float64);
        long before = Tensor.TotalCount;
        bool grad = is_grad_enabled();
        for (int i = 0; i < 5; i++)
        {
            Assert.Throws<ArgumentException>(() => { using var result = SdSamplingMath.ScaleInput(latent, wrongBatch); });
            Assert.Throws<ArgumentException>(() => { using var result = SdSamplingMath.ScaleInput(latent, negative); });
            Assert.Throws<ArgumentException>(() => { using var result = SdDiscreteSampling.Default.Timestep(nan); });
            Assert.Throws<ArgumentException>(() => { using var result = SdDiscreteSampling.Default.Timestep(doubleSigma); });
            Assert.Throws<ArgumentException>(() => { using var result = SdDiscreteSampling.Default.Sigma(nan); });
        }
        Assert.Equal(before, Tensor.TotalCount);
        Assert.Equal(grad, is_grad_enabled());
    }

    [Fact]
    public void ResultsOutliveCallerScopesDoNotMutateInputsAndDoNotRetainAutograd()
    {
        NativeRuntimeBootstrap.Initialize();
        using var outer = NewDisposeScope();
        var x = arange(48, dtype: ScalarType.Float32).reshape(1, 4, 3, 4).requires_grad_(true);
        var prediction = full_like(x, 0.25f);
        var sigma = tensor(2f);
        float[] original = x.data<float>().ToArray();
        Tensor input, epsilon, velocity, noise, guide, scaled, indices;
        long before = Tensor.TotalCount;
        using (NewDisposeScope())
        {
            input = SdSamplingMath.ScaleInput(x, sigma);
            epsilon = SdSamplingMath.Denoised(x, prediction, sigma, SdPredictionKind.Epsilon);
            velocity = SdSamplingMath.Denoised(x, prediction, sigma, SdPredictionKind.Velocity);
            noise = SdSamplingMath.NoiseScaling(prediction, x, sigma, maximumDenoise: true);
            guide = SdSamplingMath.Guide(x, prediction, 7.5);
            scaled = SdSamplingMath.ProcessLatentIn(x, SdSamplingMath.Sd15LatentScale);
            indices = SdDiscreteSampling.Default.Timestep(sigma);
        }
        Assert.Equal(original, x.data<float>().ToArray());
        foreach (var result in new[] { input, epsilon, velocity, noise, guide, scaled, indices })
        {
            Assert.False(result.IsInvalid);
            Assert.False(result.requires_grad);
            result.Dispose();
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void GuidanceKeepsNonfiniteArithmeticAndSourceScaleOneTolerance()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var positive = full(new long[] { 1, 4, 1, 1 }, float.PositiveInfinity);
        var negative = ones_like(positive);
        using var zeroScale = SdSamplingMath.Guide(positive, negative, 0);
        Assert.True(zeroScale.isnan().all().item<bool>());
        Assert.True(SdSamplingMath.CanOmitUnconditional(1.0 + 5e-10));
        Assert.True(SdSamplingMath.CanOmitUnconditional(1.0 - 5e-10));
        Assert.False(SdSamplingMath.CanOmitUnconditional(1.0 + 2e-9));
        Assert.False(SdSamplingMath.CanOmitUnconditional(1.0, disableOptimization: true));
        Assert.False(SdSamplingMath.CanOmitUnconditional(double.PositiveInfinity));
    }

    [Theory]
    [InlineData(77, 154, true, 154)]
    [InlineData(154, 231, true, 462)]
    [InlineData(77, 385, false, 0)]
    [InlineData(3, 4, true, 12)]
    [InlineData(3, 5, false, 0)]
    [InlineData(0, 77, false, 0)]
    public void ContextConcatenationUsesSourceLcmLimit(long first, long second, bool valid, long expected)
    {
        Assert.Equal(valid, SdDenoiser.TryCommonContextLength(first, second, out long common));
        Assert.Equal(expected, common);
    }

    [Fact]
    public void CancellationPrecedesNativeWork()
    {
        var canceled = new CancellationToken(canceled: true);
        Assert.Throws<OperationCanceledException>(() => { using var value = SdSamplingMath.ScaleInput(null!, null!, canceled); });
        Assert.Throws<OperationCanceledException>(() => { using var value = SdDiscreteSampling.Default.Sigma(null!, canceled); });
        Assert.Throws<OperationCanceledException>(() => SdDiscreteSampling.Default.PercentToSigma(0, canceled));
    }
}
