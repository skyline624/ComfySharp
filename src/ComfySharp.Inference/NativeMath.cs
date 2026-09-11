using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

public static class NativeMath
{
    /// <summary>CPU float32 noise with a per-call native generator. No cross-backend or ComfyUI RNG parity promise.</summary>
    public static Tensor CpuNoise(long[] shape, ulong seed, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Any(d => d < 0)) throw new ArgumentOutOfRangeException(nameof(shape));
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var generator = new Generator(seed, CPU);
        var result = randn(shape, generator: generator, dtype: ScalarType.Float32, device: CPU);
        cancellationToken.ThrowIfCancellationRequested();
        return result.MoveToOuterDisposeScope();
    }

    /// <summary>One no-churn Euler step: x + (x - denoised) / sigma * (nextSigma - sigma).
    /// Inputs are borrowed; output belongs to caller. This is not a model or a full sampler.</summary>
    public static Tensor EulerStep(Tensor x, Tensor denoised, double sigma, double nextSigma,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!double.IsFinite(sigma) || sigma <= 0 || !double.IsFinite(nextSigma) || nextSigma < 0 || nextSigma > sigma)
            throw new ArgumentOutOfRangeException(nameof(sigma));
        NativeRuntimeBootstrap.Initialize();
        if (!x.shape.SequenceEqual(denoised.shape) || x.dtype != denoised.dtype || x.device.ToString() != denoised.device.ToString())
            throw new ArgumentException("Inputs must have matching shapes, dtype and device.");
        if (x.dtype is not (ScalarType.Float32 or ScalarType.Float64)) throw new NotSupportedException("Euler foundation supports F32/F64.");
        using var scope = NewDisposeScope();
        var result = x + (x - denoised) / sigma * (nextSigma - sigma);
        cancellationToken.ThrowIfCancellationRequested();
        return result.MoveToOuterDisposeScope();
    }
}
