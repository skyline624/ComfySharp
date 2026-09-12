using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Frozen KSampler.set_steps slicing and Sampler.max_denoise for the Karras path.</summary>
public static class SdKarrasSchedule
{
    public static Tensor Create(int steps, double denoise, SdDiscreteSampling sampling,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(sampling);
        if (steps is < 1 or > 10000) throw new ArgumentOutOfRangeException(nameof(steps));
        if (!double.IsFinite(denoise) || denoise is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(denoise));
        NativeRuntimeBootstrap.Initialize();
        if (denoise == 0) return empty(new long[] { 0 }, dtype: ScalarType.Float32, device: CPU);
        double requested = denoise > .9999 ? steps : Math.Truncate(steps / denoise);
        if (requested > 10000) throw new NotSupportedException("The expanded Karras schedule currently supports at most 10000 steps.");
        using var scope = NewDisposeScope();
        using var full = SigmaSchedules.Karras((int)requested, sampling.SigmaMin, sampling.SigmaMax, cancellationToken: cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        return full.narrow(0, full.shape[0] - steps - 1, steps + 1).clone().MoveToOuterDisposeScope();
    }

    public static bool UsesMaximumNoise(double firstSigma, double sigmaMax)
    {
        if (!double.IsFinite(firstSigma) || firstSigma < 0 || !double.IsFinite(sigmaMax) || sigmaMax <= 0)
            throw new ArgumentOutOfRangeException(nameof(firstSigma));
        return firstSigma > sigmaMax || Math.Abs(firstSigma - sigmaMax) <= 1e-5 * Math.Max(firstSigma, sigmaMax);
    }
}
