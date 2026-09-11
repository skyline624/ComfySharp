using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Plain SD diffusion boundaries from model_sampling.py and samplers.py.
/// CPU/F32 tensors are borrowed, never mutated; every result has independent wrapper ownership.</summary>
public static class SdSamplingMath
{
    public const double Sd15LatentScale = 0.18215;
    public const double SdxlLatentScale = 0.13025;

    public static Tensor ScaleInput(Tensor latent, Tensor sigma, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        ValidateLatent(latent, nameof(latent));
        var shaped = ReshapeSigma(sigma, latent);
        return Finish(latent / (shaped.pow(2) + 1.0).pow(0.5), cancellationToken);
    }

    public static Tensor Denoised(Tensor latent, Tensor prediction, Tensor sigma, SdPredictionKind kind,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        ValidatePair(latent, prediction);
        if (!Enum.IsDefined(kind)) throw new ArgumentOutOfRangeException(nameof(kind));
        var shaped = ReshapeSigma(sigma, latent);
        var result = kind == SdPredictionKind.Epsilon
            ? latent - prediction * shaped
            : latent * 1.0 / (shaped.pow(2) + 1.0) - prediction * shaped * 1.0 / (shaped.pow(2) + 1.0).pow(0.5);
        return Finish(result, cancellationToken);
    }

    public static Tensor NoiseScaling(Tensor noise, Tensor latentImage, Tensor sigma, bool maximumDenoise = false,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        ValidatePair(noise, latentImage);
        var shaped = ReshapeSigma(sigma, noise);
        var scaled = maximumDenoise ? noise * (1.0 + shaped.pow(2.0)).sqrt() : noise * shaped;
        scaled.add_(latentImage);
        return Finish(scaled, cancellationToken);
    }

    public static Tensor Guide(Tensor conditional, Tensor? unconditional, double scale,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!double.IsFinite(scale)) throw new ArgumentOutOfRangeException(nameof(scale));
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        ValidateLatent(conditional, nameof(conditional));
        if (unconditional is not null) ValidatePair(conditional, unconditional);
        // calc_cond_batch leaves an absent condition as zeros. Preserve subtraction/addition
        // even at scale 0/1 and preserve non-finite arithmetic instead of short-circuiting it.
        var negative = unconditional ?? zeros_like(conditional);
        return Finish(negative + (conditional - negative) * scale, cancellationToken);
    }

    /// <summary>Python math.isclose(scale, 1), default relative tolerance 1e-9 and absolute tolerance zero.</summary>
    public static bool CanOmitUnconditional(double scale, bool disableOptimization = false) =>
        !disableOptimization && double.IsFinite(scale) &&
        Math.Abs(scale - 1.0) <= 1e-9 * Math.Max(Math.Abs(scale), 1.0);

    public static Tensor ProcessLatentIn(Tensor rawLatent, double scale, CancellationToken cancellationToken = default) =>
        ScaleLatent(rawLatent, scale, inverse: false, cancellationToken);

    public static Tensor ProcessLatentOut(Tensor diffusionLatent, double scale, CancellationToken cancellationToken = default) =>
        ScaleLatent(diffusionLatent, scale, inverse: true, cancellationToken);

    private static Tensor ScaleLatent(Tensor latent, double scale, bool inverse, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!double.IsFinite(scale) || scale <= 0) throw new ArgumentOutOfRangeException(nameof(scale));
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        ValidateLatent(latent, nameof(latent));
        return Finish(inverse ? latent / scale : latent * scale, cancellationToken);
    }

    internal static void ValidateLatent(Tensor value, string name)
    {
        ArgumentNullException.ThrowIfNull(value, name);
        if (value.device_type != DeviceType.CPU || value.dtype != ScalarType.Float32 || value.is_sparse ||
            value.dim() != 4 || value.shape[0] <= 0 || value.shape[1] != 4 || value.shape[2] <= 0 || value.shape[3] <= 0)
            throw new ArgumentException("Plain SD latents require a dense CPU/F32 tensor [batch, 4, height, width] with positive dimensions.", name);
    }

    internal static void ValidateSigma(Tensor sigma)
    {
        ArgumentNullException.ThrowIfNull(sigma);
        if (sigma.device_type != DeviceType.CPU || sigma.dtype != ScalarType.Float32 || sigma.is_sparse || sigma.dim() > 1)
            throw new ArgumentException("Sigma requires a dense CPU/F32 scalar or vector.", nameof(sigma));
        if (!sigma.isfinite().all().item<bool>() || sigma.lt(0).any().item<bool>())
            throw new ArgumentException("Sigma must be nonnegative and finite.", nameof(sigma));
    }

    internal static Tensor ReshapeSigma(Tensor sigma, Tensor latent)
    {
        ValidateSigma(sigma);
        if (sigma.numel() == 1) return sigma.reshape(Array.Empty<long>());
        if (sigma.dim() != 1 || sigma.shape[0] != latent.shape[0])
            throw new ArgumentException("A sigma vector must have one element or match the latent batch.", nameof(sigma));
        return sigma.reshape(latent.shape[0], 1, 1, 1);
    }

    private static void ValidatePair(Tensor first, Tensor second)
    {
        ValidateLatent(first, nameof(first));
        ValidateLatent(second, nameof(second));
        if (!first.shape.SequenceEqual(second.shape))
            throw new ArgumentException("Paired latent tensors must have identical shapes.", nameof(second));
    }

    private static Tensor Finish(Tensor value, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return value.DetachFromDisposeScope();
    }
}
