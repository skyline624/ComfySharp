using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>
/// CPU float32 sigma generators from ComfyUI's frozen k_diffusion source. Returned tensors
/// belong to the caller (or its active dispose scope). These functions do not execute a sampler.
/// </summary>
/// <remarks>
/// Source: https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py#L23
/// Steps are limited to the upstream nodes' range, 1 through 10000. Unlike upstream's unchecked
/// helpers, invalid mathematical domains and non-finite outputs are rejected explicitly.
/// Sigma bounds are not reordered; callers may request an ascending schedule.
/// </remarks>
public static class SigmaSchedules
{
    /// <summary>Karras power schedule, followed by one zero. Requires nonnegative bounds and rho &gt; 0.</summary>
    public static Tensor Karras(int steps, double sigmaMin, double sigmaMax, double rho = 7,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ValidateSteps(steps);
        ValidateNonnegative(sigmaMin, nameof(sigmaMin));
        ValidateNonnegative(sigmaMax, nameof(sigmaMax));
        ValidatePositive(rho, nameof(rho));
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var ramp = linspace(0, 1, steps, dtype: ScalarType.Float32, device: CPU);
        double minInvRho = Math.Pow(sigmaMin, 1 / rho);
        double maxInvRho = Math.Pow(sigmaMax, 1 / rho);
        var sigmas = (maxInvRho + ramp * (minInvRho - maxInvRho)).pow(rho);
        return Complete(sigmas, appendZero: true, nameof(Karras), cancellationToken).MoveToOuterDisposeScope();
    }

    /// <summary>Log-uniform schedule, followed by one zero. Both sigma bounds must be positive.</summary>
    public static Tensor Exponential(int steps, double sigmaMin, double sigmaMax,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ValidateSteps(steps);
        ValidatePositive(sigmaMin, nameof(sigmaMin));
        ValidatePositive(sigmaMax, nameof(sigmaMax));
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var sigmas = linspace(Math.Log(sigmaMax), Math.Log(sigmaMin), steps,
            dtype: ScalarType.Float32, device: CPU).exp();
        return Complete(sigmas, appendZero: true, nameof(Exponential), cancellationToken).MoveToOuterDisposeScope();
    }

    /// <summary>Polynomial interpolation in log-sigma, followed by one zero. Rho may be zero.</summary>
    public static Tensor Polyexponential(int steps, double sigmaMin, double sigmaMax, double rho = 1,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ValidateSteps(steps);
        ValidatePositive(sigmaMin, nameof(sigmaMin));
        ValidatePositive(sigmaMax, nameof(sigmaMax));
        ValidateNonnegative(rho, nameof(rho));
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var ramp = linspace(1, 0, steps, dtype: ScalarType.Float32, device: CPU).pow(rho);
        var sigmas = (ramp * (Math.Log(sigmaMax) - Math.Log(sigmaMin)) + Math.Log(sigmaMin)).exp();
        return Complete(sigmas, appendZero: true, nameof(Polyexponential), cancellationToken).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Clamped Laplace schedule. Returns exactly steps values; upstream does not append a terminal zero.
    /// Bounds and beta must be nonnegative; mu must be finite.
    /// </summary>
    public static Tensor Laplace(int steps, double sigmaMin, double sigmaMax, double mu = 0, double beta = 0.5,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ValidateSteps(steps);
        ValidateNonnegative(sigmaMin, nameof(sigmaMin));
        ValidateNonnegative(sigmaMax, nameof(sigmaMax));
        ValidateFinite(mu, nameof(mu));
        ValidateNonnegative(beta, nameof(beta));
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var x = linspace(0, 1, steps, dtype: ScalarType.Float32, device: CPU);
        var lambda = mu - beta * (0.5 - x).sign() * (1 - 2 * (0.5 - x).abs() + 1e-5).log();
        var sigmas = lambda.exp().clamp(min: sigmaMin, max: sigmaMax);
        return Complete(sigmas, appendZero: false, nameof(Laplace), cancellationToken).MoveToOuterDisposeScope();
    }

    /// <summary>Continuous variance-preserving schedule, followed by one zero. EpsS is in [0, 1].</summary>
    public static Tensor VP(int steps, double betaD = 19.9, double betaMin = 0.1, double epsS = 1e-3,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ValidateSteps(steps);
        ValidateNonnegative(betaD, nameof(betaD));
        ValidateNonnegative(betaMin, nameof(betaMin));
        ValidateNonnegative(epsS, nameof(epsS));
        if (epsS > 1) throw new ArgumentOutOfRangeException(nameof(epsS), "EpsS must be in [0, 1].");
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var t = linspace(1, epsS, steps, dtype: ScalarType.Float32, device: CPU);
        var sigmas = (betaD * t.pow(2) / 2 + betaMin * t).expm1().sqrt();
        return Complete(sigmas, appendZero: true, nameof(VP), cancellationToken).MoveToOuterDisposeScope();
    }

    private static Tensor Complete(Tensor sigmas, bool appendZero, string name, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var result = appendZero
            ? cat(new[] { sigmas, zeros(new long[] { 1 }, dtype: ScalarType.Float32, device: CPU) })
            : sigmas;
        if (!result.isfinite().all().item<bool>())
            throw new ArithmeticException($"{name} produced a non-finite float32 sigma schedule.");
        cancellationToken.ThrowIfCancellationRequested();
        return result;
    }

    private static void ValidateSteps(int steps)
    {
        if (steps is < 1 or > 10000)
            throw new ArgumentOutOfRangeException(nameof(steps), "Steps must be between 1 and 10000.");
    }

    private static void ValidatePositive(double value, string name)
    {
        if (!double.IsFinite(value) || value <= 0)
            throw new ArgumentOutOfRangeException(name, "Value must be finite and positive.");
    }

    private static void ValidateNonnegative(double value, string name)
    {
        if (!double.IsFinite(value) || value < 0)
            throw new ArgumentOutOfRangeException(name, "Value must be finite and nonnegative.");
    }

    private static void ValidateFinite(double value, string name)
    {
        if (!double.IsFinite(value))
            throw new ArgumentOutOfRangeException(name, "Value must be finite.");
    }
}
