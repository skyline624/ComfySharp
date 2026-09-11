using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Plain no-churn Euler trajectory over an already initialized diffusion latent.
/// CPU/F32 only; the explicit nonincreasing schedule must end at zero. This component does not
/// initialize noise, choose a scheduler, load a checkpoint or convert VAE latent scales.</summary>
public sealed class SdEulerSampler : IDisposable
{
    private readonly object gate = new();
    private SdDenoiser? denoiser;

    public SdEulerSampler(SdDenoiser denoiser)
    {
        ArgumentNullException.ThrowIfNull(denoiser);
        this.denoiser = denoiser.Retain();
        Config = denoiser.Config;
        PredictionKind = denoiser.PredictionKind;
    }

    public SdUnetConfig Config { get; }
    public SdPredictionKind PredictionKind { get; }

    // Diagnostic values are borrowed, before the Euler update. An observer must not
    // mutate, retain or dispose them. No tensor callback is part of the public API.
    internal Action<long, Tensor, Tensor, Tensor>? DiagnosticObserver { get; set; }

    public SdEulerSampler Retain()
    {
        lock (gate) return new(denoiser ?? throw new ObjectDisposedException(nameof(SdEulerSampler)));
    }

    /// <summary>Inputs remain borrowed and unchanged. The returned wrapper belongs to the caller,
    /// independently of this sampler and ambient tensor scopes. Cancellation is checked between
    /// synchronous native operations, which cannot be interrupted in their middle.</summary>
    public Tensor Sample(Tensor initialDiffusionLatent, Tensor sigmas, Tensor positiveContext,
        Tensor? negativeContext, SdGuidanceOptions? guidance = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var operation = RetainDenoiser();
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        SdSamplingMath.ValidateLatent(initialDiffusionLatent, nameof(initialDiffusionLatent));
        ValidateSchedule(sigmas);
        cancellationToken.ThrowIfCancellationRequested();
        var observer = DiagnosticObserver;
        var sigmaBatchUnits = ones(new[] { initialDiffusionLatent.shape[0] }, dtype: ScalarType.Float32, device: CPU);
        Tensor current = initialDiffusionLatent;
        bool ownsCurrent = false;
        try
        {
            for (long step = 0; step < sigmas.shape[0] - 1; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var stepScope = NewDisposeScope();
                var sigma = sigmas[step];
                var modelSigma = sigma * sigmaBatchUnits;
                using var denoised = operation.DenoiseGuided(current, modelSigma, positiveContext, negativeContext,
                    guidance, cancellationToken);
                // Keep source Tensor/F32 arithmetic and evaluation order, including the last
                // interval. NativeMath.EulerStep is a separate scalar-double foundation API.
                var derivative = (current - denoised) / sigma.reshape(1, 1, 1, 1);
                observer?.Invoke(step, current, denoised, sigma);
                cancellationToken.ThrowIfCancellationRequested();
                var delta = sigmas[step + 1] - sigma;
                var next = current + derivative * delta;
                cancellationToken.ThrowIfCancellationRequested();
                next.MoveToOuterDisposeScope();
                if (ownsCurrent) current.Dispose();
                current = next;
                ownsCurrent = true;
            }
            cancellationToken.ThrowIfCancellationRequested();
            var result = current.DetachFromDisposeScope();
            ownsCurrent = false;
            return result;
        }
        finally
        {
            if (ownsCurrent) current.Dispose();
        }
    }

    private static void ValidateSchedule(Tensor sigmas)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        using var scope = NewDisposeScope();
        if (sigmas.device_type != DeviceType.CPU || sigmas.dtype != ScalarType.Float32 || sigmas.is_sparse ||
            sigmas.dim() != 1 || sigmas.shape[0] < 2)
            throw new ArgumentException("Euler requires a dense CPU/F32 vector containing at least two sigmas.", nameof(sigmas));
        var current = sigmas.narrow(0, 0, sigmas.shape[0] - 1);
        var next = sigmas.narrow(0, 1, sigmas.shape[0] - 1);
        if (!sigmas.isfinite().all().item<bool>() || current.le(0).any().item<bool>() ||
            next.gt(current).any().item<bool>() || sigmas[-1].item<float>() != 0)
            throw new ArgumentException("This Euler trajectory requires finite nonincreasing sigmas, positive before a final zero.", nameof(sigmas));
    }

    private SdDenoiser RetainDenoiser()
    {
        lock (gate) return (denoiser ?? throw new ObjectDisposedException(nameof(SdEulerSampler))).Retain();
    }

    public void Dispose()
    {
        SdDenoiser? owned;
        lock (gate) { owned = denoiser; denoiser = null; }
        owned?.Dispose();
    }
}
