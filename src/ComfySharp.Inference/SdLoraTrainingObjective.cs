using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Plain SD Float32 denoised-latent objective from TrainSampler.fwd_bwd.
/// Latents must already be in diffusion space, not raw VAE space. Dataset selection,
/// RNG, latent-format conversion, mixed precision and conditioning regions are separate responsibilities.</summary>
public static class SdLoraTrainingObjective
{
    /// <summary>Returns an unnormalized scalar loss. LoraTrainingOptimizer owns accumulation scaling.
    /// Inputs are borrowed without mutating values, grad flags or gradient buffers. Only adapter leaves train.</summary>
    public static Tensor CalculateLoss(SdDenoiser denoiser, Tensor diffusionLatent, Tensor noise, Tensor sigma, Tensor context,
        IReadOnlyDictionary<string, TrainableLoraPatch> patches, string lossName,
        long maxPatchedWeightBytes = 512L * 1024 * 1024, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(denoiser);
        ArgumentNullException.ThrowIfNull(patches); ArgumentNullException.ThrowIfNull(context);
        if (lossName is not ("MSE" or "L1" or "Huber" or "SmoothL1")) throw new ArgumentException("Unknown training loss.", nameof(lossName));
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var enabled = set_grad_enabled(true);
        using var noisy = SdSamplingMath.NoiseScaling(noise, diffusionLatent, sigma, maximumDenoise: false, cancellationToken);
        using var target = SdSamplingMath.NoiseScaling(zeros_like(noise), diffusionLatent, zeros_like(sigma), maximumDenoise: false, cancellationToken);
        using var trainingSigma = sigma.detach().clone().requires_grad_();
        using var trainingContext = context.detach();
        noisy.requires_grad_();
        using var prediction = denoiser.DenoiseForTraining(noisy, trainingSigma, trainingContext, patches, maxPatchedWeightBytes, cancellationToken);
        var loss = TrainingLoss.Calculate(lossName, prediction, target);
        cancellationToken.ThrowIfCancellationRequested();
        return loss.MoveToOuterDisposeScope();
    }
}
