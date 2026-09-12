using static TorchSharp.torch;
namespace ComfySharp.Inference;

/// <summary>Frozen sample_heun with its default s_churn=0, CPU/CUDA Float32.
/// Inputs are borrowed. Every nonterminal interval evaluates the denoiser twice;
/// the terminal interval uses Euler. Synchronous native calls cannot be interrupted.</summary>
public sealed class SdHeunSampler : IDisposable
{
    private readonly object gate = new();
    private SdDenoiser? denoiser;

    public SdHeunSampler(SdDenoiser denoiser)
    {
        ArgumentNullException.ThrowIfNull(denoiser);
        this.denoiser = denoiser.Retain();
    }

    public Tensor Sample(Tensor initial, Tensor sigmas, Tensor positive, Tensor? negative,
        SdGuidanceOptions? guidance = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        SdDenoiser retained;
        lock (gate) retained = (denoiser ?? throw new ObjectDisposedException(nameof(SdHeunSampler))).Retain();
        using (retained)
        {
            InferenceDevice.RequireSame(retained.Device, initial, nameof(initial));
            return Integrate(initial, sigmas,
                (x, sigma) => retained.DenoiseGuided(x, sigma, positive, negative, guidance, cancellationToken), cancellationToken);
        }
    }

    // The numerical trajectory is shared by actual model execution and analytical ODE contracts.
    // The callable returns an owned tensor, without retaining or mutating its borrowed inputs.
    internal static Tensor Integrate(Tensor initial, Tensor sigmas, Func<Tensor, Tensor, Tensor> denoise,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(denoise);
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        SdSamplingMath.ValidateLatent(initial, nameof(initial));
        ArgumentNullException.ThrowIfNull(sigmas);
        InferenceDevice.RequireSame(initial.device, sigmas, nameof(sigmas));
        if (sigmas.dtype != ScalarType.Float32 || sigmas.is_sparse || sigmas.dim() != 1 || sigmas.shape[0] < 2)
            throw new ArgumentException("Heun requires a dense Float32 vector with at least two sigmas.", nameof(sigmas));
        var firsts = sigmas.narrow(0, 0, sigmas.shape[0] - 1);
        var lasts = sigmas.narrow(0, 1, sigmas.shape[0] - 1);
        if (!sigmas.isfinite().all().item<bool>() || firsts.le(0).any().item<bool>() ||
            lasts.gt(firsts).any().item<bool>() || sigmas[-1].item<float>() != 0)
            throw new ArgumentException("Heun requires finite nonincreasing sigmas, positive before a final zero.", nameof(sigmas));
        var units = ones(new[] { initial.shape[0] }, dtype: ScalarType.Float32, device: initial.device);
        Tensor current = initial;
        for (long step = 0; step < sigmas.shape[0] - 1; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var local = NewDisposeScope();
            // Preserve the source's sigma_hat multiplication even when gamma is zero.
            var sigma = sigmas[step] * 1;
            using var prediction = denoise(current, sigma * units);
            cancellationToken.ThrowIfCancellationRequested();
            var derivative = (current - prediction) / sigma.reshape(1, 1, 1, 1);
            var nextSigma = sigmas[step + 1]; var delta = nextSigma - sigma;
            Tensor next;
            if (nextSigma.item<float>() == 0)
                next = current + derivative * delta;
            else
            {
                var predictor = current + derivative * delta;
                cancellationToken.ThrowIfCancellationRequested();
                using var corrected = denoise(predictor, nextSigma * units);
                cancellationToken.ThrowIfCancellationRequested();
                var second = (predictor - corrected) / nextSigma.reshape(1, 1, 1, 1);
                var averaged = (derivative + second) / 2;
                next = current + averaged * delta;
            }
            cancellationToken.ThrowIfCancellationRequested();
            next.MoveToOuterDisposeScope();
            if (!ReferenceEquals(current, initial)) current.Dispose();
            current = next;
        }
        cancellationToken.ThrowIfCancellationRequested();
        return current.DetachFromDisposeScope();
    }

    public void Dispose()
    {
        SdDenoiser? owned;
        lock (gate) { owned = denoiser; denoiser = null; }
        owned?.Dispose();
    }
}
