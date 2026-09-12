using static TorchSharp.torch;
namespace ComfySharp.Inference;

/// <summary>Frozen DPM-Solver++(2M), Float32 on CPU/CUDA, with one retained prior prediction.
/// Inputs are borrowed; returned output is independently owned. Nonfinite trajectories are diagnosed.</summary>
public sealed class SdDpmpp2MSampler : IDisposable
{
    private readonly object gate = new();
    private SdDenoiser? denoiser;

    public SdDpmpp2MSampler(SdDenoiser denoiser)
    {
        ArgumentNullException.ThrowIfNull(denoiser);
        this.denoiser = denoiser.Retain();
    }

    public Tensor Sample(Tensor initial, Tensor sigmas, Tensor positive, Tensor? negative,
        SdGuidanceOptions? guidance = null, CancellationToken cancellationToken = default, SdInpaintMask? inpaint = null)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(initial);
        SdDenoiser retained;
        lock (gate) retained = (denoiser ?? throw new ObjectDisposedException(nameof(SdDpmpp2MSampler))).Retain();
        using (retained)
        {
            using var paint = inpaint?.Retain();
            InferenceDevice.RequireSame(retained.Device, initial, nameof(initial));
            Tensor Predict(Tensor x, Tensor sigma) => retained.DenoiseGuided(x, sigma, positive, negative, guidance, cancellationToken);
            return Integrate(initial, sigmas,
                (x, sigma) => paint is null ? Predict(x, sigma) : paint.Denoise(x, sigma, Predict, cancellationToken), cancellationToken);
        }
    }

    // The actual integrator accepts an owned prediction from the denoiser. Analytic ODE references
    // exercise this same path without substituting a fake network in a production workflow.
    internal static Tensor Integrate(Tensor initial, Tensor sigmas, Func<Tensor, Tensor, Tensor> denoise,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(denoise);
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var noGrad = no_grad();
        SdSamplingMath.ValidateLatent(initial, nameof(initial)); ArgumentNullException.ThrowIfNull(sigmas);
        InferenceDevice.RequireSame(initial.device, sigmas, nameof(sigmas));
        if (sigmas.dtype != ScalarType.Float32 || sigmas.is_sparse || sigmas.dim() != 1 || sigmas.shape[0] < 2)
            throw new ArgumentException("DPM++ 2M requires a dense Float32 sigma vector of length at least two.", nameof(sigmas));
        var firsts = sigmas.narrow(0,0,sigmas.shape[0]-1); var lasts = sigmas.narrow(0,1,sigmas.shape[0]-1);
        if (!sigmas.isfinite().all().item<bool>() || firsts.le(0).any().item<bool>() ||
            lasts.gt(firsts).any().item<bool>() || sigmas[-1].item<float>() != 0)
            throw new ArgumentException("DPM++ 2M requires finite nonincreasing sigmas, positive before terminal zero.", nameof(sigmas));
        var units = ones(new[] { initial.shape[0] }, dtype: ScalarType.Float32, device: initial.device);
        Tensor current = initial; Tensor? history = null;
        for (long step=0; step<sigmas.shape[0]-1; step++)
        {
            cancellationToken.ThrowIfCancellationRequested(); using var local = NewDisposeScope();
            var sigma = sigmas[step]; var nextSigma = sigmas[step+1];
            using var prediction = denoise(current, sigma * units);
            cancellationToken.ThrowIfCancellationRequested();
            var time = sigma.log().neg(); var nextTime = nextSigma.log().neg(); var h = nextTime-time;
            // Keep log/exp and expm1 exactly in source order; do not replace this with sigma ratios.
            var scale = nextTime.neg().exp() / time.neg().exp();
            Tensor adjusted = prediction;
            if (history is not null && nextSigma.item<float>() != 0)
            {
                var previousH = time - sigmas[step-1].log().neg(); var ratio = previousH / h;
                adjusted = (1 + 1 / (2 * ratio)) * prediction - (1 / (2 * ratio)) * history;
            }
            var next = scale * current - h.neg().expm1() * adjusted;
            cancellationToken.ThrowIfCancellationRequested();
            if (!next.isfinite().all().item<bool>())
                throw new ArithmeticException($"DPM++ 2M produced a nonfinite trajectory at step {step}; repeated sigmas can make its multistep ratio undefined.");
            // The prediction wrapper is disposed locally; its alias keeps the storage for one more step.
            var remembered = prediction.alias().MoveToOuterDisposeScope();
            next.MoveToOuterDisposeScope(); history?.Dispose(); history = remembered;
            if (!ReferenceEquals(current, initial)) current.Dispose(); current = next;
        }
        cancellationToken.ThrowIfCancellationRequested();
        return current.DetachFromDisposeScope();
    }

    public void Dispose()
    {
        SdDenoiser? owned;
        lock (gate) { owned=denoiser; denoiser=null; }
        owned?.Dispose();
    }
}
