using ComfySharp.Inference;
using ComfySharp.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

internal enum Sd15PipelineBoundary
{
    PositiveHidden, PositivePooled, NegativeHidden, NegativePooled,
    InitialDiffusionLatent, FinalDiffusionLatent, RawVaeLatent, Image
}

internal sealed record Sd15PipelineConditioning(ClipTokenization Positive, ClipTokenization Negative,
    double GuidanceScale, bool MaximumDenoise);

/// <summary>Three independently owned result wrappers. The image retains its native view layout.</summary>
internal sealed class Sd15PipelineResult(Tensor diffusionLatent, Tensor rawVaeLatent, Tensor image) : IDisposable
{
    internal Tensor DiffusionLatent { get; } = diffusionLatent;
    internal Tensor RawVaeLatent { get; } = rawVaeLatent;
    internal Tensor Image { get; } = image;

    public void Dispose()
    {
        try { Image.Dispose(); }
        finally
        {
            try { RawVaeLatent.Dispose(); }
            finally { DiffusionLatent.Dispose(); }
        }
    }
}

/// <summary>Diagnostic composition for the fixed reduced SD1.5 source corpus only.
/// Uses supplied noise and sigmas; no RNG, scheduler selection, model download or node registration.</summary>
internal static class Sd15PipelineExecution
{
    internal static readonly SdUnetConfig UnetConfig = new(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
    internal static readonly ClassicalVaeConfig VaeConfig = new(32);

    internal static Sd15PipelineConditioning Tokenize(string positive, string negative, double guidanceScale,
        bool maximumDenoise, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(positive);
        ArgumentNullException.ThrowIfNull(negative);
        if (!double.IsFinite(guidanceScale)) throw new ArgumentOutOfRangeException(nameof(guidanceScale));
        cancellationToken.ThrowIfCancellationRequested();
        var tokenizer = new ComfyClipTokenizer(ClipTokenizer.CreateDefault(), ClipProfile.Sd1L);
        var positiveTokens = tokenizer.Tokenize(positive, cancellationToken: cancellationToken);
        var negativeTokens = tokenizer.Tokenize(negative, cancellationToken: cancellationToken);
        return new(positiveTokens, negativeTokens, guidanceScale, maximumDenoise);
    }

    // Captures are synchronous and borrowed: never dispose, mutate or retain their tensors.
    // configureSampler is an internal diagnostic seam for tests that can observe Euler's steps.
    // Neither callback is a public application/plugin API. A callback failure aborts this operation.
    internal static Sd15PipelineResult Run(ComfyClipEncoder clip, SdUnet unet, ComfyImageVae vae,
        Sd15PipelineConditioning conditioning, Tensor noise, Tensor sigmas,
        CancellationToken cancellationToken = default,
        Action<Sd15PipelineBoundary, Tensor>? capture = null, Action<SdEulerSampler>? configureSampler = null)
    {
        ArgumentNullException.ThrowIfNull(clip);
        ArgumentNullException.ThrowIfNull(unet);
        ArgumentNullException.ThrowIfNull(vae);
        ArgumentNullException.ThrowIfNull(conditioning);
        ArgumentNullException.ThrowIfNull(conditioning.Positive);
        ArgumentNullException.ThrowIfNull(conditioning.Negative);
        ArgumentNullException.ThrowIfNull(noise);
        ArgumentNullException.ThrowIfNull(sigmas);
        cancellationToken.ThrowIfCancellationRequested();
        if (clip.Profile != ClipProfile.Sd1L || conditioning.Positive.Profile != ClipProfile.Sd1L ||
            conditioning.Negative.Profile != ClipProfile.Sd1L)
            throw new ArgumentException("The reduced SD1.5 diagnostic requires SD1-L conditioning.");
        if (unet.Config != UnetConfig || vae.Config != VaeConfig)
            throw new ArgumentException("This diagnostic requires its fixed reduced U-Net and VAE configurations.");
        if (!double.IsFinite(conditioning.GuidanceScale))
            throw new ArgumentOutOfRangeException(nameof(conditioning), "CFG scale must be finite.");

        // Acquire independent references before encoding. A failure to acquire a later graph
        // releases every earlier reference, and closing the caller's graphs cannot interrupt us.
        using var clipOwner = clip.Retain();
        using var unetOwner = unet.Retain();
        using var vaeOwner = vae.Retain();
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var inference = no_grad();
        ValidateInputs(noise, sigmas);
        using var denoiser = new SdDenoiser(unetOwner, SdPredictionKind.Epsilon, SdDiscreteSampling.Default);
        using var sampler = new SdEulerSampler(denoiser);
        configureSampler?.Invoke(sampler);
        cancellationToken.ThrowIfCancellationRequested();

        using var positive = clipOwner.Encode(conditioning.Positive, cancellationToken: cancellationToken);
        ValidateHidden(positive.Hidden, nameof(conditioning.Positive));
        Capture(Sd15PipelineBoundary.PositiveHidden, positive.Hidden);
        Capture(Sd15PipelineBoundary.PositivePooled, positive.Pooled);
        // An empty negative prompt is encoded as real empty text. It is never replaced by null.
        using var negative = clipOwner.Encode(conditioning.Negative, cancellationToken: cancellationToken);
        ValidateHidden(negative.Hidden, nameof(conditioning.Negative));
        Capture(Sd15PipelineBoundary.NegativeHidden, negative.Hidden);
        Capture(Sd15PipelineBoundary.NegativePooled, negative.Pooled);

        var emptyLatent = zeros_like(noise);
        var firstSigma = sigmas[0];
        using var initial = SdSamplingMath.NoiseScaling(noise, emptyLatent, firstSigma,
            conditioning.MaximumDenoise, cancellationToken);
        Capture(Sd15PipelineBoundary.InitialDiffusionLatent, initial);
        Tensor? diffusion = null, raw = null, image = null;
        try
        {
            diffusion = sampler.Sample(initial, sigmas, positive.Hidden, negative.Hidden,
                new SdGuidanceOptions { Scale = conditioning.GuidanceScale, BatchMode = SdGuidanceBatchMode.Separate },
                cancellationToken);
            Capture(Sd15PipelineBoundary.FinalDiffusionLatent, diffusion);
            raw = SdSamplingMath.ProcessLatentOut(diffusion, SdSamplingMath.Sd15LatentScale, cancellationToken);
            Capture(Sd15PipelineBoundary.RawVaeLatent, raw);
            image = vaeOwner.Decode(raw, cancellationToken);
            Capture(Sd15PipelineBoundary.Image, image);
            var result = new Sd15PipelineResult(diffusion, raw, image);
            diffusion = raw = image = null;
            return result;
        }
        finally
        {
            try { image?.Dispose(); }
            finally
            {
                try { raw?.Dispose(); }
                finally { diffusion?.Dispose(); }
            }
        }

        void Capture(Sd15PipelineBoundary boundary, Tensor value)
        {
            cancellationToken.ThrowIfCancellationRequested();
            capture?.Invoke(boundary, value);
            cancellationToken.ThrowIfCancellationRequested();
        }
    }

    private static void ValidateInputs(Tensor noise, Tensor sigmas)
    {
        if (noise.device_type != DeviceType.CPU || noise.dtype != ScalarType.Float32 || noise.is_sparse ||
            !noise.shape.SequenceEqual(new long[] { 1, 4, 4, 5 }) || !noise.isfinite().all().item<bool>())
            throw new ArgumentException("The fixed reduced corpus requires finite CPU/F32 noise [1,4,4,5].", nameof(noise));
        if (sigmas.device_type != DeviceType.CPU || sigmas.dtype != ScalarType.Float32 || sigmas.is_sparse ||
            sigmas.dim() != 1 || sigmas.shape[0] is < 2 or > 4 || !sigmas.isfinite().all().item<bool>())
            throw new ArgumentException("The fixed reduced corpus requires a finite CPU/F32 sigma vector with two to four values.", nameof(sigmas));
        var current = sigmas.narrow(0, 0, sigmas.shape[0] - 1);
        var next = sigmas.narrow(0, 1, sigmas.shape[0] - 1);
        if (current.le(0).any().item<bool>() || next.gt(current).any().item<bool>() || sigmas[-1].item<float>() != 0)
            throw new ArgumentException("The diagnostic requires nonincreasing positive current sigmas and a final zero.", nameof(sigmas));
    }

    private static void ValidateHidden(Tensor hidden, string name)
    {
        if (hidden.dim() != 3 || hidden.shape[0] != 1 || hidden.shape[1] <= 0 || hidden.shape[2] != UnetConfig.ContextSize)
            throw new ArgumentException("CLIP hidden output does not match the fixed reduced context width.", name);
    }
}
