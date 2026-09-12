using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

public enum SdGuidanceBatchMode { Separate, ConcatenateCompatible }

public sealed record SdGuidanceOptions
{
    public double Scale { get; init; } = 7;
    public bool DisableScaleOneOptimization { get; init; }
    public SdGuidanceBatchMode BatchMode { get; init; } = SdGuidanceBatchMode.Separate;
}

/// <summary>A retained plain SD U-Net with the default discrete schedule and explicit EPS/V prediction.
/// Supports whole-image text conditioning. Regions, masks, hooks, controls and adapters are separate capabilities.</summary>
public sealed class SdDenoiser : IDisposable
{
    private readonly object gate = new();
    private SdUnet? model;

    public SdDenoiser(SdUnet model, SdPredictionKind predictionKind, SdDiscreteSampling? sampling = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (!Enum.IsDefined(predictionKind)) throw new ArgumentOutOfRangeException(nameof(predictionKind));
        this.model = model.Retain();
        Config = model.Config;
        PredictionKind = predictionKind;
        Sampling = sampling ?? SdDiscreteSampling.Default;
    }

    public SdUnetConfig Config { get; }
    public SdPredictionKind PredictionKind { get; }
    public SdDiscreteSampling Sampling { get; }
    public Device Device { get { using var graph = RetainModel(); return graph.Device; } }

    public SdDenoiser Retain()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(model is null, this);
            return new SdDenoiser(model, PredictionKind, Sampling);
        }
    }

    public Tensor Denoise(Tensor latent, Tensor sigma, Tensor context, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var operation = RetainModel();
        return Apply(operation, latent, sigma, context, cancellationToken);
    }

    /// <summary>Differentiable whole-image EPS/V denoising with caller-owned LoRA leaves.
    /// Input/sigma/context gradients are preserved. Dataset ownership and detachment belong to the training caller.</summary>
    public Tensor DenoiseForTraining<TPatch>(Tensor latent, Tensor sigma, Tensor context,
        IReadOnlyDictionary<string, TPatch> patches, long maxPatchedWeightBytes = 512L * 1024 * 1024,
        CancellationToken cancellationToken = default, bool bypassMode = false) where TPatch : TrainableWeightPatch
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(patches);
        using var operation = RetainModel();
        return Apply(operation, latent, sigma, context, cancellationToken, TrainableWeightPatch.Widen(patches), maxPatchedWeightBytes, bypassMode);
    }

    public Tensor DenoiseGuided(Tensor latent, Tensor sigma, Tensor conditional, Tensor? unconditional,
        SdGuidanceOptions? options = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var operation = RetainModel();
        options ??= new();
        if (!double.IsFinite(options.Scale)) throw new ArgumentOutOfRangeException(nameof(options), "CFG scale must be finite.");
        if (!Enum.IsDefined(options.BatchMode)) throw new ArgumentOutOfRangeException(nameof(options), "Unknown CFG batch mode.");
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        SdSamplingMath.ValidateLatent(latent, nameof(latent));
        _ = SdSamplingMath.ReshapeSigma(sigma, latent);
        ValidateContext(conditional, latent.shape[0], nameof(conditional));
        InferenceDevice.RequireSame(operation.Device, conditional, nameof(conditional));
        bool omitted = unconditional is null || SdSamplingMath.CanOmitUnconditional(options.Scale, options.DisableScaleOneOptimization);
        if (omitted)
        {
            using var positive = Apply(operation, latent, sigma, conditional, cancellationToken);
            return SdSamplingMath.Guide(positive, null, options.Scale, cancellationToken);
        }

        ValidateContext(unconditional!, latent.shape[0], nameof(unconditional));
        InferenceDevice.RequireSame(operation.Device, unconditional!, nameof(unconditional));
        if (options.BatchMode == SdGuidanceBatchMode.ConcatenateCompatible &&
            TryCommonContextLength(conditional.shape[1], unconditional!.shape[1], out long common))
        {
            // Source CONDCrossAttn repeats whole token sequences to their LCM; it never zero-pads.
            var positiveContext = conditional.shape[1] == common ? conditional : conditional.repeat(1, common / conditional.shape[1], 1);
            var negativeContext = unconditional.shape[1] == common ? unconditional : unconditional.repeat(1, common / unconditional.shape[1], 1);
            var contexts = cat(new[] { positiveContext, negativeContext }, 0);
            var inputs = cat(new[] { latent, latent }, 0);
            var sigmas = sigma.numel() == 1 ? sigma : cat(new[] { sigma, sigma }, 0);
            using var predictions = Apply(operation, inputs, sigmas, contexts, cancellationToken);
            var positive = predictions.narrow(0, 0, latent.shape[0]);
            var negative = predictions.narrow(0, latent.shape[0], latent.shape[0]);
            return SdSamplingMath.Guide(positive, negative, options.Scale, cancellationToken);
        }

        // This explicit memory-conservative policy avoids guessing ComfyUI's hardware-dependent
        // batch heuristic. ConcatenateCompatible also falls back here when its LCM limit is exceeded.
        using var conditionalResult = Apply(operation, latent, sigma, conditional, cancellationToken);
        using var unconditionalResult = Apply(operation, latent, sigma, unconditional!, cancellationToken);
        return SdSamplingMath.Guide(conditionalResult, unconditionalResult, options.Scale, cancellationToken);
    }

    private Tensor Apply(SdUnet operation, Tensor latent, Tensor sigma, Tensor context, CancellationToken cancellationToken,
        IReadOnlyDictionary<string, TrainableWeightPatch>? patches = null, long maxPatchedWeightBytes = 0, bool bypassMode = false)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var gradMode = set_grad_enabled(patches is not null);
        SdSamplingMath.ValidateLatent(latent, nameof(latent));
        ValidateContext(context, latent.shape[0], nameof(context));
        InferenceDevice.RequireSame(operation.Device, latent, nameof(latent));
        InferenceDevice.RequireSame(operation.Device, context, nameof(context));
        using var input = patches is null
            ? SdSamplingMath.ScaleInput(latent, sigma, cancellationToken)
            : SdSamplingMath.ScaleInputForTraining(latent, sigma, cancellationToken);
        using var indices = Sampling.Timestep(sigma, cancellationToken);
        var time = indices.to_type(ScalarType.Float32).reshape(-1);
        using var prediction = patches is null
            ? operation.Forward(input, time, context, cancellationToken)
            : operation.ForwardForTraining(input, time, context, patches, maxPatchedWeightBytes, cancellationToken, bypassMode);
        return patches is null
            ? SdSamplingMath.Denoised(latent, prediction, sigma, PredictionKind, cancellationToken)
            : SdSamplingMath.DenoisedForTraining(latent, prediction, sigma, PredictionKind, cancellationToken);
    }

    private void ValidateContext(Tensor context, long batch, string name)
    {
        ArgumentNullException.ThrowIfNull(context, name);
        if (!InferenceDevice.IsSupported(context.device_type) || context.dtype != ScalarType.Float32 || context.is_sparse ||
            context.dim() != 3 || context.shape[0] != batch || context.shape[1] <= 0 || context.shape[2] != Config.ContextSize)
            throw new ArgumentException($"Text context must be a dense CPU or CUDA Float32 tensor [batch, positive token count, {Config.ContextSize}].", name);
    }

    internal static bool TryCommonContextLength(long first, long second, out long common)
    {
        common = 0;
        if (first <= 0 || second <= 0) return false;
        long small = Math.Min(first, second), large = Math.Max(first, second);
        long a = small, b = large;
        while (b != 0) (a, b) = (b, a % b);
        long factor = large / a;
        if (factor > 4 || small > long.MaxValue / factor) return false;
        common = small * factor;
        return true;
    }

    private SdUnet RetainModel()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(model is null, this);
            return model.Retain();
        }
    }

    public void Dispose()
    {
        SdUnet? released;
        lock (gate) { released = model; model = null; }
        released?.Dispose();
    }
}
