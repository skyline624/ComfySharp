using static TorchSharp.torch;

namespace ComfySharp.Inference;

public sealed record SdLoraTrainingOptions
{
    public int Steps { get; init; } = 16;
    public int BatchSize { get; init; } = 1;
    public int AccumulationSteps { get; init; } = 1;
    public double LearningRate { get; init; } = .0005;
    public ulong Seed { get; init; }
    public string Optimizer { get; init; } = "AdamW";
    public string Loss { get; init; } = "MSE";
}
public sealed record SdLoraTrainingProgress(long Microbatch, long OptimizerSteps, float Loss);
public sealed record SdLoraTrainingResult(IReadOnlyList<float> Losses, long OptimizerSteps, long Microbatches);

/// <summary>Plain Float32 SD dataset loop with one full-image text context per image.
/// Requires caller-owned initialized adapters. Region conditioning, other adapters, precision modes,
/// checkpointing, offload and training-node integration are separate capabilities.</summary>
public static class SdLoraTrainingLoop
{
    public static SdLoraTrainingResult Run(SdDenoiser denoiser, SdTrainingDataset dataset, Tensor context,
        IReadOnlyDictionary<string, TrainableLoraPatch> patches, SdLoraTrainingOptions? options = null,
        Action<SdLoraTrainingProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(denoiser); ArgumentNullException.ThrowIfNull(dataset);
        ArgumentNullException.ThrowIfNull(context); ArgumentNullException.ThrowIfNull(patches); options ??= new();
        if (options.Steps is < 1 or > 100000) throw new ArgumentOutOfRangeException(nameof(options), "Training steps must be between 1 and 100000.");
        if (options.Loss is not ("MSE" or "L1" or "Huber" or "SmoothL1")) throw new ArgumentException("Unknown training loss.", nameof(options));
        if (!InferenceDevice.IsSupported(context.device_type) || context.dtype != ScalarType.Float32 || context.is_sparse || context.dim() != 3 ||
            context.shape[1] <= 0 || context.shape[2] != denoiser.Config.ContextSize ||
            (context.shape[0] != dataset.Count && !(context.shape[0] == 1 && dataset.Mode != SdTrainingDatasetMode.Buckets)))
            throw new ArgumentException("Provide one dense Float32 full-image text context per image, or a single shared context outside bucket mode.", nameof(context));
        using var model = denoiser.Retain(); using var source = dataset.Retain();
        using var scope = NewDisposeScope(); using var enabled = set_grad_enabled(true);
        using var optimizer = new LoraTrainingOptimizer(patches.Values, options.Optimizer, options.LearningRate, options.AccumulationSteps);
        using var sampler = new SdTrainingBatchSampler(source, options.Seed, options.BatchSize, model.Device, cancellationToken: cancellationToken);
        var losses = new List<float>(); long total = checked((long)options.Steps * options.AccumulationSteps);
        for (long i = 0; i < total; i++)
        {
            cancellationToken.ThrowIfCancellationRequested(); using var iteration = NewDisposeScope(); using var batch = sampler.Next(cancellationToken);
            Tensor? totalLoss = null;
            foreach (var group in batch.Groups)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var indices = tensor(group.Indices.ToArray(), dtype: ScalarType.Int64, device: context.device);
                using var detachedContext = context.detach();
                using var selected = context.shape[0] == 1 ? detachedContext.repeat(group.Indices.Count, 1, 1) : detachedContext.index_select(0, indices);
                using var onDevice = selected.to(model.Device);
                var loss = SdLoraTrainingObjective.CalculateLoss(model, group.Latent, group.Noise, group.Sigmas, onDevice, patches, options.Loss, cancellationToken: cancellationToken);
                totalLoss = totalLoss is null ? loss : totalLoss + loss;
            }
            optimizer.AccumulateMean(totalLoss!, batch.Groups.Count, cancellationToken);
            if (optimizer.PendingMicrobatches == options.AccumulationSteps) optimizer.Step(cancellationToken);
            // Source multi-resolution callback receives its already accumulation-normalized mean;
            // standard/bucket callbacks receive the raw loss. Preserve this observable distinction.
            float reported = source.Mode == SdTrainingDatasetMode.MultiResolution
                ? (totalLoss! / options.AccumulationSteps / batch.Groups.Count).item<float>()
                : totalLoss!.item<float>();
            losses.Add(reported); progress?.Invoke(new(i + 1, optimizer.CompletedSteps, reported));
        }
        return new(losses.AsReadOnly(), optimizer.CompletedSteps, total);
    }
}
