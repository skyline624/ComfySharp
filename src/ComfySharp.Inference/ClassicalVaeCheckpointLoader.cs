using TorchSharp;

namespace ComfySharp.Inference;

public enum ClassicalVaeCheckpointLayout { Canonical, FirstStageModel, Diffusers }

/// <summary>Validated metadata for complete encoder, decoder and legacy quant/post-quant weights.</summary>
public sealed class ClassicalVaeWeightPlan
{
    internal SdModelWeightPlan Plan { get; }
    public ClassicalVaeConfig Config { get; }
    public ClassicalVaeCheckpointLayout Layout { get; }
    public IReadOnlyList<SdModelWeightMapping> Mappings => Plan.Mappings;
    public long SourceBytes => Plan.SourceBytes;
    public long ResidentBytes => Plan.ResidentBytes;
    public long TemporaryBytes => Plan.TemporaryBytes;
    public long EstimatedPeakWeightBytes => checked(ResidentBytes + TemporaryBytes);
    internal ClassicalVaeWeightPlan(ClassicalVaeConfig config, ClassicalVaeCheckpointLayout layout, SdModelWeightPlan plan)
    { Config = config; Layout = layout; Plan = plan; }
}

public static class ClassicalVaeCheckpointLoader
{
    public static ClassicalVaeWeightPlan Inspect(SafeTensorFile file, ClassicalVaeConfig config, ClassicalVaeCheckpointLayout layout)
    {
        if (!Enum.IsDefined(layout)) throw new ArgumentOutOfRangeException(nameof(layout));
        return new(config, layout, SdModelWeightLoading.Inspect(file, ClassicalVaeWeightSchema.Define(config),
            layout == ClassicalVaeCheckpointLayout.FirstStageModel ? "first_stage_model." : "",
            layout == ClassicalVaeCheckpointLayout.Diffusers, nestedQuantAliases: true));
    }

    public static ClassicalVaeWeightSet Load(SafeTensorFile file, ClassicalVaeWeightPlan plan, CancellationToken cancellationToken = default)
        => Load(file, plan, cancellationToken, null);

    internal static ClassicalVaeWeightSet Load(SafeTensorFile file, ClassicalVaeWeightPlan plan, CancellationToken cancellationToken,
        Action<IReadOnlyDictionary<string, torch.Tensor>>? materialized)
    {
        ArgumentNullException.ThrowIfNull(plan);
        return SdModelWeightLoading.Load(file, plan.Plan, cancellationToken,
            tensors => ClassicalVaeWeightSet.FromOwnedTensors(plan.Config, tensors), materialized);
    }
}
