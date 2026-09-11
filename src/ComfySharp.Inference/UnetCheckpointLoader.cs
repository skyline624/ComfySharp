using TorchSharp;

namespace ComfySharp.Inference;

public enum UnetCheckpointLayout { Canonical, ModelDiffusionModel, Diffusers }

/// <summary>Immutable metadata plan for one complete U-Net; estimates exclude activations and the native runtime.</summary>
public sealed class UnetWeightPlan
{
    internal SdModelWeightPlan Plan { get; }
    public SdUnetConfig Config { get; }
    public UnetCheckpointLayout Layout { get; }
    public IReadOnlyList<SdModelWeightMapping> Mappings => Plan.Mappings;
    public long SourceBytes => Plan.SourceBytes;
    public long ResidentBytes => Plan.ResidentBytes;
    public long TemporaryBytes => Plan.TemporaryBytes;
    public long EstimatedPeakWeightBytes => checked(ResidentBytes + TemporaryBytes);
    internal UnetWeightPlan(SdUnetConfig config, UnetCheckpointLayout layout, SdModelWeightPlan plan)
    { Config = config; Layout = layout; Plan = plan; }
}

/// <summary>Strict explicit plain SD1/SD2 adapters; prediction kind is deliberately not inferred from weights.</summary>
public static class UnetCheckpointLoader
{
    public static UnetWeightPlan Inspect(SafeTensorFile file, SdUnetConfig config, UnetCheckpointLayout layout)
    {
        if (!Enum.IsDefined(layout)) throw new ArgumentOutOfRangeException(nameof(layout));
        return new(config, layout, SdModelWeightLoading.Inspect(file, UnetWeightSchema.Define(config),
            layout == UnetCheckpointLayout.ModelDiffusionModel ? "model.diffusion_model." : "",
            layout == UnetCheckpointLayout.Diffusers, nestedQuantAliases: false));
    }

    public static UnetWeightSet Load(SafeTensorFile file, UnetWeightPlan plan, CancellationToken cancellationToken = default)
        => Load(file, plan, cancellationToken, null);

    internal static UnetWeightSet Load(SafeTensorFile file, UnetWeightPlan plan, CancellationToken cancellationToken,
        Action<IReadOnlyDictionary<string, torch.Tensor>>? materialized)
    {
        ArgumentNullException.ThrowIfNull(plan);
        return SdModelWeightLoading.Load(file, plan.Plan, cancellationToken,
            tensors => UnetWeightSet.FromOwnedTensors(plan.Config, tensors), materialized);
    }
}
