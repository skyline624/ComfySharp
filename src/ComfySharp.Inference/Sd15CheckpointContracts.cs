namespace ComfySharp.Inference;

/// <summary>Handling of tensors outside the three explicitly selected component namespaces.</summary>
public enum Sd15UnclaimedTensorHandling
{
    Reject,
    ReportAndIgnoreOutsideComponents
}

public sealed record Sd15CheckpointInspectionOptions
{
    public Sd15UnclaimedTensorHandling UnclaimedTensors { get; init; } = Sd15UnclaimedTensorHandling.Reject;
}

/// <summary>Metadata-only plan for fixed CLIP-L, SD1.5 U-Net and classical VAE configurations.
/// No prediction kind, latent scale, schedule or numerical qualification is inferred.</summary>
public sealed class Sd15CheckpointPlan
{
    internal SdCheckpointAssemblyPlan Assembly { get; }
    public ClipWeightPlan Clip => Assembly.Clip;
    public UnetWeightPlan Unet => Assembly.Unet;
    public ClassicalVaeWeightPlan Vae => Assembly.Vae;
    public bool HasClipProjection => Clip.HasProjection;
    public Sd15UnclaimedTensorHandling UnclaimedTensorHandling => Assembly.UnclaimedTensorHandling;
    /// <summary>Names deliberately omitted outside the selected component namespaces.</summary>
    public IReadOnlyList<string> UnclaimedTensorNames => Assembly.UnclaimedTensorNames;
    /// <summary>Payload bytes selected for loading, excluding explicitly ignored tensors.</summary>
    public long SourceBytes => Assembly.SourceBytes;
    public long ResidentBytes => Assembly.ResidentBytes;
    public long TemporaryBytes => Assembly.TemporaryBytes;
    /// <summary>Conservative peak for weights only; excludes activations, runtime, allocator overhead and process RSS.</summary>
    public long EstimatedPeakWeightBytes => Assembly.EstimatedPeakWeightBytes;

    internal Sd15CheckpointPlan(SdCheckpointAssemblyPlan assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        if (assembly.Profile != SdCheckpointAssemblyProfile.Sd15)
            throw new ArgumentException("The public SD1.5 plan requires the fixed stock configurations.", nameof(assembly));
        Assembly = assembly;
    }
}

// Reduced profiles exist solely behind the same internal orchestration used by the stock facade.
// Their successful execution does not demonstrate a stock checkpoint load or a qualified model family.
internal sealed record SdCheckpointAssemblyProfile(ClipTextConfig Clip, SdUnetConfig Unet, ClassicalVaeConfig Vae)
{
    internal static SdCheckpointAssemblyProfile Sd15 { get; } =
        new(ClipTextConfig.Large, SdUnetConfig.Sd15, ClassicalVaeConfig.Stock);

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Clip);
        ArgumentNullException.ThrowIfNull(Unet);
        ArgumentNullException.ThrowIfNull(Vae);
        Clip.Validate();
        Unet.Validate();
        Vae.Validate();
        if (Clip.HiddenSize != Unet.ContextSize)
            throw new ArgumentException("CLIP hidden width must match the U-Net context width.");
    }
}

internal sealed class SdCheckpointAssemblyPlan
{
    internal SafeTensorFile Source { get; }
    internal SdCheckpointAssemblyProfile Profile { get; }
    internal ClipWeightPlan Clip { get; }
    internal UnetWeightPlan Unet { get; }
    internal ClassicalVaeWeightPlan Vae { get; }
    internal Sd15UnclaimedTensorHandling UnclaimedTensorHandling { get; }
    internal IReadOnlyList<string> UnclaimedTensorNames { get; }
    internal long SourceBytes { get; }
    internal long ResidentBytes { get; }
    internal long TemporaryBytes { get; }
    internal long EstimatedPeakWeightBytes { get; }

    internal SdCheckpointAssemblyPlan(SafeTensorFile source, SdCheckpointAssemblyProfile profile,
        ClipWeightPlan clip, UnetWeightPlan unet, ClassicalVaeWeightPlan vae,
        Sd15UnclaimedTensorHandling handling, IReadOnlyList<string> unclaimed, long sourceBytes)
    {
        Source = source;
        Profile = profile;
        Clip = clip;
        Unet = unet;
        Vae = vae;
        UnclaimedTensorHandling = handling;
        UnclaimedTensorNames = Array.AsReadOnly(unclaimed.ToArray());
        SourceBytes = sourceBytes;
        ResidentBytes = checked(clip.ResidentBytes + unet.ResidentBytes + vae.ResidentBytes);
        TemporaryBytes = Math.Max(clip.TemporaryBytes, Math.Max(unet.TemporaryBytes, vae.TemporaryBytes));
        EstimatedPeakWeightBytes = checked(ResidentBytes + TemporaryBytes);
    }
}
