using TorchSharp;

namespace ComfySharp.Inference;

public enum ClipActivation { QuickGelu, Gelu, GeluTanh }

/// <summary>Dimensions actually consumed by the frozen CLIP text graph.</summary>
public sealed record ClipTextConfig(int HiddenSize, int IntermediateSize, int LayerCount, int HeadCount, ClipActivation Activation)
{
    public const int VocabularySize = 49408;
    public const int MaxPositions = 77;
    public static ClipTextConfig Large { get; } = new(768, 3072, 12, 12, ClipActivation.QuickGelu);
    public static ClipTextConfig Giant { get; } = new(1280, 5120, 32, 20, ClipActivation.Gelu);

    public void Validate()
    {
        if (HiddenSize <= 0 || IntermediateSize <= 0 || LayerCount <= 0 || HeadCount <= 0 || HiddenSize % HeadCount != 0 || !Enum.IsDefined(Activation))
            throw new ArgumentException("CLIP dimensions must be positive, with hidden width divisible by the head count and a supported activation.");
    }
}

public sealed record ClipForwardOptions
{
    public int? IntermediateLayer { get; init; }
    public bool AllIntermediateLayers { get; init; }
    public bool NormalizeIntermediate { get; init; } = true;
    public bool ProjectPooled { get; init; } = true;
    public IReadOnlyList<IReadOnlyList<int>>? AttentionMask { get; init; }
    public IReadOnlyList<int>? TokenCounts { get; init; }
}

/// <summary>Caller-owned CPU/F32 outputs; disposing a model never disposes these results.</summary>
public sealed class ClipForwardResult(torch.Tensor finalHidden, torch.Tensor? intermediateHidden, torch.Tensor? projectedPooled, torch.Tensor pooled) : IDisposable
{
    public torch.Tensor FinalHidden { get; } = finalHidden;
    public torch.Tensor? IntermediateHidden { get; } = intermediateHidden;
    public torch.Tensor? ProjectedPooled { get; } = projectedPooled;
    public torch.Tensor Pooled { get; } = pooled;

    public void Dispose()
    {
        // Distinct wrappers may share native storage; only deduplicate wrapper identity.
        var tensors = new HashSet<torch.Tensor>(ReferenceEqualityComparer.Instance) { FinalHidden, Pooled };
        if (IntermediateHidden is not null) tensors.Add(IntermediateHidden);
        if (ProjectedPooled is not null) tensors.Add(ProjectedPooled);
        foreach (var tensor in tensors) tensor.Dispose();
    }
}
