using ComfySharp.Tokenization;
using TorchSharp;

namespace ComfySharp.Inference;

public enum ClipHiddenSelection { ProfileDefault, Last, Hidden, All }

/// <summary>Integer-only special tokens consumed by process_tokens; null start/end omit that marker.</summary>
public sealed record ClipSpecialTokens(int? Start, int? End, int Pad)
{
    public static ClipSpecialTokens ForProfile(ClipProfile profile) => profile switch
    {
        ClipProfile.Sd1L or ClipProfile.SdXlL => new(49406, 49407, 49407),
        ClipProfile.SdXlG => new(49406, 49407, 0),
        _ => throw new ArgumentOutOfRangeException(nameof(profile))
    };
}

/// <summary>Per-call source wrapper options. Defaults follow the explicitly selected profile.</summary>
public sealed record ClipConditioningOptions
{
    public ClipHiddenSelection HiddenSelection { get; init; }
    public int? IntermediateLayer { get; init; }
    public bool? NormalizeIntermediate { get; init; }
    public bool? ProjectPooled { get; init; }
    public bool EnableAttentionMasks { get; init; }
    public bool ReturnAttentionMasks { get; init; }
    public bool ZeroOutMasked { get; init; }
    public ClipSpecialTokens? SpecialTokens { get; init; }
}

public sealed record ClipSdxlConditioningOptions
{
    public ClipConditioningOptions? L { get; init; }
    public ClipConditioningOptions? G { get; init; }
}

/// <summary>
/// Caller-owned CPU outputs independent of encoder and ambient tensor scopes. Hidden/pooled are F32;
/// optional attention masks are Int64. Ordinary hidden shape is [1,sections*77,H]. All-layer shape is
/// [1,layers,sections*77,H], with the source's layer-indexed weighting and broadcasting semantics.
/// </summary>
public sealed class ClipConditioningResult(torch.Tensor hidden, torch.Tensor pooled, torch.Tensor? attentionMask = null) : IDisposable
{
    public torch.Tensor Hidden { get; } = hidden;
    public torch.Tensor Pooled { get; } = pooled;
    public torch.Tensor? AttentionMask { get; } = attentionMask;

    public void Dispose()
    {
        var tensors = new HashSet<torch.Tensor>(ReferenceEqualityComparer.Instance) { Hidden, Pooled };
        if (AttentionMask is not null) tensors.Add(AttentionMask);
        foreach (var tensor in tensors) tensor.Dispose();
    }
}
