namespace ComfySharp.Tokenization;

public enum ClipProfile { Sd1L, SdXlL, SdXlG }

/// <summary>A text token. Structural tokens always have weight one and word ID zero.</summary>
public readonly record struct ClipToken(int Id, double Weight, int WordId);
public readonly record struct ClipTokenWeight(int Id, double Weight);
public readonly record struct WeightedPromptSegment(string Text, double Weight);

/// <summary>Construction options corresponding to the frozen SDTokenizer constructor.</summary>
public sealed record ClipPackingOptions
{
    public int MaxLength { get; init; } = 77;
    public int? MinLength { get; init; }
    public int? MinPadding { get; init; }
    public int? PadToken { get; init; }
    public bool PadToMaxLength { get; init; } = true;
    public bool PadLeft { get; init; }
    public bool DisableWeights { get; init; }
    public bool ResolveTextualInversions { get; init; }
    public IReadOnlyDictionary<string, int?>? TokenizerData { get; init; }
}

/// <summary>Per-call overrides. OverrideMinLength also allows explicitly overriding a configured minimum with null.</summary>
public sealed record ClipTokenizeOptions
{
    public bool ReturnWordIds { get; init; }
    public bool? DisableWeights { get; init; }
    public int? MinLength { get; init; }
    public bool OverrideMinLength { get; init; }
    public IReadOnlyDictionary<string, int?>? TokenizerOptions { get; init; }
}

/// <summary>Immutable chunks retaining all word IDs, including when pair-only output is requested.</summary>
public sealed class ClipTokenization
{
    internal ClipTokenization(ClipProfile profile, bool returnWordIds, List<List<ClipToken>> chunks)
    {
        Profile = profile;
        ReturnWordIds = returnWordIds;
        Chunks = Array.AsReadOnly(chunks.Select(c => (IReadOnlyList<ClipToken>)Array.AsReadOnly(c.ToArray())).ToArray());
    }

    public ClipProfile Profile { get; }
    public bool ReturnWordIds { get; }
    public IReadOnlyList<IReadOnlyList<ClipToken>> Chunks { get; }

    /// <summary>Projects the upstream return_word_ids=false shape without losing the retained IDs.</summary>
    public IReadOnlyList<IReadOnlyList<ClipTokenWeight>> WithoutWordIds() =>
        Array.AsReadOnly(Chunks.Select(c => (IReadOnlyList<ClipTokenWeight>)Array.AsReadOnly(
            c.Select(t => new ClipTokenWeight(t.Id, t.Weight)).ToArray())).ToArray());
}
