using ComfySharp.Tokenization;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Frozen SDClipModel and ClipTokenWeightEncoder integer-token, Float32 inference on CPU/CUDA.</summary>
public sealed class ComfyClipEncoder : IDisposable
{
    private readonly object gate = new();
    private ClipTextEncoder? encoder;

    public ComfyClipEncoder(ClipTextEncoder encoder, ClipProfile profile)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        if (!Enum.IsDefined(profile)) throw new ArgumentOutOfRangeException(nameof(profile));
        Profile = profile;
        this.encoder = encoder.Retain();
    }

    public ClipProfile Profile { get; }
    public ClipTextConfig Config { get { using var graph = RetainGraph(); return graph.Config; } }
    public Device Device { get { using var graph = RetainGraph(); return graph.Device; } }
    public ComfyClipEncoder To(Device device, CancellationToken cancellationToken = default)
    {
        using var graph = RetainGraph(); using var moved = graph.To(device, cancellationToken); return new ComfyClipEncoder(moved, Profile);
    }

    public ComfyClipEncoder WithLora(IReadOnlyDictionary<string,LoraWeightPatch> patches,
        long maxPatchedWeightBytes=512L*1024*1024,CancellationToken cancellationToken=default)
    {
        using var graph=RetainGraph();using var patched=graph.WithLora(patches,maxPatchedWeightBytes,cancellationToken);return new(patched,Profile);
    }

    public ComfyClipEncoder Retain()
    {
        lock (gate) return new(encoder ?? throw new ObjectDisposedException(nameof(ComfyClipEncoder)), Profile);
    }

    public ClipConditioningResult Encode(ClipTokenization tokens, ClipConditioningOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Profile != Profile) throw new ArgumentException("Tokenization profile must match the encoder profile.", nameof(tokens));
        return Encode(tokens.WithoutWordIds(), options, cancellationToken);
    }

    public ClipConditioningResult Encode(IReadOnlyList<IReadOnlyList<ClipTokenWeight>> sections,
        ClipConditioningOptions? options = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var operation = RetainGraph();
        options ??= new();
        var special = options.SpecialTokens ?? ClipSpecialTokens.ForProfile(Profile);
        ValidateSpecialTokens(special);
        var rows = SnapshotSections(sections, out var weights, out bool hasWeights, cancellationToken);
        int sectionCount = rows.Count;
        if (hasWeights) rows.Add(EmptyTokens(special));
        var processed = ProcessTokens(rows, special);
        var forwardOptions = ResolveOptions(options, operation.Config.LayerCount, processed, hasWeights);
        if (forwardOptions.ProjectPooled && !operation.HasProjection)
            throw new InvalidOperationException("Projected CLIP pooled output requires text_projection.weight.");

        // Native bootstrap must precede even TorchSharp's disposal/no-grad scopes.
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var inference = no_grad();
        using var encoded = operation.Forward(rows, forwardOptions, cancellationToken);
        var selected = forwardOptions.AllIntermediateLayers || forwardOptions.IntermediateLayer is not null
            ? encoded.IntermediateHidden! : encoded.FinalHidden;
        Tensor? mask = null;
        if (options.ZeroOutMasked || options.ReturnAttentionMasks)
            mask = tensor(processed.Masks.SelectMany(x => x).Select(x => (long)x).ToArray(), dtype: ScalarType.Int64, device: selected.device)
                .reshape(rows.Count, ClipTextConfig.MaxPositions);
        if (options.ZeroOutMasked) selected = selected * mask!.unsqueeze(-1).to_type(ScalarType.Float32);

        var outputs = new List<Tensor>(sectionCount);
        for (int k = 0; k < sectionCount; k++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var z = selected.slice(0, k, k + 1, 1);
            if (hasWeights)
            {
                using var rowScope = NewDisposeScope();
                var empty = selected[rows.Count - 1];
                var row = z[0];
                // Literal upstream indexing: for rank four, j is a layer, not a token position.
                for (int j = 0; j < row.shape[0]; j++)
                {
                    if (weights[k][j] == 1.0) continue;
                    using var weightScope = NewDisposeScope();
                    var target = row[j];
                    var baseline = empty[j];
                    target.copy_((target - baseline) * weights[k][j] + baseline);
                }
            }
            outputs.Add(z);
        }

        var hidden = cat(outputs.ToArray(), dim: -2);
        var pooled = (forwardOptions.ProjectPooled ? encoded.ProjectedPooled! : encoded.Pooled).slice(0, 0, 1, 1);
        var returnedMask = options.ReturnAttentionMasks
            ? mask!.slice(0, 0, sectionCount, 1).flatten().unsqueeze(0) : null;
        cancellationToken.ThrowIfCancellationRequested();
        return new(hidden.DetachFromDisposeScope(), pooled.DetachFromDisposeScope(), returnedMask?.DetachFromDisposeScope());
    }

    private ClipForwardOptions ResolveOptions(ClipConditioningOptions options, int layers, ProcessedClipTokens processed, bool hasWeights)
    {
        if (!Enum.IsDefined(options.HiddenSelection)) throw new ArgumentOutOfRangeException(nameof(options.HiddenSelection));
        bool all = options.HiddenSelection == ClipHiddenSelection.All;
        int? layer = options.HiddenSelection switch
        {
            ClipHiddenSelection.ProfileDefault => options.IntermediateLayer ?? (Profile == ClipProfile.Sd1L ? null : -2),
            ClipHiddenSelection.Hidden => options.IntermediateLayer ?? throw new ArgumentException("Hidden selection requires IntermediateLayer."),
            _ => null
        };
        if (options.HiddenSelection is ClipHiddenSelection.Last or ClipHiddenSelection.All && options.IntermediateLayer is not null)
            throw new ArgumentException("Last/all selection cannot also select an intermediate layer.");
        if (layer is not null && (layer < -layers || layer >= layers))
            throw new ArgumentOutOfRangeException(nameof(options.IntermediateLayer), "Layer index must resolve to an existing transformer block.");
        int outputLayers = layers;
        if (all && options.ZeroOutMasked)
        {
            int batch = processed.Masks.Count;
            // Source uses in-place multiplication, so broadcasting may not expand the layer axis.
            if (batch != 1 && batch != layers)
                throw new ArgumentException("Source all-layer zero_out_masked broadcasting requires batch one or batch equal to layer count (including the weighted empty row).");
        }
        if (all && hasWeights && outputLayers > ClipTextConfig.MaxPositions)
            throw new ArgumentException("Source all-layer weighting indexes token weights by layer; this output has more layers than token weights.");
        return new()
        {
            IntermediateLayer = layer,
            AllIntermediateLayers = all,
            NormalizeIntermediate = options.NormalizeIntermediate ?? Profile == ClipProfile.Sd1L,
            ProjectPooled = options.ProjectPooled ?? Profile != ClipProfile.Sd1L,
            AttentionMask = options.EnableAttentionMasks ? processed.Masks : null,
            TokenCounts = processed.Counts
        };
    }

    private static List<IReadOnlyList<int>> SnapshotSections(IReadOnlyList<IReadOnlyList<ClipTokenWeight>> sections,
        out double[][] weights, out bool hasWeights, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(sections);
        if (sections.Count == 0) throw new ArgumentException("CLIP encoding requires at least one 77-token section.", nameof(sections));
        var rows = new List<IReadOnlyList<int>>(sections.Count + 1);
        weights = new double[sections.Count][];
        hasWeights = false;
        for (int k = 0; k < sections.Count; k++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var section = sections[k];
            if (section is null || section.Count != ClipTextConfig.MaxPositions)
                throw new ArgumentException("Every CLIP encoder section must contain exactly 77 tokens; ragged or oversized tokenizer output is unsupported.", nameof(sections));
            var row = new int[ClipTextConfig.MaxPositions];
            weights[k] = new double[row.Length];
            for (int j = 0; j < row.Length; j++)
            {
                var token = section[j];
                if ((uint)token.Id >= ClipTextConfig.VocabularySize) throw new ArgumentException("CLIP token ID is outside the fixed vocabulary.", nameof(sections));
                row[j] = token.Id;
                weights[k][j] = token.Weight;
                hasWeights |= token.Weight != 1.0;
            }
            rows.Add(row);
        }
        return rows;
    }

    private static void ValidateSpecialTokens(ClipSpecialTokens special)
    {
        if ((special.Start is int start && (uint)start >= ClipTextConfig.VocabularySize)
            || (special.End is int end && (uint)end >= ClipTextConfig.VocabularySize)
            || (uint)special.Pad >= ClipTextConfig.VocabularySize)
            throw new ArgumentException("Special token IDs must belong to the fixed CLIP vocabulary.", nameof(special));
    }

    private static int[] EmptyTokens(ClipSpecialTokens special)
    {
        var row = Enumerable.Repeat(special.Pad, ClipTextConfig.MaxPositions).ToArray();
        int i = 0;
        if (special.Start is int start) row[i++] = start;
        if (special.End is int end) row[i] = end;
        return row;
    }

    internal static ProcessedClipTokens ProcessTokens(IReadOnlyList<IReadOnlyList<int>> rows, ClipSpecialTokens special)
    {
        var masks = new List<IReadOnlyList<int>>(rows.Count);
        var counts = new int[rows.Count];
        for (int k = 0; k < rows.Count; k++)
        {
            var row = rows[k];
            var mask = new int[row.Count];
            bool eos = false, leftPad = false;
            for (int i = 0; i < row.Count; i++)
            {
                int token = row[i];
                if (i == 0 && token == special.Pad) leftPad = true;
                if (!eos && !(leftPad && token == special.Pad)) { mask[i] = 1; leftPad = false; }
                if (!eos && token == (special.End ?? special.Pad) && !leftPad)
                {
                    if (special.End is null) mask[i] = 0;
                    eos = true;
                }
                counts[k] += mask[i];
            }
            masks.Add(mask);
        }
        return new(masks, counts);
    }

    private ClipTextEncoder RetainGraph()
    {
        lock (gate) return (encoder ?? throw new ObjectDisposedException(nameof(ComfyClipEncoder))).Retain();
    }

    public void Dispose()
    {
        lock (gate) { encoder?.Dispose(); encoder = null; }
    }
}

internal sealed record ProcessedClipTokens(IReadOnlyList<IReadOnlyList<int>> Masks, IReadOnlyList<int> Counts);
