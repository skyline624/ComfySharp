using System.Collections.ObjectModel;

namespace ComfySharp.Tokenization;

/// <summary>Text-only SD1/SDXL CLIP weighted tokenization. No embeddings, tensors or encoder outputs are synthesized.</summary>
public sealed class ComfyClipTokenizer
{
    private readonly IClipTokenizer tokenizer;
    private readonly ClipPackingOptions settings;
    private readonly int maxLength;
    private readonly int? minLength;

    public ComfyClipTokenizer(IClipTokenizer tokenizer, ClipProfile profile = ClipProfile.Sd1L, ClipPackingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (!Enum.IsDefined(profile)) throw new ArgumentOutOfRangeException(nameof(profile));
        this.tokenizer = tokenizer;
        Profile = profile;
        settings = options ?? new ClipPackingOptions();
        if (settings.ResolveTextualInversions)
            throw new NotSupportedException("Textual inversion resolution is not available in text-only CLIP tokenization.");
        // Resolve the caller-owned dictionary now; later mutations cannot alter this tokenizer.
        maxLength = GetSetting(settings.TokenizerData, EmbeddingKey + "_max_length", settings.MaxLength)
            ?? throw new ArgumentException("CLIP max_length cannot be null.", nameof(options));
        minLength = GetSetting(settings.TokenizerData, EmbeddingKey + "_min_length", settings.MinLength);
        if (maxLength <= 2)
            throw new ArgumentOutOfRangeException(nameof(options), "CLIP max_length must reserve BOS, EOS and at least one content token.");
    }

    public ClipProfile Profile { get; }
    public string OutputKey => Profile == ClipProfile.SdXlG ? "g" : "l";
    public string EmbeddingKey => Profile == ClipProfile.SdXlG ? "clip_g" : "clip_l";
    public int EmbeddingSize => Profile == ClipProfile.SdXlG ? 1280 : 768;
    public int PadTokenId => settings.PadToken ?? (Profile == ClipProfile.SdXlG ? 0 : tokenizer.EosTokenId);

    public ClipTokenization Tokenize(string text, ClipTokenizeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        cancellationToken.ThrowIfCancellationRequested();
        options ??= new ClipTokenizeOptions();
        var callMinLength = GetSetting(options.TokenizerOptions, EmbeddingKey + "_min_length", minLength);
        var callMinPadding = GetSetting(options.TokenizerOptions, EmbeddingKey + "_min_padding", settings.MinPadding);
        if (options.OverrideMinLength || options.MinLength.HasValue) callMinLength = options.MinLength;
        var chunks = new List<List<ClipToken>>();
        var batch = NewBatch(chunks);
        var wordId = 0;
        foreach (var segment in PromptWeights.Parse(text, options.DisableWeights ?? settings.DisableWeights, cancellationToken))
        {
            foreach (var unit in SplitEmbeddingUnits(segment.Text, cancellationToken))
            {
                cancellationToken.ThrowIfCancellationRequested();
                wordId = checked(wordId + 1);
                // Request the dependency's complete input, then remove only its external BOS/EOS.
                var ids = tokenizer.Encode(unit, addSpecialTokens: true, cancellationToken);
                if (ids.Count < 2 || ids[0] != tokenizer.BosTokenId || ids[^1] != tokenizer.EosTokenId)
                    throw new InvalidOperationException("IClipTokenizer must return external CLIP BOS and EOS tokens.");
                var count = ids.Count - 2;
                var isLarge = count >= 8;
                if (!isLarge && count > maxLength - 2)
                    throw new ArgumentException("This CLIP capacity cannot fit an indivisible segment of fewer than eight tokens.", nameof(options));
                var offset = 1;
                while (count > 0)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if ((long)count + batch.Count > maxLength - 1)
                    {
                        var remaining = maxLength - batch.Count - 1;
                        if (isLarge)
                        {
                            AddTokens(batch, ids, offset, remaining, segment.Weight, wordId, cancellationToken);
                            offset += remaining;
                            count -= remaining;
                            batch.Add(EndToken());
                        }
                        else
                        {
                            batch.Add(EndToken());
                            if (settings.PadToMaxLength) Pad(batch, remaining, cancellationToken);
                        }
                        batch = NewBatch(chunks);
                    }
                    else
                    {
                        AddTokens(batch, ids, offset, count, segment.Weight, wordId, cancellationToken);
                        count = 0;
                    }
                }
            }
        }
        batch.Add(EndToken());
        if (callMinPadding.HasValue) Pad(batch, callMinPadding.Value, cancellationToken);
        if (settings.PadToMaxLength && batch.Count < maxLength) Pad(batch, maxLength - batch.Count, cancellationToken);
        if (callMinLength.HasValue && batch.Count < callMinLength.Value) Pad(batch, callMinLength.Value - batch.Count, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        return new ClipTokenization(Profile, options.ReturnWordIds, chunks);
    }

    public string Decode(IEnumerable<int> tokenIds, bool skipSpecialTokens = true) => tokenizer.Decode(tokenIds, skipSpecialTokens);
    public IReadOnlyList<(ClipToken Token, string Text)> Untokenize(IEnumerable<ClipToken> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        return Array.AsReadOnly(tokens.Select(t => (t, tokenizer.GetToken(t.Id))).ToArray());
    }

    private List<ClipToken> NewBatch(List<List<ClipToken>> chunks)
    {
        var batch = new List<ClipToken> { new(tokenizer.BosTokenId, 1, 0) };
        chunks.Add(batch);
        return batch;
    }
    private ClipToken EndToken() => new(tokenizer.EosTokenId, 1, 0);
    private static int? GetSetting(IReadOnlyDictionary<string, int?>? values, string key, int? fallback) =>
        values is not null && values.TryGetValue(key, out var value) ? value : fallback;

    private static void AddTokens(List<ClipToken> batch, IReadOnlyList<int> ids, int offset, int count, double weight, int wordId, CancellationToken cancellationToken)
    {
        for (var i = 0; i < count; i++)
        {
            if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            batch.Add(new ClipToken(ids[offset + i], weight, wordId));
        }
    }

    private void Pad(List<ClipToken> batch, int amount, CancellationToken cancellationToken)
    {
        if (amount <= 0) return; // Python range(negative) / list repetition produces no padding.
        _ = checked(batch.Count + amount);
        var pads = new ClipToken[amount];
        for (var i = 0; i < pads.Length; i++)
        {
            if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            pads[i] = new ClipToken(PadTokenId, 1, 0);
        }
        if (settings.PadLeft) batch.InsertRange(0, pads);
        else batch.AddRange(pads);
    }

    private static IEnumerable<string> SplitEmbeddingUnits(string text, CancellationToken cancellationToken)
    {
        const string marker = "embedding:";
        var start = 0;
        for (var i = 1; i <= text.Length - marker.Length; i++)
        {
            if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            if (PromptWeights.IsPythonWhitespace(text[i - 1]) && text.AsSpan(i).StartsWith(marker, StringComparison.Ordinal))
            {
                if (i > start) yield return text[start..i];
                start = i;
                i += marker.Length - 1;
            }
        }
        if (start < text.Length) yield return text[start..];
    }
}

/// <summary>The frozen SDXL composition, emitting G followed by L with independently packed lengths.</summary>
public sealed class ComfySdxlTokenizer
{
    public ComfySdxlTokenizer(IClipTokenizer tokenizer, ClipPackingOptions? lOptions = null, ClipPackingOptions? gOptions = null)
    {
        L = new ComfyClipTokenizer(tokenizer, ClipProfile.SdXlL, lOptions);
        G = new ComfyClipTokenizer(tokenizer, ClipProfile.SdXlG, gOptions);
    }
    public ComfyClipTokenizer L { get; }
    public ComfyClipTokenizer G { get; }
    public IReadOnlyDictionary<string, ClipTokenization> Tokenize(string text, ClipTokenizeOptions? options = null, CancellationToken cancellationToken = default) =>
        new ReadOnlyDictionary<string, ClipTokenization>(new Dictionary<string, ClipTokenization>
        {
            ["g"] = G.Tokenize(text, options, cancellationToken),
            ["l"] = L.Tokenize(text, options, cancellationToken),
        });
}
