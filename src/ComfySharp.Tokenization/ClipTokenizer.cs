using System.Collections.Frozen;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace ComfySharp.Tokenization;

/// <summary>Managed CLIP byte BPE for the Transformers 5.14.1 / tokenizers 0.22.2 profile.</summary>
public sealed class ClipTokenizer : IClipTokenizer
{
    private const int MaximumCachedWordLength = 256;
    private static readonly Lazy<Tables> DefaultTables = new(() => LoadTables(ReadEmbedded));
    private static readonly string[] Contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    private readonly Tables tables;
    private readonly int cacheCapacity;
    private readonly object cacheLock = new();
    private readonly Dictionary<string, int[]> cache = new(StringComparer.Ordinal);
    private readonly Queue<string> cacheOrder = new();

    public int BosTokenId => 49406;
    public int EosTokenId => 49407;
    /// <summary>Current number of cached words; never exceeds the configured capacity.</summary>
    public int CachedTokenCount { get { lock (cacheLock) return cache.Count; } }

    public ClipTokenizer(int cacheCapacity = 4096) : this(DefaultTables.Value, cacheCapacity) { }
    private ClipTokenizer(Tables tables, int cacheCapacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(cacheCapacity);
        this.tables = tables; this.cacheCapacity = cacheCapacity;
    }
    public static ClipTokenizer CreateDefault() => new();

    /// <summary>Loads canonical profile assets supplied as streams. Streams remain caller-owned.
    /// Resource names are Clip/vocab.json, Clip/merges.txt, Clip/tokenizer_config.json,
    /// Clip/special_tokens_map.json, and ClipUnicode/unicode-profile.json.</summary>
    public static ClipTokenizer FromResources(Func<string, Stream> openResource, int cacheCapacity = 4096)
    {
        ArgumentNullException.ThrowIfNull(openResource);
        return new ClipTokenizer(LoadTables(name =>
        {
            var stream = openResource(name) ?? throw new InvalidDataException($"Missing CLIP resource '{name}'.");
            using var buffer = new MemoryStream(); stream.CopyTo(buffer); return buffer.ToArray();
        }), cacheCapacity);
    }

    public IReadOnlyList<int> Encode(string text, bool addSpecialTokens = true, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        cancellationToken.ThrowIfCancellationRequested();
        var normalized = tables.Unicode.Normalize(text, cancellationToken);
        var result = new List<int>();
        if (addSpecialTokens) result.Add(BosTokenId);
        for (var offset = 0; offset < normalized.Length;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (normalized[offset] == ' ') { offset++; continue; }
            var remaining = normalized.AsSpan(offset);
            if (remaining.StartsWith("<|startoftext|>", StringComparison.Ordinal))
            { result.Add(BosTokenId); offset += 15; continue; }
            if (remaining.StartsWith("<|endoftext|>", StringComparison.Ordinal))
            { result.Add(EosTokenId); offset += 13; continue; }
            var start = offset;
            var contraction = Contractions.FirstOrDefault(c => normalized.AsSpan(start).StartsWith(c, StringComparison.Ordinal));
            if (contraction is not null) offset += contraction.Length;
            else
            {
                var first = Rune.GetRuneAt(normalized, offset);
                var letter = tables.Unicode.IsLetter(first.Value);
                var number = tables.Unicode.IsNumber(first.Value);
                offset += first.Utf16SequenceLength;
                if (!number)
                    while (offset < normalized.Length)
                    {
                        if ((offset & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
                        if (normalized.AsSpan(offset).StartsWith("<|startoftext|>", StringComparison.Ordinal) || normalized.AsSpan(offset).StartsWith("<|endoftext|>", StringComparison.Ordinal)) break;
                        var next = Rune.GetRuneAt(normalized, offset);
                        if (tables.Unicode.IsWhitespace(next.Value) || tables.Unicode.IsNumber(next.Value) || tables.Unicode.IsLetter(next.Value) != letter) break;
                        offset += next.Utf16SequenceLength;
                    }
            }
            result.AddRange(EncodeWord(normalized[start..offset], cancellationToken));
        }
        if (addSpecialTokens) result.Add(EosTokenId);
        cancellationToken.ThrowIfCancellationRequested();
        return result.AsReadOnly();
    }

    public string GetToken(int id)
    {
        if ((uint)id >= (uint)tables.Tokens.Length) throw new ArgumentOutOfRangeException(nameof(id), id, "CLIP token ID is outside the vocabulary.");
        return tables.Tokens[id];
    }

    public string Decode(IEnumerable<int> ids, bool skipSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(ids);
        using var bytes = new MemoryStream();
        foreach (var id in ids)
        {
            var token = GetToken(id);
            if (skipSpecialTokens && (id == BosTokenId || id == EosTokenId)) continue;
            foreach (var character in token) bytes.WriteByte(tables.ByteDecoder[character]);
        }
        var decoded = Encoding.UTF8.GetString(bytes.GetBuffer(), 0, checked((int)bytes.Length)).Replace("</w>", " ", StringComparison.Ordinal);
        // Python str.strip (the CLIP decode wrapper) additionally recognizes
        // the four C0 separators that the tokenizer's Unicode regex does not.
        var start = 0; var end = decoded.Length;
        while (start < end && IsPythonWhitespace(decoded[start])) start++;
        while (end > start && IsPythonWhitespace(decoded[end - 1])) end--;
        return decoded[start..end];
    }
    private static bool IsPythonWhitespace(char c) => c is >= '\u0009' and <= '\u000d' or >= '\u001c' and <= '\u0020' or '\u0085' or '\u00a0' or '\u1680' or >= '\u2000' and <= '\u200a' or '\u2028' or '\u2029' or '\u202f' or '\u205f' or '\u3000';

    private int[] EncodeWord(string word, CancellationToken cancellationToken)
    {
        var cacheable = cacheCapacity > 0 && word.Length <= MaximumCachedWordLength;
        if (cacheable) lock (cacheLock) if (cache.TryGetValue(word, out var cached)) return cached;
        var bytes = Encoding.UTF8.GetBytes(word);
        var nodes = new Node[bytes.Length];
        for (var i = 0; i < nodes.Length; i++)
        {
            if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            var symbol = tables.ByteEncoder[bytes[i]].ToString() + (i == bytes.Length - 1 ? "</w>" : "");
            nodes[i] = new Node(tables.Vocabulary[symbol], i - 1, i + 1 < nodes.Length ? i + 1 : -1);
        }
        var candidates = new PriorityQueue<(int Left, int Right, int LeftId, int RightId, int MergedId), (int Rank, int Position)>();
        for (var i = 0; i < nodes.Length - 1; i++) { if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested(); AddPair(i); }
        var steps = 0;
        while (candidates.TryDequeue(out var pair, out _))
        {
            if ((steps++ & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            ref var left = ref nodes[pair.Left]; ref var right = ref nodes[pair.Right];
            if (left.Removed || right.Removed || left.Next != pair.Right || left.Id != pair.LeftId || right.Id != pair.RightId) continue;
            left.Id = pair.MergedId; left.Next = right.Next; right.Removed = true;
            if (right.Next >= 0) nodes[right.Next].Previous = pair.Left;
            if (left.Previous >= 0) AddPair(left.Previous);
            AddPair(pair.Left);
        }
        var output = new List<int>();
        for (var i = 0; i >= 0 && i < nodes.Length; i = nodes[i].Next)
        { if ((output.Count & 1023) == 0) cancellationToken.ThrowIfCancellationRequested(); output.Add(nodes[i].Id); }
        var value = output.ToArray();
        if (cacheable) lock (cacheLock)
        {
            if (!cache.ContainsKey(word))
            {
                if (cache.Count == cacheCapacity) cache.Remove(cacheOrder.Dequeue());
                cache.Add(word, value); cacheOrder.Enqueue(word);
            }
        }
        return value;
        void AddPair(int index)
        {
            var right = nodes[index].Next;
            if (right >= 0 && tables.Merges.TryGetValue((nodes[index].Id, nodes[right].Id), out var merge))
                candidates.Enqueue((index, right, nodes[index].Id, nodes[right].Id, merge.Id), (merge.Rank, index));
        }
    }

    private struct Node(int id, int previous, int next)
    { internal int Id = id, Previous = previous, Next = next; internal bool Removed; }

    private sealed record Tables(string[] Tokens, FrozenDictionary<string, int> Vocabulary,
        FrozenDictionary<(int, int), (int Rank, int Id)> Merges, char[] ByteEncoder, FrozenDictionary<char, byte> ByteDecoder, ClipUnicode Unicode);

    private static Tables LoadTables(Func<string, byte[]> read)
    {
        var assets = new Dictionary<string, string>
        {
            ["Clip/vocab.json"] = "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
            ["Clip/merges.txt"] = "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
            ["Clip/tokenizer_config.json"] = "f37d2056ee9743c44a79b85eb3763b7bab378339e69a4b50f58a37e89066821c",
            ["Clip/special_tokens_map.json"] = "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
            ["ClipUnicode/unicode-profile.json"] = ClipUnicode.ResourceHash
        };
        var loaded = new Dictionary<string, byte[]>();
        foreach (var (name, hash) in assets)
        {
            byte[] bytes;
            try { bytes = read(name); }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            { throw new InvalidDataException($"Cannot load required CLIP resource '{name}'.", exception); }
            if (!Convert.ToHexString(SHA256.HashData(bytes)).Equals(hash, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"CLIP resource '{name}' failed SHA-256 integrity verification; expected {hash}.");
            loaded[name] = bytes;
        }
        var vocabulary = JsonSerializer.Deserialize<Dictionary<string, int>>(loaded["Clip/vocab.json"])!;
        var tokens = new string[vocabulary.Count];
        foreach (var (token, id) in vocabulary) tokens[id] = token;
        var merges = new Dictionary<(int, int), (int, int)>();
        var lines = Encoding.UTF8.GetString(loaded["Clip/merges.txt"]).Split('\n');
        for (var i = 1; i < lines.Length && merges.Count < 48894; i++)
        {
            var pair = lines[i].Split(' ');
            if (pair.Length != 2) throw new InvalidDataException("Invalid CLIP merge entry.");
            merges.Add((vocabulary[pair[0]], vocabulary[pair[1]]), (i - 1, vocabulary[pair[0] + pair[1]]));
        }
        if (vocabulary.Count != 49408 || merges.Count != 48894 || tokens[49406] != "<|startoftext|>" || tokens[49407] != "<|endoftext|>")
            throw new InvalidDataException("CLIP resource vocabulary or special token IDs are inconsistent.");
        var encoder = new char[256]; var decoder = new Dictionary<char, byte>(); var next = 256;
        for (var b = 0; b < 256; b++)
        {
            var character = (char)(b is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255 ? b : next++);
            encoder[b] = character; decoder.Add(character, (byte)b);
        }
        return new Tables(tokens, vocabulary.ToFrozenDictionary(StringComparer.Ordinal), merges.ToFrozenDictionary(), encoder, decoder.ToFrozenDictionary(), new ClipUnicode(loaded["ClipUnicode/unicode-profile.json"]));
    }

    private static byte[] ReadEmbedded(string name)
    {
        var resourceName = "ComfySharp.Tokenization.Resources." + name.Replace('/', '.');
        using var stream = typeof(ClipTokenizer).Assembly.GetManifestResourceStream(resourceName)
            ?? throw new InvalidDataException($"Missing embedded CLIP resource '{name}'.");
        using var buffer = new MemoryStream(); stream.CopyTo(buffer); return buffer.ToArray();
    }
}
