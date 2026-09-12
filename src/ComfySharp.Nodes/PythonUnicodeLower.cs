using System.Buffers;
using System.Collections;
using System.Collections.Frozen;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace ComfySharp.Nodes;

/// <summary>
/// CPython 3.12.10 full lowercase on valid Unicode scalars, using its Unicode 15 tables.
/// Final_Sigma examines original text. No normalization, casefold or OS Unicode properties.
/// Source and modifications are documented in labs/python-unicode-tables/README.md.
/// </summary>
internal static class PythonUnicodeLower
{
    internal const string ResourceName = "ComfySharp.Nodes.Resources.PythonUnicode.lower-3.12.10.json";
    internal const string ResourceSha256 = "6381d72114b8aa4ee5c0dd04fcbb55395964ab1ca92832e7784f887dfbf8b42c";
    private static readonly Lazy<Tables> Data = new(ReadEmbedded, LazyThreadSafetyMode.ExecutionAndPublication);

    internal static void ValidateUnicode(string text, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        cancellationToken.ThrowIfCancellationRequested();
        int scalars = 0;
        for (int i = 0; i < text.Length;)
        {
            if ((scalars++ & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            if (Rune.DecodeFromUtf16(text.AsSpan(i), out _, out int consumed) != OperationStatus.Done)
                throw InvalidUnicode(i);
            i += consumed;
        }
        cancellationToken.ThrowIfCancellationRequested();
    }

    internal static string Lower(string text, CancellationToken cancellationToken = default)
    {
        ValidateUnicode(text, cancellationToken);
        if (text.Length == 0) return string.Empty;
        var data = Data.Value;
        cancellationToken.ThrowIfCancellationRequested();
        BitArray? followingCased = null;
        if (text.Contains('\u03a3'))
        {
            // One bit per UTF-16 offset, meaningful only at a capital sigma. A backward
            // pass replaces repeated context scans, including arbitrarily long combining runs.
            followingCased = new BitArray(text.Length);
            bool next = false;
            int scalars = 0;
            for (int end = text.Length; end > 0;)
            {
                if ((scalars++ & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
                if (Rune.DecodeLastFromUtf16(text.AsSpan(0, end), out var rune, out int consumed) != OperationStatus.Done)
                    throw InvalidUnicode(end - 1);
                end -= consumed;
                int cp = rune.Value;
                if (cp == 0x03a3) followingCased[end] = next;
                // Case_Ignorable wins when both properties are set, e.g. U+0345.
                if (!InRanges(data.Ignorable, cp)) next = InRanges(data.Cased, cp);
            }
        }
        var output = new StringBuilder(Math.Min(text.Length, 4096));
        bool previous = false;
        int count = 0;
        for (int i = 0; i < text.Length;)
        {
            if ((count++ & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            if (Rune.DecodeFromUtf16(text.AsSpan(i), out var rune, out int consumed) != OperationStatus.Done)
                throw InvalidUnicode(i);
            int cp = rune.Value;
            if (cp == 0x03a3)
            {
                Reserve(1);
                output.Append(previous && !followingCased![i] ? '\u03c2' : '\u03c3');
            }
            else if (data.Lower.TryGetValue(cp, out var mapped))
            {
                Reserve(mapped.Length); output.Append(mapped);
            }
            else
            {
                Reserve(consumed); output.Append(text.AsSpan(i, consumed));
            }
            if (!InRanges(data.Ignorable, cp)) previous = InRanges(data.Cased, cp);
            i += consumed;
        }
        cancellationToken.ThrowIfCancellationRequested();
        return output.ToString();

        void Reserve(int units)
        {
            if ((long)output.Length + units > int.MaxValue)
                throw new InvalidOperationException("unicode_lower_limit: output exceeds the .NET string length limit.");
        }
    }

    private static ArgumentException InvalidUnicode(int offset) => new(
        $"unsupported_unicode_value at UTF-16 position {offset}: isolated surrogate; valid Unicode scalars are required.", "text");

    private readonly record struct Range(int First, int Last);
    private sealed record Tables(FrozenDictionary<int, string> Lower, Range[] Cased, Range[] Ignorable);

    private static bool InRanges(Range[] ranges, int cp)
    {
        int lo = 0, hi = ranges.Length - 1;
        while (lo <= hi)
        {
            int middle = lo + (hi - lo) / 2;
            var range = ranges[middle];
            if (cp < range.First) hi = middle - 1;
            else if (cp > range.Last) lo = middle + 1;
            else return true;
        }
        return false;
    }

    private static Tables ReadEmbedded()
    {
        using var stream = typeof(PythonUnicodeLower).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidDataException("Pinned Python Unicode table resource is missing.");
        if (stream.Length != 29026) throw new InvalidDataException("Pinned Python Unicode resource length differs.");
        var bytes = new byte[29026]; stream.ReadExactly(bytes);
        return ReadTables(bytes, ResourceSha256);
    }

    // Test access validates malformed copies with an independently supplied digest; production
    // always uses the constant above. This method never changes the process's immutable tables.
    internal static void ValidateTableResource(byte[] bytes, string expectedSha256) => _ = ReadTables(bytes, expectedSha256);

    private static Tables ReadTables(byte[] bytes, string expectedSha256)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        if (!string.Equals(Convert.ToHexStringLower(SHA256.HashData(bytes)), expectedSha256, StringComparison.Ordinal))
            throw new InvalidDataException("Pinned Python Unicode resource SHA-256 differs.");
        try
        {
            using var document = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 8 });
            var root = document.RootElement;
            Keys(root, "schema", "pythonVersion", "unicodeVersion", "cpythonCommit", "provenanceSha256", "extractorSha256",
                "lower", "casedRanges", "caseIgnorableRanges", "counts");
            Require(root.GetProperty("schema").GetInt32() == 1 && root.GetProperty("pythonVersion").GetString() == "3.12.10" &&
                root.GetProperty("unicodeVersion").GetString() == "15.0.0" &&
                root.GetProperty("cpythonCommit").GetString() == "0cc81280367df838c4b199f8f0378837165071c2" &&
                root.GetProperty("provenanceSha256").GetString() == "8193e086fb80c5597d7fc6c92ffc8749aab78b469b77bcd0681b3da82cb0eff6",
                "Unexpected table provenance.");
            string? extractor = root.GetProperty("extractorSha256").GetString();
            Require(extractor is { Length: 64 } && extractor.All(c => char.IsAsciiDigit(c) || c is >= 'a' and <= 'f'), "Invalid extractor digest.");
            var lower = new Dictionary<int, string>();
            int previous = -1, expansions = 0;
            foreach (var entry in root.GetProperty("lower").EnumerateArray())
            {
                Require(entry.GetArrayLength() == 2, "Malformed lowercase record.");
                int cp = entry[0].GetInt32();
                Require(Scalar(cp) && cp > previous, "Lowercase keys must be sorted unique scalars.");
                previous = cp;
                int length = entry[1].GetArrayLength();
                Require(length is >= 1 and <= 3, "Invalid lowercase expansion length.");
                var text = new StringBuilder(6);
                foreach (var item in entry[1].EnumerateArray())
                {
                    int mapped = item.GetInt32(); Require(Scalar(mapped), "Invalid mapped scalar.");
                    text.Append(new Rune(mapped).ToString());
                }
                Require(length != 1 || entry[1][0].GetInt32() != cp, "Identity mappings must be omitted.");
                if (length > 1) expansions++;
                lower.Add(cp, text.ToString());
            }
            var cased = ReadRanges(root.GetProperty("casedRanges"));
            var ignorable = ReadRanges(root.GetProperty("caseIgnorableRanges"));
            var counts = root.GetProperty("counts");
            Keys(counts, "scalars", "typeRecords", "extendedScalars", "index1", "index2", "lowerMappings",
                "expandingMappings", "casedScalars", "caseIgnorableScalars");
            Require(counts.GetProperty("scalars").GetInt32() == 1112064 && counts.GetProperty("typeRecords").GetInt32() == 504 &&
                counts.GetProperty("extendedScalars").GetInt32() == 1236 && counts.GetProperty("index1").GetInt32() == 8704 &&
                counts.GetProperty("index2").GetInt32() == 36224, "Invalid C table dimensions.");
            Require(lower.Count == 1433 && counts.GetProperty("lowerMappings").GetInt32() == lower.Count &&
                expansions == 1 && counts.GetProperty("expandingMappings").GetInt32() == expansions,
                "Invalid lowercase mapping counts.");
            Require(cased.Sum(v => v.Last - v.First + 1) == 4526 && counts.GetProperty("casedScalars").GetInt32() == 4526 &&
                ignorable.Sum(v => v.Last - v.First + 1) == 2707 && counts.GetProperty("caseIgnorableScalars").GetInt32() == 2707,
                "Invalid property range counts.");
            return new(lower.ToFrozenDictionary(), cased, ignorable);
        }
        catch (Exception error) when (error is JsonException or InvalidOperationException or KeyNotFoundException or FormatException or OverflowException)
        {
            throw new InvalidDataException("Malformed pinned Python Unicode table resource.", error);
        }
    }

    private static Range[] ReadRanges(JsonElement element)
    {
        var ranges = new List<Range>(); int previous = -2;
        foreach (var entry in element.EnumerateArray())
        {
            Require(entry.GetArrayLength() == 2, "Malformed property range.");
            int first = entry[0].GetInt32(), last = entry[1].GetInt32();
            Require(Scalar(first) && Scalar(last) && first <= last && first > previous + 1 &&
                !(first < 0xd800 && last > 0xdfff), "Property ranges must be merged, sorted, disjoint and scalar-only.");
            ranges.Add(new(first, last)); previous = last;
        }
        return ranges.ToArray();
    }

    private static bool Scalar(int cp) => cp is >= 0 and <= 0x10ffff && cp is not (>= 0xd800 and <= 0xdfff);
    private static void Keys(JsonElement value, params string[] expected)
    {
        var found = value.EnumerateObject().Select(p => p.Name).ToArray();
        Require(found.Length == expected.Length && found.Distinct(StringComparer.Ordinal).Count() == found.Length &&
            expected.All(name => found.Contains(name, StringComparer.Ordinal)), "Unexpected, duplicate or missing resource property.");
    }
    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidDataException(message);
    }
}
