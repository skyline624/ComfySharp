using System.Buffers;
using System.Collections.Frozen;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace ComfySharp.Nodes;

/// <summary>
/// CPython 3.12.10 Unicode 15 full upper/title mappings and original-text capitalization.
/// No culture, normalization or casefold. See labs/python-unicode-case-tables/README.md.
/// </summary>
internal static class PythonUnicodeCase
{
    internal const string ResourceName = "ComfySharp.Nodes.Resources.PythonUnicode.upper-title-3.12.10.json";
    internal const string ResourceSha256 = "64b2f8c52989663aa8ff7e8d72afa31cdf813fbd80714ed12ab840222ee0ebe7";
    private static readonly Lazy<Tables> Data = new(ReadEmbedded, LazyThreadSafetyMode.ExecutionAndPublication);

    internal static string Lower(string text, CancellationToken cancellationToken = default) =>
        PythonUnicodeLower.Lower(text, cancellationToken);

    internal static string Upper(string text, CancellationToken cancellationToken = default)
    {
        PythonUnicodeLower.ValidateUnicode(text, cancellationToken);
        if (text.Length == 0) return string.Empty;
        var data = Data.Value;
        cancellationToken.ThrowIfCancellationRequested();
        var output = new StringBuilder(Math.Min(text.Length, 4096));
        int count = 0;
        for (int i = 0; i < text.Length;)
        {
            if ((count++ & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            if (Rune.DecodeFromUtf16(text.AsSpan(i), out var rune, out int consumed) != OperationStatus.Done)
                throw new InvalidOperationException("Validated Unicode string changed unexpectedly.");
            if (data.Upper.TryGetValue(rune.Value, out var mapped))
            {
                Reserve(mapped.Length); output.Append(mapped);
            }
            else
            {
                Reserve(consumed); output.Append(text.AsSpan(i, consumed));
            }
            i += consumed;
        }
        cancellationToken.ThrowIfCancellationRequested();
        return output.ToString();

        void Reserve(int units)
        {
            if ((long)output.Length + units > int.MaxValue)
                throw new InvalidOperationException("unicode_case_limit: output exceeds the .NET string length limit.");
        }
    }

    internal static string Capitalize(string text, CancellationToken cancellationToken = default) =>
        WithTitle(text, false, cancellationToken);

    internal static string Title(string text, CancellationToken cancellationToken = default) =>
        WithTitle(text, true, cancellationToken);

    private static string WithTitle(string text, bool allWords, CancellationToken cancellationToken)
    {
        PythonUnicodeLower.ValidateUnicode(text, cancellationToken);
        if (text.Length == 0) return string.Empty;
        var data = Data.Value;
        cancellationToken.ThrowIfCancellationRequested();
        return PythonUnicodeLower.ApplyTitle(text, data.Title, allWords, cancellationToken);
    }

    private sealed record Tables(FrozenDictionary<int, string> Upper, FrozenDictionary<int, string> Title);

    private static Tables ReadEmbedded()
    {
        using var stream = typeof(PythonUnicodeCase).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidDataException("Pinned Python Unicode upper/title resource is missing.");
        if (stream.Length != 44021) throw new InvalidDataException("Pinned upper/title resource length differs.");
        var bytes = new byte[44021]; stream.ReadExactly(bytes);
        return ReadTables(bytes, ResourceSha256);
    }

    // Validates an isolated malformed copy in tests; cannot replace the immutable runtime tables.
    internal static void ValidateTableResource(byte[] bytes, string expectedSha256) => _ = ReadTables(bytes, expectedSha256);

    private static Tables ReadTables(byte[] bytes, string expectedSha256)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        if (!string.Equals(Convert.ToHexStringLower(SHA256.HashData(bytes)), expectedSha256, StringComparison.Ordinal))
            throw new InvalidDataException("Pinned upper/title resource SHA-256 differs.");
        try
        {
            using var document = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 8 });
            var root = document.RootElement;
            Keys(root, "schema", "pythonVersion", "unicodeVersion", "cpythonCommit", "provenanceSha256", "extractorSha256",
                "upper", "title", "counts");
            Require(root.GetProperty("schema").GetInt32() == 1 && root.GetProperty("pythonVersion").GetString() == "3.12.10" &&
                root.GetProperty("unicodeVersion").GetString() == "15.0.0" &&
                root.GetProperty("cpythonCommit").GetString() == "0cc81280367df838c4b199f8f0378837165071c2" &&
                root.GetProperty("provenanceSha256").GetString() == "f453902eb2ba3a21f042667b17d132dcaa052555005e9c77c48fbc4bb104f4fa" &&
                root.GetProperty("extractorSha256").GetString() == "94aac707581c7a0c442fa4086739dbbe12d5a7d4f6efef4a279fac5f0e1c0737",
                "Unexpected upper/title provenance.");
            var (upper, upperExpansions) = ReadMappings(root.GetProperty("upper"));
            var (title, titleExpansions) = ReadMappings(root.GetProperty("title"));
            var counts = root.GetProperty("counts");
            Keys(counts, "scalars", "typeRecords", "extendedScalars", "index1", "index2",
                "upperMappings", "upperExpansions", "titleMappings", "titleExpansions");
            Require(counts.GetProperty("scalars").GetInt32() == 1112064 && counts.GetProperty("typeRecords").GetInt32() == 504 &&
                counts.GetProperty("extendedScalars").GetInt32() == 1236 && counts.GetProperty("index1").GetInt32() == 8704 &&
                counts.GetProperty("index2").GetInt32() == 36224, "Unexpected C table dimensions.");
            Require(upper.Count == 1525 && title.Count == 1452 && upperExpansions == 102 && titleExpansions == 48 &&
                counts.GetProperty("upperMappings").GetInt32() == upper.Count &&
                counts.GetProperty("titleMappings").GetInt32() == title.Count &&
                counts.GetProperty("upperExpansions").GetInt32() == upperExpansions &&
                counts.GetProperty("titleExpansions").GetInt32() == titleExpansions,
                "Unexpected full mapping counts.");
            return new(upper.ToFrozenDictionary(), title.ToFrozenDictionary());
        }
        catch (Exception error) when (error is JsonException or InvalidOperationException or KeyNotFoundException or FormatException or OverflowException)
        {
            throw new InvalidDataException("Malformed pinned upper/title resource.", error);
        }
    }

    private static (Dictionary<int, string> Mappings, int Expansions) ReadMappings(JsonElement entries)
    {
        var mappings = new Dictionary<int, string>(); int previous = -1, expansions = 0;
        foreach (var entry in entries.EnumerateArray())
        {
            Require(entry.GetArrayLength() == 2, "Malformed mapping record.");
            int cp = entry[0].GetInt32();
            Require(Scalar(cp) && cp > previous, "Mapping keys must be sorted unique scalars.");
            previous = cp;
            int length = entry[1].GetArrayLength();
            Require(length is >= 1 and <= 3, "Invalid full mapping length.");
            var text = new StringBuilder(6);
            foreach (var item in entry[1].EnumerateArray())
            {
                int mapped = item.GetInt32(); Require(Scalar(mapped), "Invalid mapped scalar.");
                text.Append(new Rune(mapped).ToString());
            }
            Require(length != 1 || entry[1][0].GetInt32() != cp, "Identity mappings must be omitted.");
            if (length > 1) expansions++;
            mappings.Add(cp, text.ToString());
        }
        return (mappings, expansions);
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
