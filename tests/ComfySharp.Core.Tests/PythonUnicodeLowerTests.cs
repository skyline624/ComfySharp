using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>Managed Unicode/data contracts and known examples, not generated builtin-Python expectations.</summary>
public sealed class PythonUnicodeLowerTests
{
    [Theory]
    [InlineData("", "")]
    [InlineData("AbC 123\0\n", "abc 123\0\n")]
    [InlineData("Iİıi", "ii\u0307ıi")]
    [InlineData("ẞStraße", "ßstraße")]
    [InlineData("É E\u0301", "é e\u0301")]
    [InlineData("\U00010400\U00010401😀", "\U00010428\U00010429😀")]
    [InlineData("\U0001e900", "\U0001e922")]
    [InlineData("\u0378\U0010ffff", "\u0378\U0010ffff")]
    public void KnownScalarMappingsPreserveFullExpansionsAndUnmappedText(string input, string expected)
    {
        Assert.Equal(expected, PythonUnicodeLower.Lower(input));
        Assert.Equal(expected, PythonUnicodeLower.Lower(expected));
    }

    [Theory]
    [InlineData("Σ", "σ")]
    [InlineData("ΟΣ", "ος")]
    [InlineData("ΟΣΑ", "οσα")]
    [InlineData("AΣ", "aς")]
    [InlineData("ΣA", "σa")]
    [InlineData("AΣ A", "aς a")]
    [InlineData("AΣ\u0301", "aς\u0301")]
    [InlineData("AΣ\u0301B", "aσ\u0301b")]
    [InlineData("\u0345Σ", "\u0345σ")]
    [InlineData("AΣ\u0345B", "aσ\u0345b")]
    [InlineData("AΣ\u0345", "aς\u0345")]
    [InlineData("A\u0345Σ", "a\u0345ς")]
    [InlineData("A!Σ", "a!σ")]
    [InlineData("ΣΣ", "σς")]
    [InlineData("\U00010400Σ", "\U00010428ς")]
    public void FinalSigmaReadsOriginalContextSkippingIgnorableBeforeTestingCased(string input, string expected) =>
        Assert.Equal(expected, PythonUnicodeLower.Lower(input));

    [Fact]
    public void LowerIsNotCasefoldOrNormalizationAndDoesNotTrim()
    {
        Assert.Equal(" ß \t", PythonUnicodeLower.Lower(" ẞ \t"));
        Assert.NotEqual(PythonUnicodeLower.Lower("É"), PythonUnicodeLower.Lower("E\u0301"));
        Assert.NotEqual(PythonUnicodeLower.Lower("SS"), PythonUnicodeLower.Lower("ß"));
        Assert.NotEqual(PythonUnicodeLower.Lower("AΣ"), PythonUnicodeLower.Lower("A") + PythonUnicodeLower.Lower("Σ"));
    }

    [Theory]
    [InlineData("tr-TR")]
    [InlineData("fr-FR")]
    [InlineData("el-GR")]
    public void AmbientCultureCannotChangeUnicodeTablesOrContext(string culture)
    {
        var original = CultureInfo.CurrentCulture; var originalUi = CultureInfo.CurrentUICulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo(culture);
            CultureInfo.CurrentUICulture = CultureInfo.GetCultureInfo(culture);
            Assert.Equal("ii\u0307οσας", PythonUnicodeLower.Lower("IİΟΣΑΣ"));
        }
        finally { CultureInfo.CurrentCulture = original; CultureInfo.CurrentUICulture = originalUi; }
    }

    [Theory]
    [InlineData("", 0xd800, "", 0)]
    [InlineData("", 0xdc00, "", 0)]
    [InlineData("A", 0xd800, "B", 1)]
    [InlineData("😀", 0xdc00, "", 2)]
    public void InvalidUtf16IsRejectedByBothEntryPoints(string prefix, int codeUnit, string suffix, int offset)
    {
        // Attribute strings are encoded in metadata; construct invalid UTF-16 only at runtime.
        string text = prefix + (char)codeUnit + suffix;
        var error = Assert.Throws<ArgumentException>(() => PythonUnicodeLower.Lower(text));
        Assert.StartsWith($"unsupported_unicode_value at UTF-16 position {offset}:", error.Message);
        Assert.Equal("text", error.ParamName);
        Assert.Equal(error.Message, Assert.Throws<ArgumentException>(() => PythonUnicodeLower.ValidateUnicode(text)).Message);
    }

    [Fact]
    public void ValidationPreservesAllWellFormedScalarBoundaries()
    {
        string original = "\0\u007f\u0080\ud7ff\ue000\uffff\U00010000\U0010ffff";
        PythonUnicodeLower.ValidateUnicode(original);
        Assert.Equal(original, PythonUnicodeLower.Lower(original));
        Assert.Throws<ArgumentNullException>(() => PythonUnicodeLower.ValidateUnicode(null!));
        Assert.Throws<ArgumentNullException>(() => PythonUnicodeLower.Lower(null!));
    }

    [Fact]
    public void CancelledTokenPreemptsInvalidTextAndEmptyText()
    {
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        foreach (string input in new[] { "", "\ud800", "Σ" })
        {
            var error = Assert.ThrowsAny<OperationCanceledException>(() => PythonUnicodeLower.Lower(input, cancelled.Token));
            Assert.Equal(cancelled.Token, error.CancellationToken);
            Assert.ThrowsAny<OperationCanceledException>(() => PythonUnicodeLower.ValidateUnicode(input, cancelled.Token));
        }
    }

    [Fact]
    public void LongIgnorableRunsAndRepeatedSigmaUseOriginalNeighbours()
    {
        string ignored = new('\u0345', 10000);
        string piece = "AΣ" + ignored;
        string expected = string.Concat(Enumerable.Repeat("aσ" + ignored, 7)) + "aς" + ignored;
        string input = string.Concat(Enumerable.Repeat(piece, 8));
        Assert.Equal(expected, PythonUnicodeLower.Lower(input));
        // No stopwatch threshold: the two-pass algorithm is the complexity guarantee.
    }

    [Fact]
    public async Task ConcurrentCallsShareOnlyImmutableTableData()
    {
        var results = await Task.WhenAll(Enumerable.Range(0, 16).Select(_ => Task.Run(() => PythonUnicodeLower.Lower("İAΣ\u0345"))));
        Assert.All(results, result => Assert.Equal("i\u0307aς\u0345", result));
    }

    [Fact]
    public void ResourceHasPinnedIdentityAndValidatedSchema()
    {
        byte[] raw = Resource();
        Assert.Equal(29026, raw.Length);
        Assert.Equal(PythonUnicodeLower.ResourceSha256, Hash(raw));
        PythonUnicodeLower.ValidateTableResource(raw, PythonUnicodeLower.ResourceSha256);
        var root = JsonNode.Parse(raw)!.AsObject();
        Assert.Equal("15.0.0", root["unicodeVersion"]!.GetValue<string>());
        Assert.Equal(1433, root["lower"]!.AsArray().Count);
        var expansion = Assert.Single(root["lower"]!.AsArray(), entry => entry![1]!.AsArray().Count > 1)!;
        Assert.Equal(0x0130, expansion[0]!.GetValue<int>());
        Assert.Equal(new[] { 0x0069, 0x0307 }, expansion[1]!.AsArray().Select(v => v!.GetValue<int>()));
    }

    [Fact]
    public void ResourceIntegrityFailureDoesNotReplaceActiveTables()
    {
        byte[] raw = Resource(); raw[^1] ^= 1;
        Assert.Throws<InvalidDataException>(() => PythonUnicodeLower.ValidateTableResource(raw, PythonUnicodeLower.ResourceSha256));
        Assert.Equal("i\u0307", PythonUnicodeLower.Lower("İ"));
    }

    [Theory]
    [InlineData("schema")]
    [InlineData("duplicate-mapping")]
    [InlineData("surrogate-mapping")]
    [InlineData("empty-expansion")]
    [InlineData("identity-mapping")]
    [InlineData("reversed-range")]
    [InlineData("overlapping-range")]
    [InlineData("surrogate-range")]
    [InlineData("wrong-count")]
    [InlineData("extra-property")]
    [InlineData("duplicate-property")]
    public void MalformedTableStructureFailsEvenWhenTheTestSuppliesItsNewDigest(string mutation)
    {
        var root = JsonNode.Parse(Resource())!.AsObject();
        switch (mutation)
        {
            case "schema": root["schema"] = 2; break;
            case "duplicate-mapping": root["lower"]!.AsArray().Insert(1, root["lower"]![0]!.DeepClone()); break;
            case "surrogate-mapping": root["lower"]![0]![1]![0] = 0xd800; break;
            case "empty-expansion": root["lower"]![0]![1] = new JsonArray(); break;
            case "identity-mapping": root["lower"]![0]![1]![0] = root["lower"]![0]![0]!.DeepClone(); break;
            case "reversed-range": root["casedRanges"]![0]![0] = 1000; break;
            case "overlapping-range": root["casedRanges"]![1]![0] = 65; break;
            case "surrogate-range": root["casedRanges"]![0] = new JsonArray(0xd7ff, 0xe000); break;
            case "wrong-count": root["counts"]!["casedScalars"] = 0; break;
            case "extra-property": root["unexpected"] = true; break;
        }
        string json = root.ToJsonString();
        if (mutation == "duplicate-property") json = json.Insert(1, "\"schema\":1,");
        byte[] raw = Encoding.UTF8.GetBytes(json);
        Assert.Throws<InvalidDataException>(() => PythonUnicodeLower.ValidateTableResource(raw, Hash(raw)));
    }

    private static string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    private static byte[] Resource()
    {
        using var stream = typeof(StringFormatNode).Assembly.GetManifestResourceStream(PythonUnicodeLower.ResourceName);
        Assert.NotNull(stream); using var copy = new MemoryStream(); stream.CopyTo(copy); return copy.ToArray();
    }
}
