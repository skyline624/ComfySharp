using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>Managed examples and invariants; independent source digests are a separate corpus.</summary>
public sealed class PythonUnicodeCaseTests
{
    [Theory]
    [InlineData("upper", "", "")]
    [InlineData("upper", "Hello 123", "HELLO 123")]
    [InlineData("upper", "straße", "STRASSE")]
    [InlineData("upper", "\ufb03", "FFI")]
    [InlineData("upper", "\u01f3", "\u01f1")]
    [InlineData("upper", "iıİ", "IIİ")]
    [InlineData("upper", "σς", "ΣΣ")]
    [InlineData("upper", "\U00010428😀\0e\u0301", "\U00010400😀\0E\u0301")]
    [InlineData("capitalize", "", "")]
    [InlineData("capitalize", "hELLO WORLD", "Hello world")]
    [InlineData("capitalize", "ßABC", "Ssabc")]
    [InlineData("capitalize", "\ufb03ABC", "Ffiabc")]
    [InlineData("capitalize", "\u01f3ABC", "\u01f2abc")]
    [InlineData("capitalize", "ΟΣ", "Ος")]
    [InlineData("capitalize", "A\u0345Σ", "A\u0345ς")]
    [InlineData("capitalize", "1ABC", "1abc")]
    [InlineData("capitalize", "\U00010428ABC", "\U00010400abc")]
    [InlineData("capitalize", "😀ABC\0DEF", "😀abc\0def")]
    [InlineData("title", "", "")]
    [InlineData("title", "hELLO wORLD", "Hello World")]
    [InlineData("title", "they're HERE", "They'Re Here")]
    [InlineData("title", "1ABC 2DEF", "1Abc 2Def")]
    [InlineData("title", "ßABC \ufb03XYZ", "Ssabc Ffixyz")]
    [InlineData("title", "\u01f3ABC", "\u01f2abc")]
    [InlineData("title", "a\u0301B", "A\u0301B")]
    [InlineData("title", "a\u0345B", "A\u0345b")]
    [InlineData("title", "A\u0301Σ", "A\u0301Σ")]
    [InlineData("title", "A\u0345Σ", "A\u0345ς")]
    [InlineData("title", "AΣ\u0345A", "Aσ\u0345a")]
    [InlineData("title", "AΣ\u0345", "Aς\u0345")]
    [InlineData("title", "\U00010428ABC\0dEF", "\U00010400abc\0Def")]
    [InlineData("lower", "ΟΣ İ ß", "ος i\u0307 ß")]
    public void FullMappingsAndOriginalContextHaveExplicitExamples(string operation, string input, string expected)
    {
        string before = input;
        Assert.Equal(expected, Convert(operation, input));
        Assert.Equal(before, input);
    }

    [Theory]
    [InlineData("en-US")]
    [InlineData("tr-TR")]
    [InlineData("el-GR")]
    public void CultureDoesNotChangeCasing(string culture)
    {
        var old = CultureInfo.CurrentCulture;
        var oldUi = CultureInfo.CurrentUICulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.CurrentUICulture = CultureInfo.GetCultureInfo(culture);
            Assert.Equal("IISS", PythonUnicodeCase.Upper("iıß"));
            Assert.Equal("Iς", PythonUnicodeCase.Capitalize("iΣ"));
            Assert.Equal("I\u0345ς", PythonUnicodeCase.Title("i\u0345Σ"));
        }
        finally { CultureInfo.CurrentCulture = old; CultureInfo.CurrentUICulture = oldUi; }
    }

    [Theory]
    [InlineData("upper")]
    [InlineData("capitalize")]
    [InlineData("title")]
    [InlineData("lower")]
    public void NullAndCancellationAreExplicitForEveryEntryPoint(string operation)
    {
        Assert.Throws<ArgumentNullException>(() => Convert(operation, null!));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        foreach (string value in new[] { "", "valid", new string((char)0xd800, 1) })
        {
            var error = Assert.Throws<OperationCanceledException>(() => Convert(operation, value, cancelled.Token));
            Assert.Equal(cancelled.Token, error.CancellationToken);
        }
    }

    [Theory]
    [InlineData("upper", 0xd800)]
    [InlineData("upper", 0xdc00)]
    [InlineData("capitalize", 0xd800)]
    [InlineData("capitalize", 0xdc00)]
    [InlineData("title", 0xd800)]
    [InlineData("title", 0xdc00)]
    public void IsolatedSurrogatesAreConstructedAtRuntimeAndRejected(string operation, int codeUnit)
    {
        string value = "😀" + (char)codeUnit + "x";
        var error = Assert.Throws<ArgumentException>(() => Convert(operation, value));
        Assert.Equal("text", error.ParamName);
        Assert.Contains("unsupported_unicode_value at UTF-16 position 2", error.Message);
    }

    [Fact]
    public void ContextPassIsBoundedForLongIgnorableRunsAndUsesTwoDifferentStates()
    {
        string marks = new('\u0345', 80_000);
        string input = "A" + marks + "Σ";
        Assert.Equal("A" + marks + "ς", PythonUnicodeCase.Capitalize(input));
        Assert.Equal("A" + marks + "ς", PythonUnicodeCase.Title(input));
        string accents = new('\u0301', 80_000);
        Assert.Equal("A" + accents + "Σ", PythonUnicodeCase.Title("A" + accents + "Σ"));
        Assert.Equal("A" + accents + "ς", PythonUnicodeCase.Capitalize("A" + accents + "Σ"));
    }

    [Fact]
    public void NoNormalizationTrimmingOrWordLibraryIsApplied()
    {
        Assert.Equal(" E\u0301 \n", PythonUnicodeCase.Upper(" e\u0301 \n"));
        Assert.NotEqual(PythonUnicodeCase.Upper("é"), PythonUnicodeCase.Upper("e\u0301"));
        Assert.Equal("a b\0σ", PythonUnicodeCase.Lower("A B\0Σ"));
        Assert.Equal(PythonUnicodeLower.Lower("İ AΣ\u0345"), PythonUnicodeCase.Lower("İ AΣ\u0345"));
    }

    [Fact]
    public async Task ImmutableTablesSupportConcurrentIndependentCalls()
    {
        await Task.WhenAll(Enumerable.Range(0, 16).Select(_ => Task.Run(() =>
        {
            for (int i = 0; i < 20; i++)
            {
                Assert.Equal("FFI STRASSE", PythonUnicodeCase.Upper("\ufb03 straße"));
                Assert.Equal("Ss ος", PythonUnicodeCase.Capitalize("ß ΟΣ"));
                Assert.Equal("A\u0345ς", PythonUnicodeCase.Title("a\u0345Σ"));
                Assert.Equal("ος", PythonUnicodeCase.Lower("ΟΣ"));
            }
        })));
    }

    [Fact]
    public void ResourceIdentityAndHistoricalLowerResourceAreBothPinned()
    {
        byte[] bytes = Resource(PythonUnicodeCase.ResourceName);
        Assert.Equal(44021, bytes.Length);
        Assert.Equal(PythonUnicodeCase.ResourceSha256, Digest(bytes));
        PythonUnicodeCase.ValidateTableResource(bytes, PythonUnicodeCase.ResourceSha256);
        byte[] lower = Resource(PythonUnicodeLower.ResourceName);
        Assert.Equal(29026, lower.Length);
        Assert.Equal("6381d72114b8aa4ee5c0dd04fcbb55395964ab1ca92832e7784f887dfbf8b42c", Digest(lower));
    }

    [Fact]
    public void WrongDigestAndMalformedJsonAreRejectedWithoutReplacingTables()
    {
        byte[] bytes = Resource(PythonUnicodeCase.ResourceName); bytes[0] ^= 1;
        Assert.Throws<InvalidDataException>(() => PythonUnicodeCase.ValidateTableResource(bytes, PythonUnicodeCase.ResourceSha256));
        byte[] malformed = Encoding.UTF8.GetBytes("{");
        Assert.Throws<InvalidDataException>(() => PythonUnicodeCase.ValidateTableResource(malformed, Digest(malformed)));
        Assert.Equal("STRASSE", PythonUnicodeCase.Upper("straße"));
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("duplicate-property")]
    [InlineData("unknown")]
    [InlineData("schema")]
    [InlineData("provenance")]
    [InlineData("extractor")]
    [InlineData("empty-map")]
    [InlineData("duplicate-key")]
    [InlineData("unsorted")]
    [InlineData("surrogate-key")]
    [InlineData("surrogate-value")]
    [InlineData("outside-value")]
    [InlineData("identity")]
    [InlineData("empty-expansion")]
    [InlineData("long-expansion")]
    [InlineData("record-shape")]
    [InlineData("count")]
    [InlineData("integer-overflow")]
    public void ResourceStructureIsValidatedEvenWithMatchingReplacementDigest(string mutation)
    {
        var root = JsonNode.Parse(Resource(PythonUnicodeCase.ResourceName))!.AsObject();
        var map = root["title"]!.AsArray();
        switch (mutation)
        {
            case "missing": root.Remove("upper"); break;
            case "unknown": root["newField"] = 1; break;
            case "schema": root["schema"] = 2; break;
            case "provenance": root["provenanceSha256"] = new string('0', 64); break;
            case "extractor": root["extractorSha256"] = new string('0', 64); break;
            case "empty-map": root["upper"] = new JsonArray(); break;
            case "duplicate-key": map[1]![0] = map[0]![0]!.DeepClone(); break;
            case "unsorted": map[0]![0] = 0x10ffff; break;
            case "surrogate-key": map[0]![0] = 0xd800; break;
            case "surrogate-value": map[0]![1]![0] = 0xdc00; break;
            case "outside-value": map[0]![1]![0] = 0x110000; break;
            case "identity": map[0]![1] = new JsonArray(map[0]![0]!.DeepClone()); break;
            case "empty-expansion": map[0]![1] = new JsonArray(); break;
            case "long-expansion": map[0]![1] = new JsonArray(65, 65, 65, 65); break;
            case "record-shape": map[0]!.AsArray().Add(0); break;
            case "count": root["counts"]!["upperExpansions"] = 101; break;
            case "integer-overflow": map[0]![0] = long.MaxValue; break;
        }
        string text = root.ToJsonString();
        if (mutation == "duplicate-property") text = "{\"schema\":1," + text[1..];
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        Assert.Throws<InvalidDataException>(() => PythonUnicodeCase.ValidateTableResource(bytes, Digest(bytes)));
        Assert.Equal("Ss", PythonUnicodeCase.Capitalize("ß"));
    }

    private static string Convert(string operation, string value, CancellationToken token = default) => operation switch
    {
        "upper" => PythonUnicodeCase.Upper(value, token),
        "capitalize" => PythonUnicodeCase.Capitalize(value, token),
        "title" => PythonUnicodeCase.Title(value, token),
        "lower" => PythonUnicodeCase.Lower(value, token),
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static byte[] Resource(string name)
    {
        using var stream = typeof(StringContainsNode).Assembly.GetManifestResourceStream(name);
        Assert.NotNull(stream); using var bytes = new MemoryStream(); stream.CopyTo(bytes); return bytes.ToArray();
    }
    private static string Digest(byte[] bytes) => System.Convert.ToHexStringLower(SHA256.HashData(bytes));
}
