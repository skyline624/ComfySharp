using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Tokenization;
using Xunit;

namespace ComfySharp.Tokenization.Tests;

public sealed class ClipTextReferenceTests
{
    private const string CorpusHash = "bc92c74136a6ed38a314029f083305acbe8fae5aa8acb6c0808db3b8237f6cc2";
    private static readonly byte[] Bytes = ReadCorpus();
    private static readonly JsonDocument Corpus = JsonDocument.Parse(Bytes);
    private static readonly IReadOnlyDictionary<string, JsonElement> Cases = Corpus.RootElement.GetProperty("cases")
        .EnumerateArray().ToDictionary(c => c.GetProperty("id").GetString()!, c => c);
    private static readonly Lazy<ClipTokenizer> Tokenizer = new(() => new ClipTokenizer());

    public static IEnumerable<object[]> BpeCases => OfKind("bpe");
    public static IEnumerable<object[]> WeightCases => OfKind("weights");
    public static IEnumerable<object[]> PackingCases => OfKind("packing");
    public static IEnumerable<object[]> WrapperCases => OfKind("wrapper");
    private static IEnumerable<object[]> OfKind(string kind) => Cases.Where(pair => pair.Value.GetProperty("kind").GetString() == kind)
        .Select(pair => new object[] { pair.Key });

    [Fact]
    public void CorpusHasPinnedSourceProfileAndUnmodifiedIndependentOutputs()
    {
        Assert.Equal(CorpusHash, Convert.ToHexStringLower(SHA256.HashData(Bytes)));
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", Corpus.RootElement.GetProperty("backendCommit").GetString());
        Assert.Equal("clip-transformers-5.14.1-tokenizers-0.22.2", Corpus.RootElement.GetProperty("profile").GetString());
        Assert.False(Corpus.RootElement.GetProperty("laboratory").GetProperty("modelWeightsUsed").GetBoolean());
        Assert.Equal(193, Cases.Count);
        Assert.Equal(113, BpeCases.Count());
        Assert.Equal(17, WeightCases.Count());
        Assert.Equal(59, PackingCases.Count());
        Assert.Equal(4, WrapperCases.Count());
    }

    [Theory, MemberData(nameof(BpeCases))]
    public void TokensAndDecodesMatchPinnedDependency(string id)
    {
        var item = Cases[id];
        var text = item.GetProperty("text").GetString()!;
        var expected = Ints(item.GetProperty("ids"));
        Assert.Equal(expected, Tokenizer.Value.Encode(text));
        Assert.Equal(Ints(item.GetProperty("bareIds")), Tokenizer.Value.Encode(text, addSpecialTokens: false));
        Assert.Equal(item.GetProperty("decode").GetString(), Tokenizer.Value.Decode(expected));
        Assert.Equal(item.GetProperty("decodeWithSpecial").GetString(), Tokenizer.Value.Decode(expected, skipSpecialTokens: false));
        Assert.Equal(item.GetProperty("tokens").EnumerateArray().Select(t => t.GetString()), expected.Select(Tokenizer.Value.GetToken));
    }

    [Theory, MemberData(nameof(WeightCases))]
    public void WeightSegmentsMatchFrozenComfySource(string id)
    {
        var item = Cases[id];
        var actual = PromptWeights.Parse(item.GetProperty("text").GetString()!, item.GetProperty("disableWeights").GetBoolean());
        var expected = item.GetProperty("segments");
        Assert.Equal(expected.GetArrayLength(), actual.Count);
        for (var i = 0; i < actual.Count; i++)
        {
            Assert.Equal(expected[i][0].GetString(), actual[i].Text);
            EqualWeight(expected[i][1].GetString()!, actual[i].Weight, $"{id}: segment {i}");
        }
    }

    [Theory, MemberData(nameof(PackingCases))]
    public void WeightedChunksMatchFrozenComfySource(string id)
    {
        var item = Cases[id];
        var tokenizer = new ComfyClipTokenizer(Tokenizer.Value, Profile(item.GetProperty("profile").GetString()!), Construction(item));
        var result = tokenizer.Tokenize(item.GetProperty("text").GetString()!, Call(item));
        Assert.True(result.ReturnWordIds);
        EqualChunks(item.GetProperty("chunks"), result.Chunks, id);
    }

    [Theory, MemberData(nameof(WrapperCases))]
    public void ProfileCompositionMatchesFrozenWrappers(string id)
    {
        var item = Cases[id];
        var text = item.GetProperty("text").GetString()!;
        var expected = item.GetProperty("outputs");
        if (item.GetProperty("profile").GetString() == "sdxl")
        {
            var settings = Construction(item);
            var actual = new ComfySdxlTokenizer(Tokenizer.Value, settings, settings).Tokenize(text, Call(item));
            Assert.Equal(expected.EnumerateObject().Select(p => p.Name), actual.Keys);
            foreach (var pair in actual) EqualChunks(expected.GetProperty(pair.Key), pair.Value.Chunks, id + ":" + pair.Key);
        }
        else
        {
            var actual = new ComfyClipTokenizer(Tokenizer.Value, ClipProfile.Sd1L, Construction(item)).Tokenize(text, Call(item));
            EqualChunks(expected.GetProperty("l"), actual.Chunks, id + ":l");
        }
    }

    private static ClipProfile Profile(string profile) => profile switch
    {
        "sd1-l" => ClipProfile.Sd1L, "sdxl-l" => ClipProfile.SdXlL, "sdxl-g" => ClipProfile.SdXlG,
        _ => throw new InvalidDataException("Unknown reference profile.")
    };

    private static ClipPackingOptions Construction(JsonElement item)
    {
        var config = item.TryGetProperty("config", out var found) ? found : default;
        return new ClipPackingOptions
        {
            MaxLength = Number(config, "max_length") ?? 77,
            MinLength = Number(config, "min_length"), MinPadding = Number(config, "min_padding"), PadToken = Number(config, "pad_token"),
            PadToMaxLength = Boolean(config, "pad_to_max_length") ?? true,
            PadLeft = Boolean(config, "pad_left") ?? false, DisableWeights = Boolean(config, "disable_weights") ?? false,
            TokenizerData = Dictionary(item.GetProperty("data"))
        };
    }

    private static ClipTokenizeOptions Call(JsonElement item)
    {
        var call = item.GetProperty("call");
        return new ClipTokenizeOptions
        {
            ReturnWordIds = true, DisableWeights = Boolean(call, "disable_weights"),
            OverrideMinLength = call.TryGetProperty("min_length", out _), MinLength = Number(call, "min_length"),
            TokenizerOptions = Dictionary(item.GetProperty("options"))
        };
    }
    private static int? Number(JsonElement value, string name) => value.ValueKind == JsonValueKind.Object && value.TryGetProperty(name, out var prop) && prop.ValueKind != JsonValueKind.Null ? prop.GetInt32() : null;
    private static bool? Boolean(JsonElement value, string name) => value.ValueKind == JsonValueKind.Object && value.TryGetProperty(name, out var prop) ? prop.GetBoolean() : null;
    private static IReadOnlyDictionary<string, int?> Dictionary(JsonElement value) => value.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.ValueKind == JsonValueKind.Null ? (int?)null : p.Value.GetInt32());
    private static int[] Ints(JsonElement values) => values.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static void EqualChunks(JsonElement expected, IReadOnlyList<IReadOnlyList<ClipToken>> actual, string context)
    {
        Assert.True(expected.GetArrayLength() == actual.Count, $"{context}: expected {expected.GetArrayLength()} chunks, got {actual.Count}.");
        for (var c = 0; c < actual.Count; c++)
        {
            Assert.True(expected[c].GetArrayLength() == actual[c].Count, $"{context}: chunk {c} length differs.");
            for (var t = 0; t < actual[c].Count; t++)
            {
                Assert.True(expected[c][t][0].GetInt32() == actual[c][t].Id, $"{context}: chunk {c}, token {t} ID differs ({expected[c][t][0]} / {actual[c][t].Id}).");
                Assert.True(expected[c][t][2].GetInt32() == actual[c][t].WordId, $"{context}: chunk {c}, token {t} word ID differs.");
                EqualWeight(expected[c][t][1].GetString()!, actual[c][t].Weight, $"{context}: chunk {c}, token {t}");
            }
        }
    }

    private static void EqualWeight(string hex, double actual, string context)
    {
        var expectedBits = ulong.Parse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture);
        var actualBits = unchecked((ulong)BitConverter.DoubleToInt64Bits(actual));
        var expected = BitConverter.Int64BitsToDouble(unchecked((long)expectedBits));
        if (double.IsNaN(expected))
            Assert.True(double.IsNaN(actual) && (expectedBits >> 63) == (actualBits >> 63), $"{context}: NaN classification/sign differs.");
        else
            Assert.True(expectedBits == actualBits, $"{context}: weight bits {hex} / {actualBits:x16} differ.");
    }

    private static byte[] ReadCorpus()
    {
        using var stream = typeof(ClipTextReferenceTests).Assembly.GetManifestResourceStream("ComfySharp.Tokenization.Tests.Fixtures.clip-text.reference.json")
            ?? throw new InvalidOperationException("Missing independent CLIP reference corpus.");
        using var output = new MemoryStream();
        stream.CopyTo(output);
        return output.ToArray();
    }
}
