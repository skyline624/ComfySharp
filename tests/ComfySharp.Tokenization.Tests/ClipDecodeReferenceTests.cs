using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Tokenization;
using Xunit;

namespace ComfySharp.Tokenization.Tests;

public sealed class ClipDecodeReferenceTests
{
    private static readonly byte[] Bytes = ReadCorpus();
    private static readonly JsonDocument Corpus = JsonDocument.Parse(Bytes);
    private static readonly IReadOnlyDictionary<string, JsonElement> Cases = Corpus.RootElement.GetProperty("cases").EnumerateArray()
        .ToDictionary(item => item.GetProperty("id").GetString()!, item => item);
    private static readonly ClipTokenizer Tokenizer = new();
    public static IEnumerable<object[]> DecodeCases => Cases.Keys.Select(id => new object[] { id });

    [Fact]
    public void DecoderCorpusIsUnmodifiedAndPinned()
    {
        Assert.Equal("bbaff41e4ebfc9fb76ba4da5cf60013cf41a9f05d6539a4738350ed7d76ac326", Convert.ToHexStringLower(SHA256.HashData(Bytes)));
        Assert.Equal("clip-transformers-5.14.1-tokenizers-0.22.2", Corpus.RootElement.GetProperty("profile").GetString());
        Assert.Equal("e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349", Corpus.RootElement.GetProperty("vocabularySha256").GetString());
        Assert.Equal(269, Cases.Count);
    }

    [Theory, MemberData(nameof(DecodeCases))]
    public void DecoderMatchesReferenceForIndividualBytesAndInvalidUtf8(string id)
    {
        var item = Cases[id];
        var ids = item.GetProperty("ids").EnumerateArray().Select(value => value.GetInt32()).ToArray();
        Assert.Equal(item.GetProperty("decode").GetString(), Tokenizer.Decode(ids));
        Assert.Equal(item.GetProperty("decodeWithSpecial").GetString(), Tokenizer.Decode(ids, skipSpecialTokens: false));
    }

    private static byte[] ReadCorpus()
    {
        using var source = typeof(ClipDecodeReferenceTests).Assembly.GetManifestResourceStream("ComfySharp.Tokenization.Tests.Fixtures.clip-decode.reference.json")
            ?? throw new InvalidOperationException("Missing independent CLIP decoder corpus.");
        using var destination = new MemoryStream();
        source.CopyTo(destination);
        return destination.ToArray();
    }
}
