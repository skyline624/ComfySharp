using ComfySharp.Tokenization;
using Xunit;

namespace ComfySharp.Tokenization.Tests;

public sealed class ClipTokenizerTests
{
    [Theory]
    [InlineData("", new[] { 49406, 49407 })]
    [InlineData("Hello world!", new[] { 49406, 3306, 1002, 256, 49407 })]
    [InlineData("caf\u00e9", new[] { 49406, 15304, 49407 })]
    [InlineData("cafe\u0301", new[] { 49406, 15304, 49407 })]
    [InlineData("a &amp; b", new[] { 49406, 320, 261, 6259, 282, 321, 49407 })]
    [InlineData("a\0b", new[] { 49406, 320, 444, 321, 49407 })]
    [InlineData("\u0130", new[] { 49406, 328, 16384, 49407 })]
    [InlineData("!<|endoftext|>!", new[] { 49406, 256, 49407, 256, 49407 })]
    [InlineData("a<|STARTOFTEXT|>b", new[] { 49406, 320, 49406, 321, 49407 })]
    public void MatchesPinnedBackendExamples(string input, int[] expected)
        => Assert.Equal(expected, new ClipTokenizer().Encode(input));

    [Theory]
    [InlineData("<|startoftext|>", 49406)]
    [InlineData("<|STARTOFTEXT|>", 49406)]
    [InlineData("<|endoftext|>", 49407)]
    [InlineData("<|ENDOFTEXT|>", 49407)]
    public void SpecialTokensAreRecognizedWithinPunctuation(string special, int id)
    {
        var tokenizer = new ClipTokenizer();
        Assert.Equal(new[] { 256, id, 256 }, tokenizer.Encode("!" + special + "!", false));
    }

    [Fact]
    public void DecodePreservesHtmlControlsAndHandlesSpecialPolicy()
    {
        var tokenizer = new ClipTokenizer();
        Assert.Equal("a & amp ; b", tokenizer.Decode(tokenizer.Encode("a &amp; b")));
        Assert.Equal("a \0 b", tokenizer.Decode(tokenizer.Encode("a\0b")));
        Assert.Equal("<|startoftext|>hello <|endoftext|>", tokenizer.Decode(tokenizer.Encode("hello"), false));
        Assert.Equal("<|startoftext|>", tokenizer.GetToken(49406));
        Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.GetToken(49408));
    }

    [Fact]
    public void ResultsCannotMutateTheCache()
    {
        var tokenizer = new ClipTokenizer();
        var ids = tokenizer.Encode("hello", false);
        Assert.Throws<NotSupportedException>(() => ((IList<int>)ids)[0] = 0);
        Assert.Equal(new[] { 3306 }, tokenizer.Encode("hello", false));
    }

    [Fact]
    public async Task ConcurrentCallsAreDeterministicAndCacheIsBounded()
    {
        var tokenizer = new ClipTokenizer(cacheCapacity: 8);
        var expected = new ClipTokenizer(cacheCapacity: 0);
        await Task.WhenAll(Enumerable.Range(0, 100).Select(i => Task.Run(() =>
        {
            var text = "hello " + new string((char)('a' + i % 26), 1 + i % 21) + " \U00010400\u0130";
            Assert.Equal(expected.Encode(text), tokenizer.Encode(text));
        })));
        Assert.InRange(tokenizer.CachedTokenCount, 1, 8);
    }

    [Fact]
    public void LongInputIsNotTruncatedAndOversizedWordsAreNotCached()
    {
        var tokenizer = new ClipTokenizer(cacheCapacity: 2);
        Assert.Equal(9002, tokenizer.Encode(string.Concat(Enumerable.Repeat("a ", 9000))).Count);
        var uncached = new ClipTokenizer(cacheCapacity: 2);
        Assert.NotEmpty(uncached.Encode(new string('a', 10000)));
        Assert.Equal(0, uncached.CachedTokenCount);
    }

    [Fact]
    public void MalformedUtf16IsRejectedExplicitly()
    {
        Assert.Throws<ArgumentException>(() => new ClipTokenizer().Encode("a\ud800b"));
        Assert.Throws<ArgumentException>(() => new ClipTokenizer().Encode("\udc00"));
    }

    [Fact]
    public void CancellationIsObservedBeforeAndDuringLongInput()
    {
        var tokenizer = new ClipTokenizer();
        Assert.Throws<OperationCanceledException>(() => tokenizer.Encode("hello", cancellationToken: new CancellationToken(true)));
        using var cancellation = new CancellationTokenSource();
        var input = new string('a', 2_000_000);
        cancellation.CancelAfter(TimeSpan.FromMilliseconds(5));
        Assert.Throws<OperationCanceledException>(() => tokenizer.Encode(input, cancellationToken: cancellation.Token));
    }

    [Theory]
    [InlineData("Clip/vocab.json")]
    [InlineData("Clip/merges.txt")]
    [InlineData("Clip/tokenizer_config.json")]
    [InlineData("Clip/special_tokens_map.json")]
    [InlineData("ClipUnicode/unicode-profile.json")]
    public void CorruptAssetsHaveNamedIntegrityDiagnostics(string damaged)
    {
        var streams = new List<Stream>();
        try
        {
            var exception = Assert.Throws<InvalidDataException>(() => ClipTokenizer.FromResources(name =>
            {
                var stream = typeof(ClipTokenizer).Assembly.GetManifestResourceStream("ComfySharp.Tokenization.Resources." + name.Replace('/', '.'))!;
                streams.Add(stream);
                if (name != damaged) return stream;
                var broken = new MemoryStream(new byte[] { 0 }); streams.Add(broken); return broken;
            }));
            Assert.Contains(damaged, exception.Message);
            Assert.Contains("SHA-256", exception.Message);
        }
        finally { foreach (var stream in streams) stream.Dispose(); }
    }
}
