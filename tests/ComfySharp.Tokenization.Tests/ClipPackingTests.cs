using ComfySharp.Tokenization;
using Xunit;

namespace ComfySharp.Tokenization.Tests;

public sealed class ClipPackingTests
{
    [Theory]
    [InlineData(ClipProfile.Sd1L, 49407)]
    [InlineData(ClipProfile.SdXlL, 49407)]
    [InlineData(ClipProfile.SdXlG, 0)]
    public void EmptyInputRetainsOnePaddedSequence(ClipProfile profile, int pad)
    {
        var result = new ComfyClipTokenizer(new WordsTokenizer(), profile).Tokenize("");
        var chunk = Assert.Single(result.Chunks);
        Assert.Equal(77, chunk.Count);
        Assert.Equal(new ClipToken(49406, 1, 0), chunk[0]);
        Assert.Equal(new ClipToken(49407, 1, 0), chunk[1]);
        Assert.All(chunk.Skip(2), token => Assert.Equal(new ClipToken(pad, 1, 0), token));
    }

    [Theory]
    [InlineData(7, 0)]
    [InlineData(8, 1)]
    public void SmallSegmentMovesWholeAndLargeSegmentSplits(int groupSize, int inFirstChunk)
    {
        var prompt = string.Concat(Enumerable.Repeat("a ", 74)) + "(" + string.Join(" ", Enumerable.Repeat("b", groupSize)) + ")";
        var chunks = new ComfyClipTokenizer(new WordsTokenizer()).Tokenize(prompt).Chunks;
        Assert.Equal(2, chunks.Count);
        Assert.All(chunks, chunk => Assert.Equal(77, chunk.Count));
        Assert.Equal(inFirstChunk, chunks[0].Count(t => t.WordId == 2));
        Assert.Equal(groupSize - inFirstChunk, chunks[1].Count(t => t.WordId == 2));
        Assert.All(chunks.SelectMany(c => c).Where(t => t.WordId == 2), token => Assert.Equal(1.1, token.Weight));
    }

    [Theory]
    [InlineData(74, 1)]
    [InlineData(75, 1)]
    [InlineData(76, 2)]
    [InlineData(9000, 120)]
    public void ContentBoundariesAndLongTextAreNotTruncated(int words, int chunkCount)
    {
        var chunks = new ComfyClipTokenizer(new WordsTokenizer()).Tokenize(string.Join(" ", Enumerable.Repeat("a", words))).Chunks;
        Assert.Equal(chunkCount, chunks.Count);
        Assert.Equal(words, chunks.SelectMany(c => c).Count(t => t.WordId == 1));
    }

    [Fact]
    public void EmptyEncodedUnitsLeaveHolesInWordIds()
    {
        var result = new ComfyClipTokenizer(new WordsTokenizer()).Tokenize(" (a) (b)", new() { ReturnWordIds = true });
        Assert.True(result.ReturnWordIds);
        Assert.Equal(new[] { 2, 4 }, result.Chunks.SelectMany(c => c).Where(t => t.WordId != 0).Select(t => t.WordId));
    }

    [Fact]
    public void EmbeddingMarkersAreOrdinaryTextButStillSplitUnits()
    {
        var stub = new WordsTokenizer();
        var result = new ComfyClipTokenizer(stub).Tokenize("a embedding:first xembedding:joined\u001cembedding:next");
        Assert.Equal(new[] { "a ", "embedding:first xembedding:joined\u001c", "embedding:next" }, stub.Inputs);
        Assert.Equal(new[] { 1, 2, 2, 3 }, result.Chunks[0].Where(t => t.WordId != 0).Select(t => t.WordId));
        Assert.Throws<NotSupportedException>(() => new ComfyClipTokenizer(stub, options: new() { ResolveTextualInversions = true }));
    }

    [Fact]
    public void FinalPaddingMayExceedNominalLengthAndCanPrecedeBos()
    {
        var result = new ComfyClipTokenizer(new WordsTokenizer(), options: new() { MinPadding = 77, PadLeft = true, PadToken = 9 }).Tokenize("");
        var chunk = Assert.Single(result.Chunks);
        Assert.Equal(79, chunk.Count);
        Assert.All(chunk.Take(77), token => Assert.Equal(new ClipToken(9, 1, 0), token));
        Assert.Equal(49406, chunk[77].Id);
        Assert.Equal(49407, chunk[78].Id);
    }

    [Fact]
    public void OptionsFollowDataThenCallThenKeywordPriorityIncludingNull()
    {
        var data = new Dictionary<string, int?> { ["clip_l_max_length"] = 12, ["clip_l_min_length"] = 15 };
        var tokenizer = new ComfyClipTokenizer(new WordsTokenizer(), options: new()
        {
            MaxLength = 77, MinLength = 20, MinPadding = 2, TokenizerData = data,
        });
        data["clip_l_max_length"] = 999;
        Assert.Equal(15, tokenizer.Tokenize("").Chunks[0].Count);
        var callOptions = new Dictionary<string, int?> { ["clip_l_min_length"] = 18, ["clip_l_min_padding"] = 20 };
        Assert.Equal(22, tokenizer.Tokenize("", new() { TokenizerOptions = callOptions, MinLength = 16 }).Chunks[0].Count);
        callOptions["clip_l_min_padding"] = null;
        Assert.Equal(16, tokenizer.Tokenize("", new() { TokenizerOptions = callOptions, MinLength = 16 }).Chunks[0].Count);
        Assert.Equal(12, tokenizer.Tokenize("", new() { TokenizerOptions = callOptions, OverrideMinLength = true }).Chunks[0].Count);
    }

    [Fact]
    public void PaddingDisabledAndNegativeMinimumPreserveSourceBehavior()
    {
        var tokenizer = new ComfyClipTokenizer(new WordsTokenizer(), options: new() { PadToMaxLength = false, MinPadding = -4, MinLength = -1 });
        Assert.Equal(2, tokenizer.Tokenize("").Chunks[0].Count);
        Assert.Equal(3, tokenizer.Tokenize("x").Chunks[0].Count);
    }

    [Fact]
    public void PerCallWeightsCanReenableConstructorDisabledWeights()
    {
        var tokenizer = new ComfyClipTokenizer(new WordsTokenizer(), options: new() { DisableWeights = true });
        Assert.Equal(1, tokenizer.Tokenize("(x:2)").Chunks[0][1].Weight);
        Assert.Equal(2, tokenizer.Tokenize("(x:2)", new() { DisableWeights = false }).Chunks[0][1].Weight);
    }

    [Fact]
    public void SdxlCompositionHasIndependentLengthsAndGThenLOrder()
    {
        var tokenizer = new ComfySdxlTokenizer(new WordsTokenizer(), new() { MaxLength = 10 }, new() { MaxLength = 12 });
        var result = tokenizer.Tokenize("x");
        Assert.Equal(new[] { "g", "l" }, result.Keys);
        Assert.Equal(12, result["g"].Chunks[0].Count);
        Assert.Equal(10, result["l"].Chunks[0].Count);
        Assert.Equal(0, result["g"].Chunks[0][^1].Id);
        Assert.Equal(49407, result["l"].Chunks[0][^1].Id);
    }

    [Fact]
    public void ImpossibleCapacitiesFailInsteadOfLooping()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ComfyClipTokenizer(new WordsTokenizer(), options: new() { MaxLength = 2 }));
        var tokenizer = new ComfyClipTokenizer(new WordsTokenizer(), options: new() { MaxLength = 3 });
        Assert.Throws<ArgumentException>(() => tokenizer.Tokenize("a b"));
        Assert.Equal(8, tokenizer.Tokenize("a b c d e f g h").Chunks.Count);
    }

    [Fact]
    public void ResultsAndPairProjectionAreImmutableAndRetainWordIds()
    {
        var result = new ComfyClipTokenizer(new WordsTokenizer()).Tokenize("x");
        Assert.False(result.ReturnWordIds);
        Assert.Equal(1, result.Chunks[0][1].WordId);
        Assert.Equal(new ClipTokenWeight(100, 1), result.WithoutWordIds()[0][1]);
        Assert.Throws<NotSupportedException>(() => ((IList<ClipToken>)result.Chunks[0]).Clear());
        Assert.Throws<NotSupportedException>(() => ((IList<IReadOnlyList<ClipToken>>)result.Chunks).Clear());
    }

    [Fact]
    public void CancellationIsForwardedAndHonoredAfterDependencyReturns()
    {
        using var source = new CancellationTokenSource();
        var stub = new WordsTokenizer { CancelAfterEncoding = source };
        Assert.Throws<OperationCanceledException>(() => new ComfyClipTokenizer(stub).Tokenize("x", cancellationToken: source.Token));
        Assert.Equal(source.Token, stub.LastCancellation);
        Assert.Throws<OperationCanceledException>(() => new ComfyClipTokenizer(stub).Tokenize("", cancellationToken: source.Token));
    }

    private sealed class WordsTokenizer : IClipTokenizer
    {
        public int BosTokenId => 49406;
        public int EosTokenId => 49407;
        public List<string> Inputs { get; } = [];
        public CancellationToken LastCancellation { get; private set; }
        public CancellationTokenSource? CancelAfterEncoding { get; init; }
        public IReadOnlyList<int> Encode(string text, bool addSpecialTokens = true, CancellationToken cancellationToken = default)
        {
            Inputs.Add(text);
            LastCancellation = cancellationToken;
            var words = text.Replace('\u001c', ' ').Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            var ids = Enumerable.Repeat(100, words.Length);
            CancelAfterEncoding?.Cancel();
            return addSpecialTokens ? new[] { BosTokenId }.Concat(ids).Append(EosTokenId).ToArray() : ids.ToArray();
        }
        public string Decode(IEnumerable<int> ids, bool skipSpecialTokens = true) => string.Join(",", ids);
        public string GetToken(int id) => id.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }
}
