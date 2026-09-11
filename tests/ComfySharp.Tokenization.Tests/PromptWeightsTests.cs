using ComfySharp.Tokenization;
using Xunit;

namespace ComfySharp.Tokenization.Tests;

public sealed class PromptWeightsTests
{
    [Fact]
    public void ExplicitInnerWeightReplacesOuterWeight()
    {
        Assert.Equal(new[]
        {
            new WeightedPromptSegment("a ", 1), new WeightedPromptSegment("b ", 3), new WeightedPromptSegment("c", 2),
        }, PromptWeights.Parse("a (b (c:2):3)"));
        Assert.Equal(BitConverter.DoubleToInt64Bits(1.2100000000000002),
            BitConverter.DoubleToInt64Bits(Assert.Single(PromptWeights.Parse("((b))")).Weight));
    }

    [Theory]
    [InlineData("(x:nope)", "x:nope", 1.1)]
    [InlineData("(:2)", ":2", 1.1)]
    [InlineData("[a]{b}", "[a]{b}", 1)]
    [InlineData("(x", "(x", 1)]
    [InlineData(")x(", ")x(", 1)]
    [InlineData("(x:1_2.5_0e-1)", "x", 1.25)]
    [InlineData("(x:١٢.٥)", "x", 12.5)]
    [InlineData("(x:𝟙𝟚.𝟝)", "x", 12.5)]
    [InlineData("(x:\U00010d40)", "x:\U00010d40", 1.1)] // Garay digits were assigned after Python's Unicode 15.
    [InlineData("(x:1__0)", "x:1__0", 1.1)]
    [InlineData("(x:1e_2)", "x:1e_2", 1.1)]
    [InlineData("(x:\u001c2)", "x:\u001c2", 1.1)]
    [InlineData("(x:\u00852\u0085)", "x", 2)]
    public void PreservesPythonGrammarAndLiteralCases(string text, string expected, double weight)
    {
        Assert.Equal(new WeightedPromptSegment(expected, weight), Assert.Single(PromptWeights.Parse(text)));
    }

    [Fact]
    public void NegativeNestingIsNotRepaired()
    {
        Assert.Equal(new[] { new WeightedPromptSegment(")(", 1), new WeightedPromptSegment("b", 1.1) },
            PromptWeights.Parse(")((b)"));
        Assert.Empty(PromptWeights.Parse("()"));
        Assert.Empty(PromptWeights.Parse(""));
    }

    [Fact]
    public void EscapesOnlyParenthesesAndRestoresLiteralSentinels()
    {
        Assert.Equal(new WeightedPromptSegment("\\x (a) (b)", 1),
            Assert.Single(PromptWeights.Parse("\\x \\(a\\) \0\u0002b\0\u0001")));
        Assert.Equal(new WeightedPromptSegment("(a:2) (b)", 1),
            Assert.Single(PromptWeights.Parse("(a:2) \\(b\\)", disableWeights: true)));
    }

    [Fact]
    public void PreservesNonfiniteWeightsAndSignedZero()
    {
        Assert.True(double.IsPositiveInfinity(Assert.Single(PromptWeights.Parse("(x:1e9999)")).Weight));
        Assert.True(double.IsNegativeInfinity(Assert.Single(PromptWeights.Parse("(x:-INFINITY)")).Weight));
        Assert.True(double.IsNaN(Assert.Single(PromptWeights.Parse("(x:+NaN)")).Weight));
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(Assert.Single(PromptWeights.Parse("(x:-0)")).Weight));
        Assert.Equal(unchecked((long)0xfff8000000000000UL),
            BitConverter.DoubleToInt64Bits(Assert.Single(PromptWeights.Parse("(x:-nan)")).Weight));
    }

    [Fact]
    public void DeepNestingDoesNotUseTheClrCallStack()
    {
        var parsed = PromptWeights.Parse(new string('(', 2000) + "x" + new string(')', 2000));
        Assert.Equal("x", Assert.Single(parsed).Text);
    }

    [Fact]
    public void HonorsCancellationAndReturnsReadOnlySegments()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => PromptWeights.Parse("x", cancellationToken: cancellation.Token));
        var result = PromptWeights.Parse("x");
        Assert.Throws<NotSupportedException>(() => ((IList<WeightedPromptSegment>)result).Clear());
    }
}
