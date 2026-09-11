using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class SigmaOperationTests
{
    public SigmaOperationTests() => NativeRuntimeBootstrap.Initialize();

    [Theory]
    [InlineData(0, 1, 4)]
    [InlineData(2, 3, 2)]
    [InlineData(3, 4, 1)]
    [InlineData(10000, 4, 0)]
    public void SplitSaturatesBoundsAndViewsSurviveTheirParent(int step, int highCount, int lowCount)
    {
        var source = tensor(new float[] { 9, 6, 3, 0 });
        var (high, low) = SigmaOperations.Split(source, step);
        using (high) using (low)
        {
            source.Dispose();
            Assert.Equal(new float[] { 9, 6, 3, 0 }.Take(highCount), high.data<float>().ToArray());
            Assert.Equal(new float[] { 9, 6, 3, 0 }.TakeLast(lowCount), low.data<float>().ToArray());
        }
    }

    [Theory]
    [InlineData(0, 0, 1)]
    [InlineData(.1, 0, 1)] // 5 * .1 = .5 rounds to even zero.
    [InlineData(.5, 4, 3)] // 5 * .5 = 2.5 rounds to two.
    [InlineData(.7, 2, 5)] // 5 * .7 = 3.5 rounds to four.
    [InlineData(1, 1, 6)]
    public void SplitDenoiseUsesPythonRoundAndNegativeZeroSlice(double denoise, int highCount, int lowCount)
    {
        using var source = tensor(new float[] { 5, 4, 3, 2, 1, 0 });
        var (high, low) = SigmaOperations.SplitDenoise(source, denoise);
        using (high) using (low)
        {
            Assert.Equal(highCount, high.numel()); Assert.Equal(lowCount, low.numel());
            Assert.Equal(source.data<float>().ToArray().Take(highCount), high.data<float>().ToArray());
            Assert.Equal(source.data<float>().ToArray().TakeLast(lowCount), low.data<float>().ToArray());
        }
    }

    [Fact]
    public void EmptyAndSingletonSplitsDoNotInventValues()
    {
        foreach (float[] values in new[] { Array.Empty<float>(), new float[] { 9 } })
        {
            using var source = tensor(values);
            var (high, low) = SigmaOperations.SplitDenoise(source, 1);
            using (high) using (low) { Assert.Equal(0, high.numel()); Assert.Equal(values, low.data<float>().ToArray()); }
        }
    }

    [Fact]
    public void FlipAndSetFirstPreserveSharedInputsAndDtype()
    {
        using var source = tensor(new double[] { 9, 3, 0 });
        using var flip = SigmaOperations.Flip(source);
        using var first = SigmaOperations.SetFirst(source, 100);
        Assert.Equal(new double[] { 9, 3, 0 }, source.data<double>().ToArray());
        Assert.Equal(new double[] { .0001, 3, 9 }, flip.data<double>().ToArray());
        Assert.Equal(new double[] { 100, 3, 0 }, first.data<double>().ToArray());
        Assert.Equal(ScalarType.Float64, flip.dtype); Assert.Equal(ScalarType.Float64, first.dtype);
    }

    [Theory]
    [InlineData("linear", 6f)]
    [InlineData("cosine", 4.343146f)]
    [InlineData("sine", 7.656854f)]
    public void ExtendPreservesSourceSpacingNamesAndSelectsCurrentSigma(string spacing, float midpoint)
    {
        using var source = tensor(new double[] { 10, 2, 0 });
        using var result = SigmaOperations.Extend(source, 2, -1, 5, spacing);
        var values = result.data<float>().ToArray();
        Assert.Equal(ScalarType.Float32, result.dtype); Assert.Equal(TorchSharp.DeviceType.CPU, result.device.type);
        Assert.Equal(4, values.Length); Assert.Equal(10f, values[0]); Assert.InRange(Math.Abs(values[1] - midpoint), 0, 1e-6f);
        Assert.Equal(new float[] { 2, 0 }, values[2..]); Assert.Equal(new double[] { 10, 2, 0 }, source.data<double>().ToArray());
    }

    [Fact]
    public void ExtendOneSubdivisionAndEmptyInputPreserveLength()
    {
        using var source = tensor(new float[] { 9, 0 });
        using var result = SigmaOperations.Extend(source, 1, -1, 0, "linear");
        Assert.Equal(new float[] { 9, 0 }, result.data<float>().ToArray());
        using var empty = tensor(Array.Empty<double>());
        using var emptyResult = SigmaOperations.Extend(empty, 3, -1, 0, "sine");
        Assert.Equal(0, emptyResult.numel()); Assert.Equal(ScalarType.Float32, emptyResult.dtype);
    }

    [Theory]
    [InlineData("abc", new float[0])]
    [InlineData("1e-3, .5 +2 -0 ٣.٥", new float[] { 1, -3, .5f, 2, 0, 3.5f })]
    [InlineData("1, .5, 0", new float[] { 1, .5f, 0 })]
    [InlineData("𝟙.𝟝 ٠.٥", new float[] { 1.5f, .5f })]
    public void ManualUsesSourceRegexInsteadOfScientificNotationParser(string text, float[] expected)
    {
        using var result = SigmaOperations.Manual(text);
        Assert.Equal(expected, result.data<float>().ToArray());
    }

    [Fact]
    public void InvalidRegexCaptureAndRankAndEmptyIndexFailExplicitly()
    {
        Assert.Throws<FormatException>(() => SigmaOperations.Manual("1..2"));
        using var empty = tensor(Array.Empty<float>());
        Assert.Throws<ArgumentException>(() => SigmaOperations.SetFirst(empty, 1));
        using var matrix = ones(2, 2);
        Assert.Throws<ArgumentException>(() => SigmaOperations.Split(matrix, 0));
        using var source = tensor(new float[] { 1, 0 });
        Assert.Throws<ArgumentException>(() => SigmaOperations.Extend(source, 2, -1, 0, "unknown"));
    }

    [Fact]
    public void CancelledOperationsLeaveBorrowedInputAlive()
    {
        using var source = tensor(new float[] { 1, 0 });
        var cancelled = new CancellationToken(true);
        Assert.Throws<OperationCanceledException>(() => SigmaOperations.Split(source, 0, cancelled));
        Assert.Throws<OperationCanceledException>(() => SigmaOperations.SplitDenoise(source, .5, cancelled));
        Assert.Throws<OperationCanceledException>(() => SigmaOperations.Flip(source, cancelled));
        Assert.Throws<OperationCanceledException>(() => SigmaOperations.SetFirst(source, 2, cancelled));
        Assert.Throws<OperationCanceledException>(() => SigmaOperations.Extend(source, 2, -1, 0, "linear", cancelled));
        Assert.Throws<OperationCanceledException>(() => SigmaOperations.Manual("1", cancelled));
        Assert.Equal(new float[] { 1, 0 }, source.data<float>().ToArray());
    }
}
