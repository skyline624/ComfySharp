using ComfySharp.Inference;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

// Fixed behavioral examples; the independent source corpus is tested separately.
[Collection("Classical VAE")]
public sealed class ImageBatchOperationsTests
{
    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void EqualSpatialSizeConcatenatesDistinctBatchCountsWithoutResizing(int channels)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var first = arange(2 * channels, dtype: ScalarType.Float32).reshape(2, 1, 1, channels);
        var second = arange(2 * channels, 3 * channels, dtype: ScalarType.Float32).reshape(1, 1, 1, channels);
        using var actual = ImageOperations.Batch(first, second);
        Assert.Equal(new long[] { 3, 1, 1, channels }, actual.shape);
        Assert.Equal(Enumerable.Range(0, 3 * channels).Select(i => (float)i), Values(actual));
        first.fill_(99); second.fill_(99);
        Assert.Equal(Enumerable.Range(0, 3 * channels).Select(i => (float)i), Values(actual));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ChannelPromotionPadsOnlyRgbAndPreservesExistingAlpha(bool rgbFirst)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var rgb = tensor(new float[] { -1, 0.5f, 2 }).reshape(1, 1, 1, 3);
        var rgba = tensor(new float[] { 3, 4, 5, -0.0f, 6, 7, 8, 0.25f }).reshape(2, 1, 1, 4);
        using var actual = ImageOperations.Batch(rgbFirst ? rgb : rgba, rgbFirst ? rgba : rgb);
        Assert.Equal(new long[] { 3, 1, 1, 4 }, actual.shape);
        float[] promoted = [-1, 0.5f, 2, 1], original = [3, 4, 5, -0.0f, 6, 7, 8, 0.25f];
        Assert.Equal((rgbFirst ? promoted.Concat(original) : original.Concat(promoted)).Select(BitConverter.SingleToInt32Bits),
            Values(actual).Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(new float[] { -1, 0.5f, 2 }, Values(rgb));
        Assert.Equal(original.Select(BitConverter.SingleToInt32Bits), Values(rgba).Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void ResizingRgbaInterpolatesItsExistingAlphaInsteadOfReplacingIt()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var first = tensor(new float[] { 1, 2, 3 }).reshape(1, 1, 1, 3);
        var second = tensor(new float[] { 4, 5, 6, 0, 4, 5, 6, 0.25f,
            4, 5, 6, 0.75f, 4, 5, 6, 1 }).reshape(1, 2, 2, 4);
        using var actual = ImageOperations.Batch(first, second);
        Assert.Equal(new long[] { 2, 1, 1, 4 }, actual.shape);
        Assert.Equal(new float[] { 1, 2, 3, 1, 4, 5, 6, 0.5f }, Values(actual));
    }

    [Fact]
    public void UpsamplingUsesHalfPixelCoordinatesAndKeepsTheFirstBatchUnchanged()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var first = full(new long[] { 2, 1, 4, 3 }, -2.0);
        var second = tensor(new float[] { 0, 0, 0, 8, 8, 8 }).reshape(1, 1, 2, 3);
        using var actual = ImageOperations.Batch(first, second);
        Assert.Equal(new long[] { 3, 1, 4, 3 }, actual.shape);
        float[] expectedLast = [0, 0, 0, 2, 2, 2, 6, 6, 6, 8, 8, 8];
        Assert.Equal(Enumerable.Repeat(-2f, 24).Concat(expectedLast), Values(actual));
        Assert.Equal(new float[] { 0, 0, 0, 8, 8, 8 }, Values(second));
    }

    [Fact]
    public void DownsamplingDoesNotEnableAntialiasFiltering()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        // Same aspect ratio prevents cropping. Last column impulses have different
        // bilinear downsampling weights if antialias is enabled.
        var first = zeros(new long[] { 1, 1, 2, 3 });
        var second = tensor(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8 }).reshape(1, 2, 4, 3);
        using var actual = ImageOperations.Batch(first, second);
        Assert.Equal(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4 }, Values(actual));
    }

    [Theory]
    [InlineData(false, 3)]
    [InlineData(false, 5)]
    [InlineData(true, 3)]
    [InlineData(true, 5)]
    public void CenterCropRoundsHalfMarginsToEvenOnEitherAxis(bool vertical, int sourceLength)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var first = zeros(new long[] { 1, vertical ? 2 : 1, vertical ? 1 : 2, 3 });
        var second = tensor(Enumerable.Range(0, sourceLength).SelectMany(i => new[] { i * 8f, i * 8f, i * 8f }).ToArray())
            .reshape(1, vertical ? sourceLength : 1, vertical ? 1 : sourceLength, 3);
        using var actual = ImageOperations.Batch(first, second);
        // Length3 gives a margin0.5 ->0, then interpolation at .25/1.75.
        // Length5 gives a margin1.5 ->2, leaving the center scalar16.
        float[] tail = sourceLength == 3 ? [2, 2, 2, 14, 14, 14] : [16, 16, 16, 16, 16, 16];
        Assert.Equal(Enumerable.Repeat(0f, 6).Concat(tail), Values(actual));
    }

    [Fact]
    public void NonContiguousInputsSurviveCallerScopesAndProduceAnIndependentNoGradResult()
    {
        NativeRuntimeBootstrap.Initialize();
        long before = Tensor.TotalCount;
        bool grad = is_grad_enabled();
        Tensor result;
        using (NewDisposeScope())
        {
            var first = tensor(Enumerable.Range(0, 6).SelectMany(_ => new[] { -1f, 0.5f, 2f, 0.25f }).ToArray())
                .reshape(1, 2, 3, 4).transpose(1, 2).requires_grad_(true);
            var second = tensor(Enumerable.Range(0, 8).SelectMany(_ => new[] { 7f, 8f, 9f }).ToArray())
                .reshape(2, 2, 2, 3).transpose(1, 2).requires_grad_(true);
            Assert.False(first.is_contiguous()); Assert.False(second.is_contiguous());
            float[] savedFirst = Values(first), savedSecond = Values(second);
            result = ImageOperations.Batch(first, second);
            try
            {
                Assert.Equal(savedFirst, Values(first)); Assert.Equal(savedSecond, Values(second));
                Assert.Equal(grad, is_grad_enabled()); Assert.False(result.requires_grad);
                using var noGrad = no_grad();
                first.fill_(99); second.fill_(99);
            }
            catch { result.Dispose(); throw; }
        }
        using (result)
        {
            Assert.Equal(new long[] { 3, 3, 2, 4 }, result.shape);
            Assert.Equal(Enumerable.Range(0, 6).SelectMany(_ => new[] { -1f, 0.5f, 2f, 0.25f })
                .Concat(Enumerable.Range(0, 12).SelectMany(_ => new[] { 7f, 8f, 9f, 1f })), Values(result));
        }
        Assert.Equal(before, Tensor.TotalCount); Assert.Equal(grad, is_grad_enabled());
    }

    [Theory]
    [InlineData(false, false, 24L)]
    [InlineData(true, false, 48L)]
    [InlineData(false, true, 36L)]
    [InlineData(true, true, 112L)]
    public void BudgetIncludesPaddingResizeAndConcatenation(bool promote, bool resize, long payloadBytes)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var first = zeros(new long[] { 1, 1, 1, promote ? 4 : 3 });
        var second = zeros(new long[] { 1, resize ? 2 : 1, resize ? 2 : 1, 3 });
        long before = Tensor.TotalCount;
        Assert.Throws<InvalidOperationException>(() => { using var result = ImageOperations.Batch(first, second, maxAllocationBytes: payloadBytes - 1); });
        Assert.Equal(before, Tensor.TotalCount);
        using (var admitted = ImageOperations.Batch(first, second, maxAllocationBytes: payloadBytes))
            Assert.Equal(new long[] { 2, 1, 1, promote ? 4 : 3 }, admitted.shape);
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void InvalidSecondInputAndZeroCropErrorsCleanUpAndPreserveGradMode()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var first = zeros(new long[] { 1, 1, 100, 4 });
        var validSecond = zeros(new long[] { 1, 2, 2, 3 });
        var wrongType = zeros(new long[] { 1, 1, 1, 3 }, dtype: ScalarType.Float64);
        var wrongChannels = zeros(new long[] { 1, 1, 1, 5 });
        var wrongRank = zeros(new long[] { 1, 1, 3 });
        long before = Tensor.TotalCount;
        using (no_grad())
        {
            foreach (var invalid in new[] { wrongType, wrongChannels, wrongRank })
                Assert.Throws<ArgumentException>(() => { using var result = ImageOperations.Batch(first, invalid); });
            Assert.Throws<ArgumentNullException>(() => { using var result = ImageOperations.Batch(first, null!); });
            Assert.Throws<ArgumentOutOfRangeException>(() => { using var result = ImageOperations.Batch(first, validSecond, maxAllocationBytes: 0); });
            // A full-channel pad is allocated before the source-compatible crop becomes empty.
            // Check failure/cleanup, not an invented equality of native exception wording.
            var error = Record.Exception(() => { using var result = ImageOperations.Batch(first, validSecond); });
            Assert.NotNull(error);
            Assert.Equal(before, Tensor.TotalCount); Assert.False(is_grad_enabled());
        }
    }

    [Fact]
    public void PreCancellationPrecedesInputValidationAndNativeAccess()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() =>
        {
            using var result = ImageOperations.Batch(null!, null!, cancellation.Token, maxAllocationBytes: 0);
        });
    }

    private static float[] Values(Tensor tensor)
    {
        using var scope = NewDisposeScope();
        return tensor.contiguous().data<float>().ToArray();
    }
}
