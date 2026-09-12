using ComfySharp.Inference;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class ImageOperationsTests
{
    [Theory]
    [InlineData(0x000000, 0f, 0f, 0f)]
    [InlineData(0xFFFFFF, 1f, 1f, 1f)]
    [InlineData(0x017FFE, 0.003921568859368563f, 0.49803921580314636f, 0.9960784316062927f)]
    [InlineData(0xFF8000, 1f, 0.501960813999176f, 0f)]
    public void EmptyUsesRgbOrderAndFillsEveryPixelAndBatch(int color, float red, float green, float blue)
    {
        using var result = ImageOperations.EmptyImage(3, 2, 2, color);
        Assert.Equal(new long[] { 2, 2, 3, 3 }, result.shape);
        Assert.Equal(ScalarType.Float32, result.dtype);
        Assert.Equal(DeviceType.CPU, result.device_type);
        Assert.False(result.requires_grad);
        Assert.Equal(Enumerable.Range(0, 12).SelectMany(_ => new[] { red, green, blue }), Values(result));
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void InvertDoesNotClampAndPreservesAlphaOnStridedImages(int channels)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        float[] pixel = channels == 4 ? [-0.5f, 0.25f, 2f, -0.0f] : [-0.5f, 0.25f, 2f];
        var storage = tensor(Enumerable.Range(0, 6).SelectMany(_ => pixel).ToArray()).reshape(1, 2, 3, channels);
        var image = storage.transpose(1, 2);
        Assert.False(image.is_contiguous());
        float[] before = Values(image);
        using var result = ImageOperations.Invert(image);
        Assert.Equal(image.shape, result.shape);
        float[] expected = channels == 4 ? [1.5f, 0.75f, -1f, -0.0f] : [1.5f, 0.75f, -1f];
        Assert.Equal(Enumerable.Range(0, 6).SelectMany(_ => expected).Select(BitConverter.SingleToInt32Bits),
            Values(result).Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(before, Values(image));
        image.fill_(9);
        Assert.Equal(Enumerable.Range(0, 6).SelectMany(_ => expected), Values(result));
    }

    [Fact]
    public void RepeatPreservesWholeBatchSequenceOnNonContiguousInput()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var image = arange(36, dtype: ScalarType.Float32).reshape(2, 2, 3, 3).transpose(1, 2);
        Assert.False(image.is_contiguous());
        float[] original = [0, 1, 2, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17,
            18, 19, 20, 27, 28, 29, 21, 22, 23, 30, 31, 32, 24, 25, 26, 33, 34, 35];
        using var repeated = ImageOperations.RepeatBatch(image, 3);
        Assert.Equal(new long[] { 6, 3, 2, 3 }, repeated.shape);
        Assert.Equal(original.Concat(original).Concat(original), Values(repeated));
        Assert.Equal(original, Values(image));
    }

    [Fact]
    public void InvertRestoresEachAlphaValueInsteadOfReplacingItWithOpacity()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var image = tensor(new float[] { 0, 0.5f, 1, 0, 0, 0.5f, 1, 0.25f,
            0, 0.5f, 1, 0.75f, 0, 0.5f, 1, 1 }).reshape(1, 2, 2, 4);
        using var result = ImageOperations.Invert(image);
        Assert.Equal(new float[] { 1, 0.5f, 0, 0, 1, 0.5f, 0, 0.25f,
            1, 0.5f, 0, 0.75f, 1, 0.5f, 0, 1 }, Values(result));
    }

    [Theory]
    [InlineData(0L, 100L, 0, 3)]
    [InlineData(1L, 100L, 1, 2)]
    [InlineData(2L, 2L, 2, 1)]
    [InlineData(-1L, 2L, 2, 1)]
    [InlineData(-3L, 1L, 0, 1)]
    [InlineData(-4L, 2L, 0, 2)]
    [InlineData(long.MinValue, 1L, 0, 1)]
    [InlineData(long.MaxValue, long.MaxValue, 2, 1)]
    public void FromBatchAddsOnceThenClampsAndTruncates(long index, long length, int start, int count)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var image = tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }).reshape(3, 1, 1, 3);
        using var result = ImageOperations.FromBatch(image, index, length);
        Assert.Equal(new long[] { count, 1, 1, 3 }, result.shape);
        Assert.Equal(Enumerable.Range(1 + start * 3, count * 3).Select(x => (float)x), Values(result));
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, Values(image));
    }

    [Theory]
    [InlineData("invert")]
    [InlineData("repeat")]
    [InlineData("from")]
    public void OutputsSurviveInputAndCallerScopesWithoutAliasingOrAutograd(string operation)
    {
        NativeRuntimeBootstrap.Initialize();
        long before = Tensor.TotalCount;
        bool grad = is_grad_enabled();
        Tensor result;
        using (NewDisposeScope())
        {
            var image = arange(24, dtype: ScalarType.Float32).reshape(1, 2, 3, 4).transpose(1, 2).requires_grad_(true);
            result = Apply(operation, image);
            Assert.Equal(grad, is_grad_enabled());
            Assert.False(result.requires_grad);
            using var noGrad = no_grad();
            image.fill_(99);
        }
        using (result)
        {
            Assert.Equal(new long[] { 1, 3, 2, 4 }, result.shape);
            float[] logical = [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23];
            if (operation == "invert")
                logical = [1, 0, -1, 3, -11, -12, -13, 15, -3, -4, -5, 7, -15, -16, -17, 19, -7, -8, -9, 11, -19, -20, -21, 23];
            Assert.Equal(logical, Values(result));
        }
        Assert.Equal(before, Tensor.TotalCount);
        Assert.Equal(grad, is_grad_enabled());
    }

    [Fact]
    public void EmptyOutlivesCallerScopeAndNoGradModeIsRestored()
    {
        NativeRuntimeBootstrap.Initialize();
        long before = Tensor.TotalCount;
        using (no_grad())
        {
            Tensor result;
            using (NewDisposeScope()) result = ImageOperations.EmptyImage(1, 1, color: 0xFF0000);
            using (result) Assert.Equal(new float[] { 1, 0, 0 }, Values(result));
            Assert.False(is_grad_enabled());
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory]
    [InlineData("invert")]
    [InlineData("repeat")]
    [InlineData("from")]
    public void InvalidImageProfilesAndBudgetsDoNotLeakOrChangeGradMode(string operation)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        Tensor[] invalid = [zeros(new long[] { 1, 2, 3 }), zeros(new long[] { 0, 1, 1, 3 }),
            zeros(new long[] { 1, 0, 1, 3 }), zeros(new long[] { 1, 1, 0, 3 }),
            zeros(new long[] { 1, 1, 1, 1 }), zeros(new long[] { 1, 1, 1, 5 }),
            zeros(new long[] { 1, 1, 1, 3 }, dtype: ScalarType.Float64)];
        var valid = zeros(new long[] { 1, 1, 1, 3 });
        long before = Tensor.TotalCount;
        bool grad = is_grad_enabled();
        foreach (var image in invalid) Assert.Throws<ArgumentException>(() => { using var result = Apply(operation, image); });
        Assert.Throws<ArgumentNullException>(() => { using var result = Apply(operation, null!); });
        Assert.Throws<InvalidOperationException>(() => { using var result = Apply(operation, valid, budget: 11); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { using var result = Apply(operation, valid, budget: 0); });
        using (var admitted = Apply(operation, valid, budget: 12)) Assert.Equal(new long[] { 1, 1, 1, 3 }, admitted.shape);
        Assert.Equal(before, Tensor.TotalCount);
        Assert.Equal(grad, is_grad_enabled());
    }

    [Fact]
    public void AllocationBoundsRejectBeforeMaterializationAndUseTruncatedOutputSize()
    {
        Assert.Throws<InvalidOperationException>(() => { using var result = ImageOperations.EmptyImage(1, 1, maxAllocationBytes: 23); });
        Assert.Throws<OverflowException>(() => { using var result = ImageOperations.EmptyImage(long.MaxValue, 2); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { using var result = ImageOperations.EmptyImage(0, 1); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { using var result = ImageOperations.EmptyImage(1, -1); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { using var result = ImageOperations.EmptyImage(1, 1, 0); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { using var result = ImageOperations.EmptyImage(1, 1, color: -1); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { using var result = ImageOperations.EmptyImage(1, 1, color: 0x1000000); });
        using var empty = ImageOperations.EmptyImage(1, 1, maxAllocationBytes: 24);
        using var last = ImageOperations.FromBatch(empty, long.MaxValue, long.MaxValue, maxAllocationBytes: 12);
        Assert.Throws<OverflowException>(() => { using var result = ImageOperations.RepeatBatch(empty, long.MaxValue); });
        Assert.Throws<InvalidOperationException>(() => { using var result = ImageOperations.RepeatBatch(empty, 2, maxAllocationBytes: 23); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { using var result = ImageOperations.RepeatBatch(empty, 0); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { using var result = ImageOperations.FromBatch(empty, 0, 0); });
    }

    [Theory]
    [InlineData("empty")]
    [InlineData("invert")]
    [InlineData("repeat")]
    [InlineData("from")]
    public void CancellationPrecedesInputValidationAndAllocation(string operation)
    {
        using var canceled = new CancellationTokenSource();
        canceled.Cancel();
        Assert.Throws<OperationCanceledException>(() =>
        {
            using var result = operation == "empty"
                ? ImageOperations.EmptyImage(0, 0, cancellationToken: canceled.Token)
                : Apply(operation, null!, canceled.Token);
        });
    }

    private static Tensor Apply(string operation, Tensor image, CancellationToken token = default,
        long budget = ImageOperations.DefaultMaxAllocationBytes) => operation switch
    {
        "invert" => ImageOperations.Invert(image, token, budget),
        "repeat" => ImageOperations.RepeatBatch(image, 1, token, budget),
        "from" => ImageOperations.FromBatch(image, 0, long.MaxValue, token, budget),
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static float[] Values(Tensor tensor)
    {
        using var scope = NewDisposeScope();
        // Read logical values, never a flat span over a strided view's storage.
        return tensor.contiguous().data<float>().ToArray();
    }
}
