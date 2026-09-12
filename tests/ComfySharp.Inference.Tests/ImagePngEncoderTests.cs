using static ComfySharp.Testing.PngFixtureReader;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class ImagePngEncoderTests
{
    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void DecodedPixelsClipAndTruncateWithoutPremultiplyingAlpha(int channels)
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        float[] source = channels == 3 ? [-1, .5f, 2, float.MaxValue, float.MinValue, -0.0f]
            : [1, .5f, 0, .5f, 1, 0, 0, 0];
        byte[] expected = channels == 3 ? [0, 127, 255, 255, 0, 0] : [255, 127, 0, 127, 255, 0, 0, 0];
        var image = tensor(source).reshape(1, 1, 2, channels);
        bool grad = is_grad_enabled(); long before = Tensor.TotalCount;
        var decoded = Read(ImagePngEncoder.EncodeFrame(image, 0));
        Assert.Equal(2, decoded.Width); Assert.Equal(1, decoded.Height); Assert.Equal(channels, decoded.Channels);
        Assert.Equal(expected, decoded.Pixels); Assert.Empty(decoded.Text);
        Assert.Equal(before, Tensor.TotalCount); Assert.Equal(grad, is_grad_enabled());
        Assert.Equal(source.Select(BitConverter.SingleToInt32Bits), image.data<float>().ToArray().Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void SelectedStridedFrameUsesLogicalCoordinatesAndDoesNotMutateItsStorage()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        var storage = tensor(new float[] { 0,0,0, 0,0,0, 0,0,0, 0,0,0,
            1,0,0, 0,1,0, 0,0,1, .5f,.5f,.5f }).reshape(2, 2, 2, 3);
        var image = storage.transpose(1, 2);
        Assert.False(image.is_contiguous());
        long before = Tensor.TotalCount;
        byte[] png = ImagePngEncoder.EncodeFrame(image, 1);
        Assert.Equal(new byte[] { 255,0,0, 0,0,255, 0,255,0, 127,127,127 }, Read(png).Pixels);
        using (var observed = storage.select(0, 1))
            Assert.Equal(new float[] { 1,0,0, 0,1,0, 0,0,1, .5f,.5f,.5f }, observed.data<float>().ToArray());
        storage.fill_(99);
        Assert.Equal(new byte[] { 255,0,0, 0,0,255, 0,255,0, 127,127,127 }, Read(png).Pixels);
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void OrderedDuplicateMetadataUsesLatin1AndInternationalTextWithoutLosingValues()
    {
        NativeRuntimeBootstrap.Initialize(); using var image = ImageOperations.EmptyImage(1, 1);
        PngText[] text = [new("prompt", "{\"x\": 1}"), new("note", "café"), new("prompt", "猫 😀")];
        var decoded = Read(ImagePngEncoder.EncodeFrame(image, 0, text));
        Assert.Equal(text, decoded.Text);
        Assert.Equal(new[] { "IHDR", "tEXt", "tEXt", "iTXt", "IDAT", "IEND" }, decoded.Chunks);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void SourceCompressionLevelsDecodeToTheSameKnownPixels(int level)
    {
        NativeRuntimeBootstrap.Initialize(); using var image = ImageOperations.EmptyImage(8, 4, color: 0xff00ff);
        var decoded = Read(ImagePngEncoder.EncodeFrame(image, 0, compressionLevel: level));
        Assert.Equal(Enumerable.Range(0, 32).SelectMany(_ => new byte[] { 255, 0, 255 }), decoded.Pixels);
    }

    [Fact]
    public void ConsecutiveIdatChunksFormOneValidZlibStream()
    {
        NativeRuntimeBootstrap.Initialize(); using var image = ImageOperations.EmptyImage(200, 200, color: 0x00ff00);
        var decoded = Read(ImagePngEncoder.EncodeFrame(image, 0, compressionLevel: 0));
        Assert.True(decoded.Chunks.Count(c => c == "IDAT") >= 2);
        Assert.Equal(Enumerable.Range(0, 40000).SelectMany(_ => new byte[] { 0, 255, 0 }), decoded.Pixels);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void NonFiniteValuesFailWithoutLeavingNativeWrappers(float invalid)
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        var image = tensor(new float[] { 0, 1, invalid }).reshape(1, 1, 1, 3);
        long before = Tensor.TotalCount; bool grad = is_grad_enabled();
        Assert.Throws<ArgumentException>(() => ImagePngEncoder.EncodeFrame(image, 0));
        Assert.Equal(before, Tensor.TotalCount); Assert.Equal(grad, is_grad_enabled());
    }

    [Fact]
    public void GradAndInvalidLayoutsAreRejectedExplicitly()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        var grad = zeros(new long[] { 1, 1, 1, 3 }, requires_grad: true);
        var wrongType = zeros(new long[] { 1, 1, 1, 3 }, dtype: ScalarType.Float64);
        var wrongChannels = zeros(new long[] { 1, 1, 1, 2 });
        foreach (var image in new[] { grad, wrongType, wrongChannels })
            Assert.Throws<ArgumentException>(() => ImagePngEncoder.EncodeFrame(image, 0));
        using var valid = ImageOperations.EmptyImage(1, 1);
        Assert.Throws<ArgumentOutOfRangeException>(() => ImagePngEncoder.EncodeFrame(valid, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImagePngEncoder.EncodeFrame(valid, 1));
    }

    [Fact]
    public void PixelAndEncodedLimitsFailWithoutLargeAllocationOrNativeLeaks()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        var scalar = zeros(new long[] { 1, 1, 1, 3 });
        var hugeView = scalar.expand(1, 100000, 100000, 3);
        long before = Tensor.TotalCount;
        Assert.Throws<InvalidOperationException>(() => ImagePngEncoder.EncodeFrame(hugeView, 0));
        Assert.Throws<InvalidOperationException>(() => ImagePngEncoder.EncodeFrame(scalar, 0, maxEncodedBytes: 40));
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory]
    [InlineData("")]
    [InlineData(" leading")]
    [InlineData("trailing ")]
    [InlineData("two  spaces")]
    [InlineData("猫")]
    [InlineData("bad\0key")]
    public void InvalidPngKeywordsAreNotSilentlySubstituted(string keyword)
    {
        Assert.Throws<ArgumentException>(() => ImagePngEncoder.EncodeFrame(null!, 0, [new(keyword, "text")]));
    }

    [Fact]
    public void MetadataAndCancellationAreBoundedBeforeNativeAccess()
    {
        Assert.Throws<InvalidOperationException>(() => ImagePngEncoder.EncodeFrame(null!, 0, [new("note", new string('x', ImagePngEncoder.MaxMetadataBytes))]));
        Assert.Throws<OperationCanceledException>(() => ImagePngEncoder.EncodeFrame(null!, -1,
            cancellationToken: new CancellationToken(true), maxPixels: 0));
    }

}
