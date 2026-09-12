using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
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

    private sealed record Decoded(int Width, int Height, int Channels, byte[] Pixels, IReadOnlyList<PngText> Text, IReadOnlyList<string> Chunks);

    // Independent structural reader: checks CRC with the bitwise algorithm and supports all five PNG row filters.
    // Pixel expectations above are explicit; this reader never calls the encoder or its quantization routine.
    private static Decoded Read(byte[] png)
    {
        Assert.Equal(new byte[] { 137,80,78,71,13,10,26,10 }, png[..8]);
        int offset = 8, width = 0, height = 0, channels = 0;
        var chunks = new List<string>(); var text = new List<PngText>();
        using var compressed = new MemoryStream();
        while (offset < png.Length)
        {
            int size = BinaryPrimitives.ReadInt32BigEndian(png.AsSpan(offset));
            Assert.InRange(size, 0, png.Length - offset - 12);
            string kind = Encoding.ASCII.GetString(png, offset + 4, 4);
            ReadOnlySpan<byte> data = png.AsSpan(offset + 8, size);
            uint crc = uint.MaxValue;
            foreach (byte value in png.AsSpan(offset + 4, size + 4))
            {
                crc ^= value;
                for (int i = 0; i < 8; i++) crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xedb88320 : crc >> 1;
            }
            Assert.Equal(~crc, BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(offset + size + 8)));
            chunks.Add(kind);
            switch (kind)
            {
                case "IHDR":
                    Assert.Single(chunks); Assert.Equal(13, size);
                    width = BinaryPrimitives.ReadInt32BigEndian(data); height = BinaryPrimitives.ReadInt32BigEndian(data[4..]);
                    Assert.Equal(8, data[8]); Assert.Contains(data[9], new byte[] { 2, 6 }); channels = data[9] == 2 ? 3 : 4;
                    Assert.Equal(new byte[] { 0, 0, 0 }, data[10..].ToArray()); break;
                case "IDAT": compressed.Write(data); break;
                case "tEXt":
                case "iTXt":
                    int separator = data.IndexOf((byte)0); Assert.True(separator > 0);
                    string keyword = Encoding.Latin1.GetString(data[..separator]);
                    if (kind == "tEXt") text.Add(new(keyword, Encoding.Latin1.GetString(data[(separator + 1)..])));
                    else
                    {
                        Assert.Equal(new byte[5], data.Slice(separator, 5).ToArray());
                        text.Add(new(keyword, new UTF8Encoding(false, true).GetString(data[(separator + 5)..])));
                    }
                    break;
                case "IEND": Assert.Equal(0, size); Assert.Equal(png.Length, offset + 12); break;
                default: throw new InvalidDataException("Unexpected PNG chunk " + kind);
            }
            offset += size + 12;
        }
        Assert.Equal("IEND", chunks[^1]);
        compressed.Position = 0; using var zlib = new ZLibStream(compressed, CompressionMode.Decompress);
        int rowBytes = checked(width * channels); byte[] pixels = new byte[checked(height * rowBytes)];
        for (int y = 0; y < height; y++)
        {
            int filter = zlib.ReadByte(); Assert.InRange(filter, 0, 4);
            Span<byte> row = pixels.AsSpan(y * rowBytes, rowBytes); zlib.ReadExactly(row);
            for (int x = 0; x < rowBytes; x++)
            {
                int left = x >= channels ? row[x - channels] : 0;
                int up = y > 0 ? pixels[(y - 1) * rowBytes + x] : 0;
                int corner = y > 0 && x >= channels ? pixels[(y - 1) * rowBytes + x - channels] : 0;
                int predictor = filter switch { 0 => 0, 1 => left, 2 => up, 3 => (left + up) / 2, _ => Paeth(left, up, corner) };
                row[x] = unchecked((byte)(row[x] + predictor));
            }
        }
        Assert.Equal(-1, zlib.ReadByte());
        return new(width, height, channels, pixels, text, chunks);
    }
    private static int Paeth(int left, int up, int corner)
    {
        int p = left + up - corner, a = Math.Abs(p - left), b = Math.Abs(p - up), c = Math.Abs(p - corner);
        return a <= b && a <= c ? left : b <= c ? up : corner;
    }
}
