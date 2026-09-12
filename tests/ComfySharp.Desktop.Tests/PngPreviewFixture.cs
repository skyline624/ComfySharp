using System.Buffers.Binary;
using System.IO.Compression;
using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;

namespace ComfySharp.Desktop.Tests;

internal static class PngPreviewFixture
{
    // Explicit solid-color PNG fixture, independent of ComfySharp.Inference and its encoder.
    public static byte[] Create(int width = 2, int height = 1, byte red = 255, byte green = 0, byte blue = 0)
    {
        using var result = new MemoryStream(); result.Write(new byte[] { 137,80,78,71,13,10,26,10 });
        void Chunk(ReadOnlySpan<byte> kind, ReadOnlySpan<byte> data)
        {
            Span<byte> word = stackalloc byte[4]; BinaryPrimitives.WriteInt32BigEndian(word, data.Length); result.Write(word); result.Write(kind); result.Write(data);
            uint crc = uint.MaxValue;
            foreach (byte b in kind.ToArray().Concat(data.ToArray())) { crc ^= b; for (int i = 0; i < 8; i++) crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xedb88320 : crc >> 1; }
            BinaryPrimitives.WriteUInt32BigEndian(word, ~crc); result.Write(word);
        }
        var header = new byte[13]; BinaryPrimitives.WriteInt32BigEndian(header, width); BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), height); header[8] = 8; header[9] = 2;
        Chunk("IHDR"u8, header);
        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, CompressionLevel.Fastest, true))
            for (int y = 0; y < height; y++) { zlib.WriteByte(0); for (int x = 0; x < width; x++) { zlib.WriteByte(red); zlib.WriteByte(green); zlib.WriteByte(blue); } }
        Chunk("IDAT"u8, compressed.ToArray()); Chunk("IEND"u8, []); return result.ToArray();
    }

    public static byte[] Pixels(Bitmap bitmap)
    {
        using var copy = new WriteableBitmap(bitmap.PixelSize, new Vector(96, 96), PixelFormat.Rgba8888, AlphaFormat.Unpremul);
        using var frame = copy.Lock(); bitmap.CopyPixels(frame);
        var data = new byte[bitmap.PixelSize.Width * bitmap.PixelSize.Height * 4];
        for (int row = 0; row < bitmap.PixelSize.Height; row++)
            Marshal.Copy(frame.Address + row * frame.RowBytes, data, row * bitmap.PixelSize.Width * 4, bitmap.PixelSize.Width * 4);
        return data;
    }
}
