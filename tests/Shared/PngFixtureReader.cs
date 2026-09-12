using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using ComfySharp.Inference;
using Xunit;

namespace ComfySharp.Testing;

internal static class PngFixtureReader
{
    internal sealed record Decoded(int Width, int Height, int Channels, byte[] Pixels, IReadOnlyList<PngText> Text, IReadOnlyList<string> Chunks);

    // Independent structural reader: checks CRC with the bitwise algorithm and supports all five PNG row filters.
    // Pixel expectations in callers are explicit; this reader never calls the encoder or its quantization routine.
    public static Decoded Read(byte[] png)
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
