using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using ComfySharp.Contracts;

namespace ComfySharp.Media;

// Skia's general codec output cannot preserve all 16-bit PNG grayscale/RGB samples.
// Decode the PNG sample representation directly, including Adam7, without a color-space transform.
internal static class Png16Decoder
{
    internal static (int Frames, int? Orientation) ReadMetadata(Stream stream, CancellationToken token)
    {
        stream.Seek(8, SeekOrigin.Current); var header = new byte[8]; int frames = 1; int? orientation = null;
        while (stream.Position < stream.Length)
        {
            token.ThrowIfCancellationRequested();
            stream.ReadExactly(header); uint length = BinaryPrimitives.ReadUInt32BigEndian(header);
            if (length > int.MaxValue || length + 4L > stream.Length - stream.Position) throw new InvalidDataException("Invalid PNG chunk size.");
            if (header.AsSpan(4).SequenceEqual("acTL"u8))
            {
                if (length != 8) throw new InvalidDataException("Invalid APNG control chunk.");
                var count = new byte[4]; stream.ReadExactly(count); frames = checked((int)BinaryPrimitives.ReadUInt32BigEndian(count));
                if (frames <= 0) throw new InvalidDataException("Invalid APNG frame count.");
                stream.Seek(8, SeekOrigin.Current); continue;
            }
            if (header.AsSpan(4).SequenceEqual("eXIf"u8))
            {
                if (length > 1024 * 1024) throw new InvalidDataException("PNG EXIF exceeds the metadata budget.");
                var exif = new byte[(int)length]; stream.ReadExactly(exif); orientation = ExifOrientation(exif);
                stream.Seek(4, SeekOrigin.Current); continue;
            }
            if (header.AsSpan(4).SequenceEqual("IEND"u8)) return (frames, orientation);
            stream.Seek(length + 4L, SeekOrigin.Current);
        }
        throw new InvalidDataException("PNG image data is missing.");
    }

    private static int? ExifOrientation(byte[] data)
    {
        if (data.Length < 8) throw new InvalidDataException("Truncated PNG EXIF.");
        bool little = data.AsSpan(0,2).SequenceEqual("II"u8);
        if (!little && !data.AsSpan(0,2).SequenceEqual("MM"u8)) throw new InvalidDataException("Unknown PNG EXIF byte order.");
        ushort Short(int offset) => little ? BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(offset,2)) : BinaryPrimitives.ReadUInt16BigEndian(data.AsSpan(offset,2));
        uint Long(int offset) => little ? BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset,4)) : BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(offset,4));
        if (Short(2) != 42) throw new InvalidDataException("Invalid PNG TIFF header.");
        uint start = Long(4);
        if (start < 8 || start > data.Length - 2) throw new InvalidDataException("Invalid PNG EXIF directory.");
        int count = Short((int)start), entries = (int)start + 2;
        if ((long)entries + count * 12L > data.Length) throw new InvalidDataException("Truncated PNG EXIF entries.");
        for (int i = 0; i < count; i++)
        {
            int offset = entries + i * 12;
            if (Short(offset) == 0x112 && Short(offset + 2) == 3 && Long(offset + 4) == 1)
            {
                int value = Short(offset + 8);
                if (value is < 1 or > 8) throw new InvalidDataException("Invalid PNG EXIF orientation.");
                return value;
            }
        }
        return null;
    }

    internal static DecodedImageBatch Decode(Stream stream, int orientation, CancellationToken token)
    {
        stream.Seek(8, SeekOrigin.Current); int width = 0, height = 0, channels = 0, color = -1, interlace = -1;
        byte[]? transparent = null; bool ended = false, sawData = false;
        using var compressed = new MemoryStream(); var header = new byte[8];
        while (!ended)
        {
            token.ThrowIfCancellationRequested(); stream.ReadExactly(header);
            uint length = BinaryPrimitives.ReadUInt32BigEndian(header);
            if (length > 64 * 1024 * 1024 || length + 4L > stream.Length - stream.Position) throw new InvalidDataException("Invalid PNG chunk size.");
            string kind = Encoding.ASCII.GetString(header, 4, 4); var data = new byte[(int)length]; stream.ReadExactly(data);
            var checksum = new byte[4]; stream.ReadExactly(checksum);
            uint crc = uint.MaxValue;
            foreach (byte b in header.AsSpan(4)) crc = UpdateCrc(crc, b);
            foreach (byte b in data) crc = UpdateCrc(crc, b);
            if (~crc != BinaryPrimitives.ReadUInt32BigEndian(checksum)) throw new InvalidDataException("PNG chunk checksum mismatch.");
            switch (kind)
            {
                case "IHDR":
                    if (width != 0 || data.Length != 13) throw new InvalidDataException("Invalid PNG header.");
                    width = checked((int)BinaryPrimitives.ReadUInt32BigEndian(data)); height = checked((int)BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(4)));
                    color = data[9]; interlace = data[12]; channels = color switch { 0 => 1, 2 => 3, 4 => 2, 6 => 4, _ => 0 };
                    if (width <= 0 || height <= 0 || (long)width * height > NativeImageDecoder.MaximumPixels || data[8] != 16 || channels == 0 || data[10] != 0 || data[11] != 0 || interlace > 1)
                        throw new InvalidDataException("Unsupported or invalid 16-bit PNG header.");
                    break;
                case "acTL": throw new NotSupportedException("Animated 16-bit PNG requires a dedicated animation decoder.");
                case "tRNS":
                    if (sawData || transparent is not null || color is not (0 or 2) || data.Length != channels * 2) throw new InvalidDataException("Invalid PNG transparency chunk.");
                    transparent = data; break;
                case "IDAT":
                    if (width == 0 || compressed.Length + data.Length > 64 * 1024 * 1024) throw new InvalidDataException("PNG data exceeds its budget or precedes its header.");
                    sawData = true; compressed.Write(data); break;
                case "IEND": if (data.Length != 0 || !sawData) throw new InvalidDataException("Invalid PNG end."); ended = true; break;
                case "PLTE": break;
                default: if (char.IsUpper(kind[0])) throw new NotSupportedException("Unsupported critical PNG chunk: " + kind); break;
            }
        }
        int outWidth = orientation >= 5 ? height : width, outHeight = orientation >= 5 ? width : height;
        var rgb = new float[checked(width * height * 3)]; var alpha = color is 4 or 6 || transparent is not null ? new float[checked(width * height)] : null;
        compressed.Position = 0; using var inflate = new ZLibStream(compressed, CompressionMode.Decompress);
        int[][] passes = interlace == 0 ? [[0,0,1,1]] : [[0,0,8,8],[4,0,8,8],[0,4,4,8],[2,0,4,4],[0,2,2,4],[1,0,2,2],[0,1,1,2]];
        foreach (var pass in passes)
        {
            int pw = Math.Max(0, (width - pass[0] + pass[2] - 1) / pass[2]), ph = Math.Max(0, (height - pass[1] + pass[3] - 1) / pass[3]);
            if (pw == 0 || ph == 0) continue;
            int bpp = channels * 2; var previous = new byte[pw * bpp]; var row = new byte[previous.Length];
            for (int y = 0; y < ph; y++)
            {
                token.ThrowIfCancellationRequested(); int filter = inflate.ReadByte(); inflate.ReadExactly(row);
                for (int i = 0; i < row.Length; i++)
                {
                    int a = i >= bpp ? row[i - bpp] : 0, b = previous[i], c = i >= bpp ? previous[i - bpp] : 0;
                    int predictor = filter switch { 0 => 0, 1 => a, 2 => b, 3 => (a + b) / 2, 4 => Paeth(a,b,c), _ => throw new InvalidDataException("Invalid PNG row filter.") };
                    row[i] = unchecked((byte)(row[i] + predictor));
                }
                for (int x = 0; x < pw; x++)
                {
                    int offset = x * bpp; var (dx,dy) = NativeImageDecoder.Orient(pass[0] + x * pass[2], pass[1] + y * pass[3], width, height, orientation);
                    int target = dy * outWidth + dx;
                    ushort r = BinaryPrimitives.ReadUInt16BigEndian(row.AsSpan(offset));
                    ushort g = color is 0 or 4 ? r : BinaryPrimitives.ReadUInt16BigEndian(row.AsSpan(offset + 2));
                    ushort b = color is 0 or 4 ? r : BinaryPrimitives.ReadUInt16BigEndian(row.AsSpan(offset + 4));
                    rgb[target * 3] = r / 65535f; rgb[target * 3 + 1] = g / 65535f; rgb[target * 3 + 2] = b / 65535f;
                    if (alpha is not null) alpha[target] = color is 4 or 6 ? BinaryPrimitives.ReadUInt16BigEndian(row.AsSpan(offset + bpp - 2)) / 65535f
                        : row.AsSpan(offset, bpp).SequenceEqual(transparent) ? 0 : 1;
                }
                (row, previous) = (previous, row);
            }
        }
        if (inflate.ReadByte() != -1) throw new InvalidDataException("Unexpected trailing PNG sample data.");
        return new(outWidth,outHeight,1,rgb,alpha);
    }
    private static int Paeth(int a,int b,int c) { int p=a+b-c, pa=Math.Abs(p-a), pb=Math.Abs(p-b), pc=Math.Abs(p-c); return pa<=pb && pa<=pc ? a : pb<=pc ? b : c; }
    private static uint UpdateCrc(uint crc,byte b) { crc ^= b; for(int i=0;i<8;i++) crc=(crc>>1)^((crc&1)!=0?0xedb88320u:0);return crc; }
}
