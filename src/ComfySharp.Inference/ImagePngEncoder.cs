using System.Buffers.Binary;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>One ordered PNG text chunk. Repeated keywords are retained.</summary>
public sealed record PngText(string Keyword, string Text);

/// <summary>PNG8 RGB/RGBA encoding for finite, non-grad CPU Float32 NHWC tensors.
/// Quantization follows Float32 multiplication by 255, saturation and truncation.
/// Inputs are borrowed. Frame pixel count, metadata bytes and encoded bytes have separate admission bounds;
/// these are not an aggregate process-memory guarantee. No Pillow or Python runtime is used.</summary>
public static class ImagePngEncoder
{
    public const int DefaultMaxPixels = 16_777_216;
    public const int DefaultMaxEncodedBytes = 128 * 1024 * 1024;
    public const int MaxMetadataBytes = 1024 * 1024;
    private static readonly UTF8Encoding Utf8 = new(false, true);
    private static readonly uint[] CrcTable = BuildCrcTable();

    public static byte[] EncodeFrame(Tensor images, long batchIndex, IReadOnlyList<PngText>? metadata = null,
        int compressionLevel = 4, CancellationToken cancellationToken = default,
        int maxPixels = DefaultMaxPixels, int maxEncodedBytes = DefaultMaxEncodedBytes)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxPixels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxEncodedBytes);
        if (compressionLevel is < 0 or > 9) throw new ArgumentOutOfRangeException(nameof(compressionLevel));
        var chunks = TextChunks(metadata, cancellationToken);
        NativeRuntimeBootstrap.Initialize();
        ArgumentNullException.ThrowIfNull(images);
        long[] shape = images.shape;
        if (images.device_type != DeviceType.CPU || images.dtype != ScalarType.Float32 || images.is_sparse || images.requires_grad ||
            shape.Length != 4 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 || shape[3] is not (3 or 4))
            throw new ArgumentException("PNG encoding requires non-grad dense CPU/Float32 NHWC RGB or RGBA with positive dimensions.", nameof(images));
        if (batchIndex < 0 || batchIndex >= shape[0]) throw new ArgumentOutOfRangeException(nameof(batchIndex));
        long pixels = checked(shape[1] * shape[2]);
        if (pixels > maxPixels) throw new InvalidOperationException($"PNG frame has {pixels} pixels, exceeding the configured limit {maxPixels}.");
        int width = checked((int)shape[2]), height = checked((int)shape[1]), channels = (int)shape[3];
        byte[] samples = Quantize(images, batchIndex, checked((int)(pixels * channels)), cancellationToken);
        using var destination = new MemoryStream();
        using var bounded = new BoundedOutput(destination, maxEncodedBytes, cancellationToken);
        bounded.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });
        Span<byte> header = stackalloc byte[13]; header.Clear();
        BinaryPrimitives.WriteInt32BigEndian(header, width); BinaryPrimitives.WriteInt32BigEndian(header[4..], height);
        header[8] = 8; header[9] = channels == 3 ? (byte)2 : (byte)6;
        WriteChunk(bounded, "IHDR"u8, header);
        foreach (var (international, data) in chunks) WriteChunk(bounded, international ? "iTXt"u8 : "tEXt"u8, data);
        using (var idat = new IdatOutput(bounded))
        {
            using (var compression = new ZLibStream(idat, new ZLibCompressionOptions { CompressionLevel = compressionLevel }, leaveOpen: true))
            {
                int rowBytes = checked(width * channels);
                for (int row = 0; row < height; row++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    compression.WriteByte(0); // Legal PNG filter None. No byte-parity claim with Pillow's filter choices.
                    compression.Write(samples.AsSpan(row * rowBytes, rowBytes));
                }
            }
            idat.Finish();
        }
        WriteChunk(bounded, "IEND"u8, []);
        cancellationToken.ThrowIfCancellationRequested();
        return destination.ToArray();
    }

    private static byte[] Quantize(Tensor images, long batchIndex, int elements, CancellationToken cancellationToken)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        var frame = images.select(0, batchIndex).contiguous();
        ReadOnlySpan<float> source = MemoryMarshal.Cast<byte, float>(frame.bytes);
        var pixels = new byte[elements];
        for (int i = 0; i < elements; i++)
        {
            if ((i & 4095) == 0) cancellationToken.ThrowIfCancellationRequested();
            float value = source[i];
            if (!float.IsFinite(value)) throw new ArgumentException("PNG input values must be finite.", nameof(images));
            float scaled = value * 255.0f;
            // A finite extreme input may overflow the multiplication; saturation still matches NumPy clip.
            pixels[i] = scaled <= 0 ? (byte)0 : scaled >= 255 ? (byte)255 : (byte)scaled;
        }
        return pixels;
    }

    private static List<(bool International, byte[] Data)> TextChunks(IReadOnlyList<PngText>? metadata, CancellationToken cancellationToken)
    {
        var chunks = new List<(bool, byte[])>();
        if (metadata is null) return chunks;
        int total = 0;
        foreach (var item in metadata)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ArgumentNullException.ThrowIfNull(item); ArgumentNullException.ThrowIfNull(item.Keyword); ArgumentNullException.ThrowIfNull(item.Text);
            string key = item.Keyword;
            if (key.Length is < 1 or > 79 || key[0] == ' ' || key[^1] == ' ' || key.Contains("  ", StringComparison.Ordinal) ||
                key.Any(c => c is < (char)32 or > (char)255 || c is >= (char)127 and <= (char)160))
                throw new ArgumentException("PNG text keywords require 1-79 printable Latin-1 bytes with single internal spaces.", nameof(metadata));
            if (item.Text.Contains('\0')) throw new ArgumentException("PNG text must not contain a null character.", nameof(metadata));
            bool international = item.Text.Any(c => c > 255);
            var encoding = international ? Utf8 : Encoding.Latin1;
            int length = checked(key.Length + (international ? 5 : 1) + encoding.GetByteCount(item.Text));
            total = checked(total + length + 12);
            if (total > MaxMetadataBytes) throw new InvalidOperationException("PNG text exceeds the 1 MiB metadata limit.");
            byte[] data = new byte[length];
            Encoding.Latin1.GetBytes(key, data.AsSpan());
            // iTXt uses no compression, no language tag and no translated keyword: five zero separators/flags.
            encoding.GetBytes(item.Text, data.AsSpan(key.Length + (international ? 5 : 1)));
            chunks.Add((international, data));
        }
        return chunks;
    }

    private static void WriteChunk(Stream destination, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        Span<byte> number = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(number, data.Length); destination.Write(number); destination.Write(type); destination.Write(data);
        uint crc = uint.MaxValue;
        foreach (byte value in type) crc = CrcTable[(crc ^ value) & 255] ^ (crc >> 8);
        foreach (byte value in data) crc = CrcTable[(crc ^ value) & 255] ^ (crc >> 8);
        BinaryPrimitives.WriteUInt32BigEndian(number, ~crc); destination.Write(number);
    }

    private static uint[] BuildCrcTable()
    {
        var table = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint value = i;
            for (int bit = 0; bit < 8; bit++) value = (value & 1) != 0 ? 0xedb88320 ^ (value >> 1) : value >> 1;
            table[i] = value;
        }
        return table;
    }

    private abstract class WriteOnlyStream : Stream
    {
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
        public override void WriteByte(byte value) { Span<byte> one = stackalloc byte[1]; one[0] = value; Write(one); }
    }

    private sealed class BoundedOutput(Stream destination, int limit, CancellationToken cancellationToken) : WriteOnlyStream
    {
        private long written;
        public override void Write(ReadOnlySpan<byte> buffer)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (buffer.Length > limit - written) throw new InvalidOperationException("PNG exceeds the configured encoded-byte limit.");
            destination.Write(buffer); written += buffer.Length;
        }
        public override void Flush() => destination.Flush();
    }

    private sealed class IdatOutput(Stream destination) : WriteOnlyStream
    {
        private readonly byte[] buffer = new byte[65536];
        private int count;
        private bool finished;
        public override void Write(ReadOnlySpan<byte> data)
        {
            ObjectDisposedException.ThrowIf(finished, this);
            while (!data.IsEmpty)
            {
                int copied = Math.Min(buffer.Length - count, data.Length);
                data[..copied].CopyTo(buffer.AsSpan(count)); count += copied; data = data[copied..];
                if (count == buffer.Length) Flush();
            }
        }
        public override void Flush()
        {
            if (count == 0) return;
            WriteChunk(destination, "IDAT"u8, buffer.AsSpan(0, count)); count = 0;
        }
        public void Finish() { if (finished) return; Flush(); finished = true; }
    }
}
