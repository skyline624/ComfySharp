using System.Runtime.InteropServices;
using ComfySharp.Contracts;
using SkiaSharp;

namespace ComfySharp.Media;

/// <summary>Native image codecs and lossless 16-bit PNG samples, with explicit EXIF orientation.</summary>
public static class NativeImageDecoder
{
    public const long MaximumPixels = 16 * 1024 * 1024;

    public static DecodedImageBatch Decode(Stream encoded, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!encoded.CanSeek) throw new ArgumentException("Image decoding requires a seekable bounded stream.", nameof(encoded));
        long start = encoded.Position;
        byte[] header = new byte[26]; int headerBytes = encoded.ReadAtLeast(header, 26, throwOnEndOfStream: false); encoded.Position = start;
        bool png = headerBytes == 26 && header.AsSpan(0, 8).SequenceEqual(new byte[] { 137,80,78,71,13,10,26,10 });
        var metadata = png ? Png16Decoder.ReadMetadata(encoded, cancellationToken) : (Frames: 1, Orientation: (int?)null); encoded.Position = start;
        using var codec = SKCodec.Create(encoded) ?? throw new InvalidDataException("Unsupported or invalid image encoding.");
        int width = codec.Info.Width, height = codec.Info.Height, frames = Math.Max(1, codec.FrameCount);
        long total = checked((long)width * height * frames);
        if (width <= 0 || height <= 0 || total > MaximumPixels)
            throw new NotSupportedException("Decoded image batches currently support at most 16 million pixels.");
        int orientation = metadata.Orientation ?? (int)codec.EncodedOrigin;
        if (orientation is < 1 or > 8) throw new InvalidDataException("Unknown image orientation.");
        if (png)
        {
            if (metadata.Frames > frames) throw new NotSupportedException("This native codec cannot decode all frames of the animated PNG.");
            if (header[24] == 16) { encoded.Position = start; return Png16Decoder.Decode(encoded, orientation, cancellationToken); }
        }
        int outWidth = orientation >= 5 ? height : width, outHeight = orientation >= 5 ? width : height;
        // The source treats decoded palette frames as alpha-bearing, even if every entry is opaque.
        bool hasAlpha = codec.Info.AlphaType != SKAlphaType.Opaque || codec.EncodedFormat == SKEncodedImageFormat.Gif || png && header[25] is 3 or 4 or 6;
        var rgb = new float[checked((int)total * 3)];
        var alpha = hasAlpha ? new float[checked((int)total)] : null;
        var info = new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        using var bitmap = new SKBitmap(info);
        if (bitmap.GetPixels() == IntPtr.Zero) throw new OutOfMemoryException("Image decode buffer allocation failed.");
        var rgba = new byte[checked(width * height * 4)];
        for (int frame = 0; frame < frames; frame++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            // -1 asks the codec to reconstruct any required earlier frames, including restore-previous disposal.
            var result = codec.GetPixels(info, bitmap.GetPixels(), bitmap.RowBytes, new SKCodecOptions(frame, -1));
            if (result != SKCodecResult.Success) throw new InvalidDataException($"Image frame {frame} decoding failed: {result}.");
            for (int y = 0; y < height; y++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Marshal.Copy(bitmap.GetPixels() + y * bitmap.RowBytes, rgba, y * width * 4, width * 4);
                for (int x = 0; x < width; x++)
                {
                    var (dx, dy) = Orient(x, y, width, height, orientation);
                    int target = checked((frame * outHeight + dy) * outWidth + dx), source = (y * width + x) * 4;
                    rgb[target * 3] = rgba[source] / 255f; rgb[target * 3 + 1] = rgba[source + 1] / 255f; rgb[target * 3 + 2] = rgba[source + 2] / 255f;
                    if (alpha is not null) alpha[target] = rgba[source + 3] / 255f;
                }
            }
        }
        return new(outWidth, outHeight, frames, rgb, alpha);
    }

    internal static (int X, int Y) Orient(int x, int y, int width, int height, int origin) => origin switch
    {
        1 => (x, y), 2 => (width - 1 - x, y), 3 => (width - 1 - x, height - 1 - y), 4 => (x, height - 1 - y),
        5 => (y, x), 6 => (height - 1 - y, x), 7 => (height - 1 - y, width - 1 - x), 8 => (y, width - 1 - x),
        _ => throw new ArgumentOutOfRangeException(nameof(origin))
    };
}
