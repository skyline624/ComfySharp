using System.Security.Cryptography;
using ComfySharp.Contracts;
using ComfySharp.Media;
using ComfySharp.Storage;

namespace ComfySharp.Host;

public sealed class ImageInputService(ImageFileStore store) : IImageInputService
{
    public const int MaximumEncodedBytes = 64 * 1024 * 1024;
    private readonly SemaphoreSlim uploadGate = new(1, 1);
    private static readonly HashSet<string> Extensions = new(StringComparer.OrdinalIgnoreCase)
        { ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".ico", ".avif", ".tif", ".tiff", ".heic", ".heif", ".jxl", ".exr", ".hdr" };

    public IReadOnlyList<string> Names() => store.PrepareDirectory("input", "").Where(n => Extensions.Contains(Path.GetExtension(n)) && store.IsFile(new(n, "", "input"))).Order(StringComparer.Ordinal).ToArray();

    public async ValueTask<(DecodedImageBatch Images, string Sha256)> ReadAsync(string name, CancellationToken cancellationToken)
    {
        using var source = store.OpenRead(Descriptor(name));
        byte[] bytes = await ReadBoundedAsync(source, cancellationToken);
        using var encoded = new MemoryStream(bytes, writable: false);
        return (NativeImageDecoder.Decode(encoded, cancellationToken), Convert.ToHexStringLower(SHA256.HashData(bytes)));
    }

    public static ImageFileDescriptor Descriptor(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        string type = "input";
        foreach (string candidate in new[] { "input", "output", "temp" })
            if (name.EndsWith(" [" + candidate + "]", StringComparison.Ordinal))
            { type = candidate; name = name[..^(candidate.Length + 3)]; break; }
        name = name.Replace('\\', '/');
        if (name.StartsWith('/') || name.Contains(':') || name.Split('/').Any(p => p is "" or "." or ".."))
            throw new ArgumentException("Expected a relative image name inside a media directory.");
        int separator = name.LastIndexOf('/');
        return new(separator < 0 ? name : name[(separator + 1)..], separator < 0 ? "" : name[..separator], type);
    }

    public async Task<ImageFileDescriptor> UploadAsync(string filename, string subfolder, Stream source, bool overwrite, CancellationToken token)
    {
        if (string.IsNullOrWhiteSpace(filename) || filename.Contains('/') || filename.Contains('\\') || filename.Contains(':') || filename is "." or "..")
            throw new ArgumentException("Expected a plain image filename.");
        byte[] bytes = await ReadBoundedAsync(source, token);
        // Validate before committing an upload; a codec failure never leaves a partial file.
        using (var encoded = new MemoryStream(bytes, writable: false)) NativeImageDecoder.Decode(encoded, token);
        byte[] digest = SHA256.HashData(bytes);
        await uploadGate.WaitAsync(token);
        try
        {
            var names = store.PrepareDirectory("input", subfolder, token).ToHashSet(OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
            string target = filename; int index = 1;
            if (!overwrite)
                while (names.Contains(target))
                {
                    token.ThrowIfCancellationRequested();
                    var existing = new ImageFileDescriptor(target, subfolder, "input");
                    try
                    {
                        if (store.IsFile(existing))
                        {
                            using var stream = store.OpenRead(existing);
                            if (stream.Length == bytes.Length && SHA256.HashData(stream).AsSpan().SequenceEqual(digest)) return existing;
                        }
                    }
                    catch (Exception error) when (error is IOException or UnauthorizedAccessException) { }
                    target = $"{Path.GetFileNameWithoutExtension(filename)} ({index++}){Path.GetExtension(filename)}";
                }
            var file = new ImageFileDescriptor(target, subfolder, "input");
            await store.WriteAtomicAsync(file, bytes, token); return file;
        }
        finally { uploadGate.Release(); }
    }

    private static async Task<byte[]> ReadBoundedAsync(Stream source, CancellationToken token)
    {
        if (source.CanSeek && source.Length > MaximumEncodedBytes) throw new InvalidDataException("Encoded images are limited to 64 MiB.");
        using var buffer = new MemoryStream(); var chunk = new byte[65536];
        int count;
        while ((count = await source.ReadAsync(chunk, token)) != 0)
        {
            if (buffer.Length + count > MaximumEncodedBytes) throw new InvalidDataException("Encoded images are limited to 64 MiB.");
            buffer.Write(chunk, 0, count);
        }
        return buffer.ToArray();
    }
}
