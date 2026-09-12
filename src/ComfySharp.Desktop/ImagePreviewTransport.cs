using System.Buffers.Binary;
using System.Text.Json.Nodes;
using Avalonia.Media.Imaging;

namespace ComfySharp.Desktop;

public sealed record PreviewImageFile(string Filename, string Subfolder, string Type)
{
    public static PreviewImageFile Parse(JsonNode? node)
    {
        if (node is not JsonObject entry || entry["filename"] is not JsonValue name || !name.TryGetValue<string>(out var filename) ||
            entry["subfolder"] is not JsonValue folder || !folder.TryGetValue<string>(out var subfolder) ||
            entry["type"] is not JsonValue kind || !kind.TryGetValue<string>(out var type))
            throw new InvalidDataException("Invalid image file descriptor.");
        var result = new PreviewImageFile(filename, subfolder, type); result.Validate(); return result;
    }
    public void Validate()
    {
        if (Type is not ("output" or "temp") || string.IsNullOrWhiteSpace(Filename) ||
            !Filename.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || Filename.IndexOfAny(['/', '\\', ':', '\0']) >= 0 ||
            Subfolder is null || Subfolder.Contains('\0'))
            throw new InvalidDataException("Only original output/temp PNG descriptors can be previewed.");
    }
}

/// <summary>Reads bounded original PNGs from the supervised local Host. Does not load tensor libraries.</summary>
public static class ImagePreviewTransport
{
    public const int MaximumPngBytes = 128 * 1024 * 1024;
    public const long MaximumPixels = 16_777_216;
    public const int ThumbnailExtent = 512;

    public static async Task<byte[]> ReadAsync(HttpClient client, Uri address, PreviewImageFile file, CancellationToken token = default)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(token); timeout.CancelAfter(TimeSpan.FromSeconds(30)); token = timeout.Token;
        file.Validate();
        if (!address.IsAbsoluteUri || address.Scheme != "http" || !address.IsLoopback || address.UserInfo.Length != 0)
            throw new InvalidDataException("Image retrieval requires the local Host address.");
        var uri = new Uri(address, "/view?filename=" + Uri.EscapeDataString(file.Filename) +
            "&subfolder=" + Uri.EscapeDataString(file.Subfolder) + "&type=" + file.Type);
        using var response = await client.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, token);
        response.EnsureSuccessStatusCode();
        if (response.Content.Headers.ContentType?.MediaType != "image/png") throw new InvalidDataException("Host did not return a PNG image.");
        if (response.Content.Headers.ContentLength is > MaximumPngBytes) throw new InvalidDataException("Preview PNG exceeds the size limit.");
        await using var stream = await response.Content.ReadAsStreamAsync(token);
        using var bytes = new MemoryStream(); var buffer = new byte[64 * 1024];
        while (true)
        {
            int count = await stream.ReadAsync(buffer, token); if (count == 0) break;
            if (bytes.Length + count > MaximumPngBytes) throw new InvalidDataException("Preview PNG exceeds the size limit.");
            bytes.Write(buffer, 0, count);
        }
        return bytes.ToArray();
    }

    public static Bitmap Decode(byte[] png)
    {
        if (png.Length is < 33 or > MaximumPngBytes || !png.AsSpan(0, 8).SequenceEqual(new byte[] { 137,80,78,71,13,10,26,10 }) ||
            BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(8)) != 13 || !png.AsSpan(12, 4).SequenceEqual("IHDR"u8))
            throw new InvalidDataException("Invalid PNG preview header.");
        uint width = BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(16));
        uint height = BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(20));
        if (width == 0 || height == 0 || (ulong)width * height > MaximumPixels)
            throw new InvalidDataException("Preview image dimensions exceed the pixel limit.");
        using var stream = new MemoryStream(png, writable: false);
        return Math.Max(width, height) <= ThumbnailExtent ? new Bitmap(stream) : width >= height
            ? Bitmap.DecodeToWidth(stream, ThumbnailExtent) : Bitmap.DecodeToHeight(stream, ThumbnailExtent);
    }
}
