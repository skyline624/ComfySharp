using System.Text.Json.Serialization;

namespace ComfySharp.Contracts;

public sealed record ImageFileDescriptor(
    [property: JsonPropertyName("filename")] string Filename,
    [property: JsonPropertyName("subfolder")] string Subfolder,
    [property: JsonPropertyName("type")] string Type);

/// <summary>Host-owned local image files. Paths are scoped to the requested media directory.
/// Implementations must check containment for both directory preparation and each actual write.</summary>
public interface IImageFileStore
{
    /// <summary>Creates a missing subdirectory and returns all entry names, including directories.</summary>
    IReadOnlyList<string> PrepareDirectory(string type, string subfolder, CancellationToken cancellationToken = default);
    /// <summary>Writes one complete encoded image. Existing names may be replaced, as in SaveImage.
    /// Earlier successful writes in a batch remain if a subsequent write fails.</summary>
    ValueTask WriteAsync(ImageFileDescriptor file, ReadOnlyMemory<byte> png, CancellationToken cancellationToken = default);
}
