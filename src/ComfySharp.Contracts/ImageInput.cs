namespace ComfySharp.Contracts;

/// <summary>Decoded RGB NHWC frames and optional alpha NHW, with independent managed storage.</summary>
public sealed record DecodedImageBatch(int Width, int Height, int Frames, float[] Rgb, float[]? Alpha);

/// <summary>Host-owned image input catalogue. Every read observes the current file contents.</summary>
public interface IImageInputService
{
    IReadOnlyList<string> Names();
    ValueTask<(DecodedImageBatch Images, string Sha256)> ReadAsync(string name, CancellationToken cancellationToken);
}
