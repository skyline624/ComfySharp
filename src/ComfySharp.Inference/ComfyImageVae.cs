using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Comfy image conventions over the classical CPU/Float32 VAE.
/// Uses complete images; no tiled, sliced or GPU fallback is implied.</summary>
public sealed class ComfyImageVae : IDisposable
{
    private readonly object gate = new();
    private ClassicalVae? vae;

    public ComfyImageVae(ClassicalVae vae)
    {
        ArgumentNullException.ThrowIfNull(vae);
        this.vae = vae.Retain();
        Config = vae.Config;
    }

    public ClassicalVaeConfig Config { get; }

    public ComfyImageVae Retain()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(vae is null, this);
            return new ComfyImageVae(vae);
        }
    }

    /// <summary>Encode NHWC image pixels into an owned raw mean latent.
    /// Centrally crop to multiples of eight, keep RGB, then apply image*2-1 without clamping.</summary>
    public Tensor Encode(Tensor image, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var operation = RetainVae();
        NativeRuntimeBootstrap.Initialize();
        ClassicalVae.ValidateCpuTensor(image, nameof(image));
        var shape = image.shape;
        const int factor = ClassicalVaeConfig.Compression;
        if (shape.Length != 4 || shape[0] <= 0 || shape[1] < factor || shape[2] < factor || shape[3] < ClassicalVaeConfig.ImageChannels)
            throw new ArgumentException("VAE image input must be positive-batch NHWC, at least 8x8 pixels and at least three channels.", nameof(image));
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        long height = shape[1] / factor * factor, width = shape[2] / factor * factor;
        var pixels = image.narrow(1, (shape[1] - height) / 2, height)
            .narrow(2, (shape[2] - width) / 2, width).narrow(3, 0, ClassicalVaeConfig.ImageChannels)
            .permute(0, 3, 1, 2);
        var normalized = pixels * 2 - 1;
        using var mean = operation.Encode(normalized, cancellationToken);
        // Source VAE.encode copies the raw posterior mean into its own NCHW
        // output buffer; keep the source compute strides until this boundary.
        var samples = empty(mean.shape, dtype: ScalarType.Float32, device: CPU);
        samples.copy_(mean);
        cancellationToken.ThrowIfCancellationRequested();
        return samples.DetachFromDisposeScope();
    }

    /// <summary>Decode raw NCHW latents into an owned NHWC image view in [0,1].</summary>
    public Tensor Decode(Tensor latent, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var operation = RetainVae();
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        using var raw = operation.Decode(latent, cancellationToken);
        // Source materializes a NCHW output buffer before its in-place image
        // normalization, then returns movedim's NHWC view of that buffer.
        var pixels = empty(raw.shape, dtype: ScalarType.Float32, device: CPU);
        pixels.copy_(raw);
        pixels.add_(1.0).div_(2.0).clamp_(0, 1);
        var image = pixels.permute(0, 2, 3, 1);
        cancellationToken.ThrowIfCancellationRequested();
        return image.DetachFromDisposeScope();
    }

    private ClassicalVae RetainVae()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(vae is null, this);
            return vae.Retain();
        }
    }

    public void Dispose()
    {
        ClassicalVae? released;
        lock (gate) { released = vae; vae = null; }
        released?.Dispose();
    }
}
