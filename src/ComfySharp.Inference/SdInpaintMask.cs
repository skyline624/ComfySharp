using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Default SD inpainting: immutable snapshots of scaled latent, original noise and soft mask.
/// White regenerates; black preserves the latent prediction. This is not nine-channel inpaint conditioning.</summary>
public sealed class SdInpaintMask : IDisposable
{
    private readonly object gate = new();
    private Tensor? latent;
    private Tensor? noise;
    private Tensor? mask;

    public SdInpaintMask(Tensor scaledLatent, Tensor noise, Tensor mask, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        SdSamplingMath.ValidateLatent(scaledLatent, nameof(scaledLatent));
        SdSamplingMath.ValidateLatent(noise, nameof(noise));
        InferenceDevice.RequireSame(scaledLatent.device, noise, nameof(noise));
        if (!scaledLatent.shape.SequenceEqual(noise.shape)) throw new ArgumentException("Noise and latent shapes must match.", nameof(noise));
        // Resize on the mask's original device, then transfer, as prepare_mask does upstream.
        using var prepared = PrepareMask(mask, scaledLatent.shape, cancellationToken);
        var ownedMask = prepared.to(scaledLatent.device, copy: true);
        var ownedLatent = scaledLatent.clone(); var ownedNoise = noise.clone();
        cancellationToken.ThrowIfCancellationRequested();
        this.mask = ownedMask.DetachFromDisposeScope();
        latent = ownedLatent.DetachFromDisposeScope(); this.noise = ownedNoise.DetachFromDisposeScope();
    }

    private SdInpaintMask(Tensor latent, Tensor noise, Tensor mask)
    {
        using var scope = NewDisposeScope();
        var l = latent.alias(); var n = noise.alias(); var m = mask.alias();
        this.latent = l.DetachFromDisposeScope(); this.noise = n.DetachFromDisposeScope(); this.mask = m.DetachFromDisposeScope();
    }

    internal SdInpaintMask Retain()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(latent is null, this);
            return new(latent, noise!, mask!);
        }
    }

    public static Tensor PrepareMask(Tensor mask, long[] latentShape, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        ArgumentNullException.ThrowIfNull(mask); ArgumentNullException.ThrowIfNull(latentShape);
        if (latentShape.Length != 4 || latentShape.Any(v => v <= 0) || latentShape[1] != 4)
            throw new ArgumentException("Expected SD latent shape [batch,4,height,width].", nameof(latentShape));
        if (!InferenceDevice.IsSupported(mask.device_type) || mask.dtype != ScalarType.Float32 || mask.is_sparse ||
            mask.dim() < 2 || mask.shape.Any(v => v <= 0) || !mask.isfinite().all().item<bool>())
            throw new ArgumentException("Mask requires a nonempty dense finite CPU/CUDA Float32 tensor with at least two dimensions.", nameof(mask));
        var shaped = mask.reshape(-1, 1, mask.shape[^2], mask.shape[^1]);
        var resized = nn.functional.interpolate(shaped, size: latentShape[2..], mode: InterpolationMode.Bilinear, align_corners: false);
        var channels = resized.repeat(1, latentShape[1], 1, 1);
        var batch = channels.shape[0] < latentShape[0]
            ? channels.repeat(checked((latentShape[0] + channels.shape[0] - 1) / channels.shape[0]), 1, 1, 1) : channels;
        var result = batch.narrow(0, 0, latentShape[0]);
        cancellationToken.ThrowIfCancellationRequested(); return result.DetachFromDisposeScope();
    }

    internal Tensor Denoise(Tensor current, Tensor sigma, Func<Tensor, Tensor, Tensor> model,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(model);
        using var operation = Retain(); using var scope = NewDisposeScope(); using var noGrad = no_grad();
        SdSamplingMath.ValidateLatent(current, nameof(current));
        InferenceDevice.RequireSame(operation.latent!.device, current, nameof(current));
        if (!current.shape.SequenceEqual(operation.latent.shape)) throw new ArgumentException("Current latent shape does not match inpaint data.", nameof(current));
        var inverse = 1.0 - operation.mask!;
        // This call intentionally uses ordinary sigma*noise, never maximum-denoise sqrt(1+sigma^2).
        using var sourceNoised = SdSamplingMath.NoiseScaling(operation.noise!, operation.latent, sigma, cancellationToken: cancellationToken);
        var blended = current * operation.mask! + sourceNoised * inverse;
        cancellationToken.ThrowIfCancellationRequested();
        using var predicted = model(blended, sigma);
        cancellationToken.ThrowIfCancellationRequested();
        SdSamplingMath.ValidateLatent(predicted, nameof(predicted));
        InferenceDevice.RequireSame(current.device, predicted, nameof(predicted));
        if (!predicted.shape.SequenceEqual(current.shape)) throw new ArgumentException("Prediction shape must match latent.", nameof(model));
        var result = predicted * operation.mask! + operation.latent * inverse;
        cancellationToken.ThrowIfCancellationRequested(); return result.DetachFromDisposeScope();
    }

    public void Dispose()
    {
        Tensor? l, n, m;
        lock (gate) { l = latent; n = noise; m = mask; latent = noise = mask = null; }
        l?.Dispose(); n?.Dispose(); m?.Dispose();
    }
}
