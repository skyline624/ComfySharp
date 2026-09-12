using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Frozen VAEEncodeForInpaint preprocessing for SD's spatial compression of eight.
/// Caller owns both returned tensors; input IMAGE and MASK storages are unchanged.</summary>
public static class SdInpaintImage
{
    public static (Tensor Pixels, Tensor NoiseMask) Prepare(Tensor pixels, Tensor mask, int growMaskBy = 6,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        ArgumentNullException.ThrowIfNull(pixels); ArgumentNullException.ThrowIfNull(mask);
        if (!InferenceDevice.IsSupported(pixels.device_type) || pixels.dtype != ScalarType.Float32 || pixels.is_sparse ||
            pixels.dim() != 4 || pixels.shape[0] <= 0 || pixels.shape[1] < 8 || pixels.shape[2] < 8 || pixels.shape[3] is not (3 or 4))
            throw new ArgumentException("Expected dense CPU/CUDA Float32 IMAGE [batch,height>=8,width>=8,3 or 4].", nameof(pixels));
        if (growMaskBy is < 0 or > 64) throw new ArgumentOutOfRangeException(nameof(growMaskBy));
        if (mask.dtype != ScalarType.Float32 || mask.is_sparse || mask.dim() < 2 || mask.shape.Any(v => v <= 0) ||
            !mask.isfinite().all().item<bool>()) throw new ArgumentException("Expected a nonempty finite Float32 mask.", nameof(mask));
        InferenceDevice.RequireSame(pixels.device, mask, nameof(mask));
        var shaped = mask.reshape(-1, 1, mask.shape[^2], mask.shape[^1]);
        if (shaped.shape[0] != 1 && shaped.shape[0] != pixels.shape[0])
            throw new ArgumentException("Mask batch must be one or match the image batch.", nameof(mask));
        var resized = nn.functional.interpolate(shaped, size: pixels.shape[1..3], mode: InterpolationMode.Bilinear, align_corners: false);
        long height = pixels.shape[1] / 8 * 8, width = pixels.shape[2] / 8 * 8;
        long y = pixels.shape[1] % 8 / 2, x = pixels.shape[2] % 8 / 2;
        var cropped = pixels.clone().narrow(1, y, height).narrow(2, x, width);
        var croppedMask = resized.narrow(2, y, height).narrow(3, x, width);
        Tensor grown = croppedMask;
        if (growMaskBy != 0)
        {
            var kernel = ones(new long[] { 1, 1, growMaskBy, growMaskBy }, device: pixels.device, dtype: ScalarType.Float32);
            grown = nn.functional.conv2d(croppedMask.round(), kernel, padding: new long[] { growMaskBy / 2, growMaskBy / 2 }).clamp(0, 1);
        }
        var inverse = (1.0 - croppedMask.round()).squeeze(1);
        for (int channel = 0; channel < 3; channel++)
        {
            var plane = cropped.select(3, channel);
            plane.sub_(.5); plane.mul_(inverse); plane.add_(.5);
        }
        // Even kernels produce an extra bottom/right pixel upstream; crop, do not center again.
        var resultMask = grown.narrow(2, 0, height).narrow(3, 0, width).round();
        cancellationToken.ThrowIfCancellationRequested();
        return (cropped.DetachFromDisposeScope(), resultMask.DetachFromDisposeScope());
    }
}
