using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Image primitives for dense CPU/Float32 NHWC RGB or RGBA tensors with positive dimensions.
/// Inputs are borrowed and never mutated. Each returned tensor owns independent storage and must be disposed.
/// This profile does not select devices or intermediate dtypes from upstream runtime options.</summary>
public static partial class ImageOperations
{
    /// <summary>Default per-call limit for newly allocated tensor payloads: 512 MiB.
    /// EmptyImage admits twice its output bytes (three channel tensors plus concatenation); Invert, RepeatBatch and FromBatch
    /// admit their output bytes. Batch also admits its padding and resize payloads. Borrowed inputs, allocator overhead/caches and concurrent calls are excluded.
    /// This is a local admission limit, configurable per call, not an upstream schema constraint or RAM guarantee.</summary>
    public const long DefaultMaxAllocationBytes = 512L * 1024 * 1024;

    public static Tensor EmptyImage(long width, long height, long batchSize = 1, int color = 0,
        CancellationToken cancellationToken = default, long maxAllocationBytes = DefaultMaxAllocationBytes)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        if (color is < 0 or > 0xFFFFFF) throw new ArgumentOutOfRangeException(nameof(color));
        Admit(batchSize, height, width, 3, 2, maxAllocationBytes);
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        long[] channelShape = [batchSize, height, width, 1];
        // Preserve the source's double division followed by Float32 full, then RGB concatenation.
        var red = full(channelShape, ((color >> 16) & 255) / 255.0, dtype: ScalarType.Float32, device: CPU);
        cancellationToken.ThrowIfCancellationRequested();
        var green = full(channelShape, ((color >> 8) & 255) / 255.0, dtype: ScalarType.Float32, device: CPU);
        cancellationToken.ThrowIfCancellationRequested();
        var blue = full(channelShape, (color & 255) / 255.0, dtype: ScalarType.Float32, device: CPU);
        cancellationToken.ThrowIfCancellationRequested();
        return Finish(cat([red, green, blue], dim: -1), cancellationToken);
    }

    /// <summary>Compute 1-image without clamping, restoring the original alpha channel for RGBA.</summary>
    public static Tensor Invert(Tensor image, CancellationToken cancellationToken = default,
        long maxAllocationBytes = DefaultMaxAllocationBytes)
    {
        cancellationToken.ThrowIfCancellationRequested();
        NativeRuntimeBootstrap.Initialize();
        var shape = Validate(image);
        Admit(shape[0], shape[1], shape[2], shape[3], 1, maxAllocationBytes);
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        var result = 1.0 - image;
        cancellationToken.ThrowIfCancellationRequested();
        if (shape[3] == 4) result.narrow(3, 3, 1).copy_(image.narrow(3, 3, 1));
        return Finish(result, cancellationToken);
    }

    /// <summary>Repeat the whole batch sequence, including an independent copy when amount is one.</summary>
    public static Tensor RepeatBatch(Tensor image, long amount, CancellationToken cancellationToken = default,
        long maxAllocationBytes = DefaultMaxAllocationBytes)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(amount);
        NativeRuntimeBootstrap.Initialize();
        var shape = Validate(image);
        Admit(checked(shape[0] * amount), shape[1], shape[2], shape[3], 1, maxAllocationBytes);
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        return Finish(image.repeat(amount, 1, 1, 1), cancellationToken);
    }

    /// <summary>Add batch size once to a negative index, clamp it into the batch, truncate length,
    /// then clone the selected images. Positive length is required by this local profile.</summary>
    public static Tensor FromBatch(Tensor image, long batchIndex, long length,
        CancellationToken cancellationToken = default, long maxAllocationBytes = DefaultMaxAllocationBytes)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        NativeRuntimeBootstrap.Initialize();
        var shape = Validate(image);
        // Negative + positive cannot overflow Int64; avoid negating long.MinValue.
        if (batchIndex < 0) batchIndex += shape[0];
        batchIndex = Math.Clamp(batchIndex, 0, shape[0] - 1);
        length = Math.Min(shape[0] - batchIndex, length);
        Admit(length, shape[1], shape[2], shape[3], 1, maxAllocationBytes);
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        return Finish(image.narrow(0, batchIndex, length).clone(), cancellationToken);
    }

    private static long[] Validate(Tensor image)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.device_type != DeviceType.CPU || image.dtype != ScalarType.Float32 || image.is_sparse)
            throw new ArgumentException("Images require dense CPU/Float32 NHWC RGB or RGBA tensors.", nameof(image));
        var shape = image.shape;
        if (shape.Length != 4 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 || shape[3] is not (3 or 4))
            throw new ArgumentException("Images require positive batch, height and width and exactly three or four channels.", nameof(image));
        return shape;
    }

    private static void Admit(long batch, long height, long width, long channels, long copies, long maxAllocationBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxAllocationBytes);
        long bytes = checked(batch * height * width * channels * sizeof(float) * copies);
        if (bytes > maxAllocationBytes)
            throw new InvalidOperationException($"Image allocation requires {bytes} payload bytes, exceeding the explicit per-call budget of {maxAllocationBytes} bytes.");
    }

    private static Tensor Finish(Tensor result, CancellationToken cancellationToken)
    {
        // Synchronous native kernels cannot be interrupted; cancellation at this boundary disposes the result.
        cancellationToken.ThrowIfCancellationRequested();
        return result.DetachFromDisposeScope();
    }
}
