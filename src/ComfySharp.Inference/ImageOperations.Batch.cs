using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

public static partial class ImageOperations
{
    /// <summary>Concatenate IMAGE batches. Promote RGB to RGBA with alpha=1 when needed,
    /// then center-crop and bilinearly resize only image2 to image1's spatial size.
    /// Both inputs are borrowed; the result owns independent storage.</summary>
    /// <remarks>The allocation limit includes every newly allocated image payload retained
    /// by this call: optional padding, optional resized second input and final concatenation.
    /// Borrowed inputs, allocator caches/overhead and internal kernel temporaries are excluded.</remarks>
    public static Tensor Batch(Tensor image1, Tensor image2,
        CancellationToken cancellationToken = default,
        long maxAllocationBytes = DefaultMaxAllocationBytes)
    {
        cancellationToken.ThrowIfCancellationRequested();
        NativeRuntimeBootstrap.Initialize();
        var firstShape = Validate(image1);
        var secondShape = Validate(image2);
        AdmitBatch(firstShape, secondShape, maxAllocationBytes);
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        var first = image1;
        var second = image2;
        // Within RGB/RGBA this is exactly the source's one-channel promotion.
        // Do not replace it with an arbitrary padding-to-max-channels algorithm.
        if (firstShape[3] != secondShape[3])
        {
            if (firstShape[3] > secondShape[3])
                second = nn.functional.pad(second, new long[] { 0, 1 }, PaddingModes.Constant, 1.0);
            else
                first = nn.functional.pad(first, new long[] { 0, 1 }, PaddingModes.Constant, 1.0);
            cancellationToken.ThrowIfCancellationRequested();
        }

        // After RGB/RGBA promotion the channel counts match; only H/W can differ.
        // The source intentionally ignores different batch counts here.
        if (firstShape[1] != secondShape[1] || firstShape[2] != secondShape[2])
        {
            var samples = second.permute(0, 3, 1, 2);
            long oldWidth = secondShape[2], oldHeight = secondShape[1];
            long width = firstShape[2], height = firstShape[1];
            double oldAspect = (double)oldWidth / oldHeight;
            double newAspect = (double)width / height;
            long x = 0, y = 0;
            // Preserve Python's binary64 expression order and round-to-even.
            if (oldAspect > newAspect)
                x = checked((long)Math.Round((oldWidth - oldWidth * (newAspect / oldAspect)) / 2,
                    MidpointRounding.ToEven));
            else if (oldAspect < newAspect)
                y = checked((long)Math.Round((oldHeight - oldHeight * (oldAspect / newAspect)) / 2,
                    MidpointRounding.ToEven));
            var cropped = samples.narrow(-2, y, checked(oldHeight - y * 2))
                .narrow(-1, x, checked(oldWidth - x * 2));
            // No crop-size clamp/fallback: upstream also lets interpolation reject
            // a zero-size crop caused by an extreme aspect ratio.
            var resized = nn.functional.interpolate(cropped, size: new long[] { height, width },
                mode: InterpolationMode.Bilinear, align_corners: false, antialias: false);
            second = resized.permute(0, 2, 3, 1);
            cancellationToken.ThrowIfCancellationRequested();
        }

        return Finish(cat(new[] { first, second }, dim: 0), cancellationToken);
    }

    private static void AdmitBatch(long[] first, long[] second, long maxAllocationBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxAllocationBytes);
        long channels = Math.Max(first[3], second[3]);
        long bytes = BatchPayload(checked(first[0] + second[0]), first[1], first[2], channels);
        if (first[3] != second[3])
        {
            var padded = first[3] < second[3] ? first : second;
            bytes = checked(bytes + BatchPayload(padded[0], padded[1], padded[2], channels));
        }
        if (first[1] != second[1] || first[2] != second[2])
            bytes = checked(bytes + BatchPayload(second[0], first[1], first[2], channels));
        if (bytes > maxAllocationBytes)
            throw new InvalidOperationException($"Image batch allocation requires {bytes} payload bytes, exceeding the explicit per-call budget of {maxAllocationBytes} bytes.");
    }

    private static long BatchPayload(long batch, long height, long width, long channels) =>
        checked(batch * height * width * channels * sizeof(float));
}
