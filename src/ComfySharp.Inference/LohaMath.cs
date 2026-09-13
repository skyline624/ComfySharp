using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Frozen LoHaAdapter inference arithmetic, including DoRA and linear/Conv2d bypass.
/// Borrowed inputs are never mutated. Trainable LohaDiff has a separate custom backward.</summary>
public static class LohaMath
{
    internal static long[] ReconstructedShape(IReadOnlyList<long> a, IReadOnlyList<long> b,
        IReadOnlyList<long> c, IReadOnlyList<long> d, IReadOnlyList<long>? t1, IReadOnlyList<long>? t2)
    {
        if (new[] { a, b, c, d }.Any(s => s.Count != 2 || s.Any(n => n <= 0)))
            throw new ArgumentException("LoHa side factors must be nonempty matrices.");
        if ((t1 is null) != (t2 is null)) throw new ArgumentException("LoHa Tucker requires both cores.");
        long[] left, right;
        if (t1 is null)
        {
            if (a[1] != b[0] || c[1] != d[0]) throw new ArgumentException("LoHa matrix ranks differ within a side.");
            left = [a[0], b[1]]; right = [c[0], d[1]];
        }
        else
        {
            // Inference uses the frozen explicit ijkl equation, unlike LohaDiff's general ellipsis.
            if (t1.Count != 4 || t2!.Count != 4 || t1.Concat(t2).Any(n => n <= 0) ||
                t1[0] != a[0] || t1[1] != b[0] || t2[0] != c[0] || t2[1] != d[0])
                throw new ArgumentException("LoHa inference Tucker cores must have four compatible dimensions.");
            left = [a[1], b[1], t1[2], t1[3]]; right = [c[1], d[1], t2[2], t2[3]];
        }
        // Source multiplies reconstructed tensors using ordinary broadcasting before reshaping.
        var shape = new long[left.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            if (left[i] != right[i] && left[i] != 1 && right[i] != 1) throw new ArgumentException("LoHa products cannot broadcast.");
            shape[i] = Math.Max(left[i], right[i]);
        }
        return shape;
    }
    private static Tensor Difference(Tensor a, Tensor b, Tensor c, Tensor d, Tensor? t1, Tensor? t2)
    {
        using var scope = NewDisposeScope();
        var left = t1 is null ? mm(a, b) : einsum("ijkl,jr,ip->prkl", t1, b, a);
        var right = t2 is null ? mm(c, d) : einsum("ijkl,jr,ip->prkl", t2, d, c);
        return (left * right).MoveToOuterDisposeScope();
    }
    private static void Validate(Tensor reference, Tensor a, Tensor b, Tensor c, Tensor d, Tensor? t1, Tensor? t2)
    {
        if (reference.dtype != ScalarType.Float32 || reference.is_sparse || !InferenceDevice.IsSupported(reference.device_type))
            throw new ArgumentException("LoHa inference requires a dense Float32 target or activation.");
        foreach (var value in new[] { a, b, c, d, t1, t2 }.OfType<Tensor>())
        {
            if (value.dtype != ScalarType.Float32 || value.is_sparse) throw new ArgumentException("LoHa arithmetic requires dense Float32 factors.");
            InferenceDevice.RequireSame(reference.device, value, "factor");
        }
        ReconstructedShape(a.shape, b.shape, c.shape, d.shape, t1?.shape, t2?.shape);
    }
    public static Tensor Apply(Tensor weight, Tensor a, Tensor b, Tensor c, Tensor d,
        double strength = 1, double? alpha = null, Tensor? t1 = null, Tensor? t2 = null,
        Tensor? doraScale = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); NativeRuntimeBootstrap.Initialize();
        ArgumentNullException.ThrowIfNull(weight); ArgumentNullException.ThrowIfNull(a); ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c); ArgumentNullException.ThrowIfNull(d);
        if (!double.IsFinite(strength) || alpha is not null && !double.IsFinite(alpha.Value)) throw new ArgumentOutOfRangeException(nameof(strength));
        Validate(weight, a, b, c, d, t1, t2);
        if (weight.dim() < 2 || weight.shape.Any(n => n <= 0)) throw new ArgumentException("LoHa target must be a nonempty weight.");
        using var scope = NewDisposeScope();
        var diff = Difference(a, b, c, d, t1, t2);
        if (diff.numel() != weight.numel()) throw new ArgumentException("LoHa reconstruction cannot reshape to the target weight.");
        var result = LoraMath.ApplyDifference(weight, diff.reshape(weight.shape), strength, alpha is null ? 1 : alpha.Value / b.shape[0], doraScale);
        if (!result.isfinite().all().item<bool>()) throw new ArithmeticException("LoHa produced nonfinite weights.");
        cancellationToken.ThrowIfCancellationRequested(); return result.DetachFromDisposeScope();
    }
    public static Tensor ApplyBypass(Tensor input, Tensor baseOutput, Tensor a, Tensor b, Tensor c, Tensor d,
        double strength = 1, double? alpha = null, Tensor? t1 = null, Tensor? t2 = null,
        IReadOnlyList<long>? kernelSize = null, long stride = 1, long padding = 0, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); NativeRuntimeBootstrap.Initialize();
        ArgumentNullException.ThrowIfNull(input); ArgumentNullException.ThrowIfNull(baseOutput);
        ArgumentNullException.ThrowIfNull(a); ArgumentNullException.ThrowIfNull(b); ArgumentNullException.ThrowIfNull(c); ArgumentNullException.ThrowIfNull(d);
        if (!double.IsFinite(strength) || alpha is not null && !double.IsFinite(alpha.Value)) throw new ArgumentOutOfRangeException(nameof(strength));
        Validate(input, a, b, c, d, t1, t2); InferenceDevice.RequireSame(input.device, baseOutput, nameof(baseOutput));
        if (baseOutput.dtype != ScalarType.Float32 || baseOutput.is_sparse || stride < 1 || padding < 0)
            throw new ArgumentException("LoHa bypass requires dense Float32 output and valid module geometry.");
        using var scope = NewDisposeScope();
        // Source h scales the full difference BEFORE the linear/convolution operation.
        var diff = Difference(a, b, c, d, t1, t2) * ((alpha is null ? 1 : alpha.Value / b.shape[0]) * strength);
        Tensor output;
        if (kernelSize is null)
        {
            if (diff.dim() != 2) throw new ArgumentException("Linear LoHa bypass requires a matrix reconstruction.");
            output = nn.functional.linear(input, diff);
        }
        else
        {
            if (input.dim() != 4 || kernelSize.Count != 2 || kernelSize.Any(n => n <= 0)) throw new ArgumentException("SD LoHa bypass requires Conv2d geometry.");
            if (diff.dim() == 2) diff = diff.reshape(diff.shape[0], input.shape[1], kernelSize[0], kernelSize[1]);
            output = nn.functional.conv2d(input, diff, strides: new[] { stride, stride }, padding: new[] { padding, padding });
        }
        if (!output.shape.SequenceEqual(baseOutput.shape)) throw new ArgumentException("LoHa bypass output differs from the base module shape.");
        cancellationToken.ThrowIfCancellationRequested(); return (baseOutput + output).MoveToOuterDisposeScope();
    }
}
