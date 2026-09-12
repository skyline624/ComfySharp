using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Frozen ordinary LoRA bypass: base output + up(mid(down(input))) * scale.
/// Uses module geometry only, never original weight values. Plain SD linear/Conv2d path.</summary>
public static class LoraBypassMath
{
    public static Tensor Apply(Tensor input, Tensor baseOutput, Tensor up, Tensor down,
        double strength = 1, double? alpha = null, Tensor? mid = null,
        IReadOnlyList<long>? kernelSize = null, long stride = 1, long padding = 0,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(input); ArgumentNullException.ThrowIfNull(baseOutput);
        ArgumentNullException.ThrowIfNull(up); ArgumentNullException.ThrowIfNull(down);
        if (!double.IsFinite(strength) || alpha is not null && !double.IsFinite(alpha.Value)) throw new ArgumentOutOfRangeException(nameof(strength));
        if (input.dtype != ScalarType.Float32 || baseOutput.dtype != ScalarType.Float32 || input.is_sparse || baseOutput.is_sparse)
            throw new ArgumentException("Plain SD bypass requires dense Float32 activations.");
        if (stride < 1 || padding < 0) throw new ArgumentOutOfRangeException(nameof(stride));
        InferenceDevice.RequireSame(input.device, baseOutput, nameof(baseOutput));
        foreach (var factor in new[] { up, down, mid }.OfType<Tensor>())
        {
            InferenceDevice.RequireSame(input.device, factor, "factor");
            if (factor.is_sparse || factor.dtype is not (ScalarType.Float32 or ScalarType.Float16 or ScalarType.BFloat16) || factor.dim() < 2 || factor.shape.Any(d => d <= 0))
                throw new ArgumentException("Bypass factors require nonempty dense floating tensors.");
        }
        using var scope = NewDisposeScope();
        var u = up.to_type(input.dtype); var d = down.to_type(input.dtype); var m = mid?.to_type(input.dtype);
        Tensor hidden, result;
        if (kernelSize is null)
        {
            if (u.dim() != 2 || d.dim() != 2 || m is not null && m.dim() != 2) throw new ArgumentException("Linear bypass requires matrix factors.");
            hidden = nn.functional.linear(input, d);
            if (m is not null) hidden = nn.functional.linear(hidden, m);
            result = nn.functional.linear(hidden, u);
        }
        else
        {
            if (input.dim() != 4 || kernelSize.Count != 2 || kernelSize.Any(d => d <= 0)) throw new ArgumentException("Conv2d bypass requires NCHW input and two kernel dimensions.");
            if (d.dim() == 2) d = d.reshape(d.shape[0], input.shape[1], kernelSize[0], kernelSize[1]);
            if (u.dim() == 2) u = u.reshape(u.shape[0], u.shape[1], 1, 1);
            if (m is not null && m.dim() == 2) m = m.reshape(m.shape[0], m.shape[1], 1, 1);
            if (m is null)
                hidden = nn.functional.conv2d(input, d, strides: new[] { stride, stride }, padding: new[] { padding, padding });
            else
            {
                hidden = nn.functional.conv2d(input, d);
                hidden = nn.functional.conv2d(hidden, m, strides: new[] { stride, stride }, padding: new[] { padding, padding });
            }
            result = nn.functional.conv2d(hidden, u);
        }
        if (!result.shape.SequenceEqual(baseOutput.shape)) throw new ArgumentException("Bypass output shape differs from the base module output.");
        // Alpha is a Python scalar after load_lora's item(), so compute its scalar scale before multiplication.
        var combined = baseOutput + result * ((alpha is null ? 1.0 : alpha.Value / down.shape[0]) * strength);
        cancellationToken.ThrowIfCancellationRequested(); return combined.MoveToOuterDisposeScope();
    }
}
