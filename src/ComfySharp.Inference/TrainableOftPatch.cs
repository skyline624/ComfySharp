using System.Collections.ObjectModel;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Owned Float32 OFTDiff leaves and the frozen source Cayley rotation.
/// Alpha is exported as a parameter but the constraint is captured at construction;
/// changing alpha later does not change the rotation or give alpha a gradient.</summary>
public sealed class TrainableOftPatch : TrainableWeightPatch
{
    private sealed class Shared(Dictionary<string, Tensor> values, double constraint)
    {
        internal readonly object Gate = new();
        internal readonly IReadOnlyDictionary<string, Tensor> Named = new ReadOnlyDictionary<string, Tensor>(values);
        internal readonly IReadOnlyList<Tensor> Parameters = Array.AsReadOnly(values.Values.ToArray());
        internal readonly double Constraint = constraint;
        internal int Owners = 1;
    }
    private readonly Shared shared;
    private bool disposed;
    private TrainableOftPatch(Shared shared) => this.shared = shared;

    public TrainableOftPatch(Tensor blocks, double alpha = 1, Tensor? rescale = null)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        if (!double.IsFinite(alpha) || !float.IsFinite((float)alpha)) throw new ArgumentOutOfRangeException(nameof(alpha));
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var noGrad = no_grad();
        var original = new Dictionary<string, Tensor>(StringComparer.Ordinal) { { "oft_blocks", blocks } };
        if (rescale is not null) original.Add("rescale", rescale);
        foreach (var (name, value) in original)
        {
            if (value.dtype != ScalarType.Float32 || value.is_sparse || !InferenceDevice.IsSupported(value.device_type) ||
                value.shape.Any(n => n <= 0) || !value.isfinite().all().item<bool>())
                throw new ArgumentException("OFT requires finite dense Float32 parameters: " + name);
            InferenceDevice.RequireSame(blocks.device, value, name);
        }
        if (blocks.dim() != 3 || blocks.shape[1] != blocks.shape[2])
            throw new ArgumentException("OFT requires a nonempty batch of square blocks.", nameof(blocks));
        var snapshots = original.ToDictionary(p => p.Key, p => p.Value.detach().clone().requires_grad_(), StringComparer.Ordinal);
        // The training factory enables all leaves; OFTDiff still uses a captured float for its constraint.
        snapshots.Add("alpha", tensor((float)alpha, device: blocks.device).requires_grad_());
        shared = new(snapshots, alpha);
        foreach (var value in snapshots.Values) value.DetachFromDisposeScope();
    }

    public IReadOnlyDictionary<string, Tensor> NamedParameters { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Named; } } }
    public override IReadOnlyList<Tensor> Parameters { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Parameters; } } }
    public override TrainableOftPatch Retain() { lock (shared.Gate) { ThrowIfDisposed(); shared.Owners = checked(shared.Owners + 1); return new(shared); } }

    private Tensor Rotation()
    {
        var blocks = shared.Named["oft_blocks"];
        var identity = eye(blocks.shape[1], device: blocks.device);
        var q = blocks - blocks.transpose(1, 2);
        // Training tests truthiness, unlike inference's alpha > 0. Preserve negative constraints.
        if (shared.Constraint != 0)
        {
            var norm = q.norm() + 1e-8;
            if ((norm > shared.Constraint).item<bool>()) q = q * shared.Constraint / norm;
        }
        return (identity + q).matmul((identity - q).to_type(ScalarType.Float32).inverse());
    }
    private void ValidateTarget(Tensor value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (value.is_sparse || value.dim() < 2 || value.dtype is not (ScalarType.Float32 or ScalarType.Float16 or ScalarType.BFloat16))
            throw new ArgumentException("OFT requires a dense floating-point target.");
        InferenceDevice.RequireSame(shared.Parameters[0].device, value, nameof(value));
    }
    public Tensor Apply(Tensor weight) => Apply(weight, default);
    internal override Tensor Apply(Tensor weight, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested(); using var owner = Retain(); using var scope = NewDisposeScope();
        ValidateTarget(weight);
        var blocks = shared.Named["oft_blocks"];
        if (weight.shape[0] != checked(blocks.shape[0] * blocks.shape[1])) throw new ArgumentException("OFT block channels differ from target.");
        var rotation = Rotation();
        var grouped = weight.to_type(rotation.dtype).unflatten(0, new[] { blocks.shape[0], blocks.shape[1] });
        var result = einsum("knm,kn...->km...", rotation, grouped).flatten(0, 1);
        if (shared.Named.TryGetValue("rescale", out var rescale)) result = rescale * result;
        result = result.to_type(weight.dtype);
        cancellationToken.ThrowIfCancellationRequested(); return result.MoveToOuterDisposeScope();
    }

    /// <summary>Frozen OFTDiff.g with explicit module kind (a rank-three linear output is not a convolution).
    /// Float32 activations are implemented; reduced-precision bypass requires separate native qualification.</summary>
    public Tensor ApplyOutput(Tensor output, bool isConvolution, double multiplier = 1, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); using var owner = Retain(); using var scope = NewDisposeScope();
        ValidateTarget(output);
        if (!double.IsFinite(multiplier)) throw new ArgumentOutOfRangeException(nameof(multiplier));
        if (output.dtype != ScalarType.Float32) throw new NotSupportedException("Trainable OFT bypass currently requires Float32 activations.");
        if (isConvolution && output.dim() < 3) throw new ArgumentException("OFT convolution output requires at least three dimensions.");
        var blocks = shared.Named["oft_blocks"];
        var y = isConvolution ? output.transpose(1, -1) : output;
        if (y.shape[^1] != checked(blocks.shape[0] * blocks.shape[1])) throw new ArgumentException("OFT block channels differ from module output.");
        var rotation = Rotation() * multiplier + (1 - multiplier) * eye(blocks.shape[1], device: output.device);
        var grouped = y.reshape(y.shape.SkipLast(1).Concat(new[] { blocks.shape[0], blocks.shape[1] }).ToArray());
        var result = einsum("knm,...kn->...km", rotation, grouped).reshape(y.shape);
        if (shared.Named.TryGetValue("rescale", out var rescale)) result = result * rescale.view(-1);
        if (isConvolution) result = result.transpose(1, -1);
        cancellationToken.ThrowIfCancellationRequested(); return result.MoveToOuterDisposeScope();
    }
    internal override Tensor ApplyBypass(Tensor input, Tensor baseOutput, IReadOnlyList<long>? kernelSize, long stride, long padding)
    {
        using var scope = NewDisposeScope();
        // The frozen injection calls g(base_out + h(x, base_out)); OFT.h returns zeros_like(base_out).
        return ApplyOutput(baseOutput + zeros_like(baseOutput), kernelSize is not null).MoveToOuterDisposeScope();
    }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
    public override void Dispose() { lock (shared.Gate) { if (disposed) return; disposed = true; if (--shared.Owners == 0) foreach (var value in shared.Parameters) value.Dispose(); } }
}
