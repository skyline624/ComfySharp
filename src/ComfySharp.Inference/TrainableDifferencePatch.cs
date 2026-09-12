using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Owned additive Float32 leaf for source BiasDiff: a one-dimensional norm weight or bias.</summary>
public sealed class TrainableDifferencePatch : TrainableWeightPatch
{
    private sealed class Shared(Tensor difference)
    {
        internal readonly object Gate = new();
        internal readonly Tensor Difference = difference;
        internal int Owners = 1;
    }
    private readonly Shared shared;
    private bool disposed;
    private TrainableDifferencePatch(Shared shared) => this.shared = shared;
    public TrainableDifferencePatch(Tensor difference)
    {
        ArgumentNullException.ThrowIfNull(difference); NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        if (difference.dtype != ScalarType.Float32 || !InferenceDevice.IsSupported(difference.device_type) || difference.is_sparse ||
            difference.dim() != 1 || difference.numel() <= 0 || !difference.isfinite().all().item<bool>())
            throw new ArgumentException("A trainable difference requires a finite dense CPU/CUDA Float32 vector.", nameof(difference));
        var copy = difference.detach().clone().requires_grad_();
        shared = new(copy); copy.DetachFromDisposeScope();
    }
    public Tensor Difference { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Difference; } } }
    public override IReadOnlyList<Tensor> Parameters => new[] { Difference };
    public override TrainableDifferencePatch Retain()
    {
        lock (shared.Gate) { ThrowIfDisposed(); shared.Owners = checked(shared.Owners + 1); return new(shared); }
    }
    public Tensor Apply(Tensor weight) => Apply(weight, default);
    internal override Tensor Apply(Tensor weight, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(weight);
        using var operation = Retain(); using var scope = NewDisposeScope();
        if (weight.dtype != ScalarType.Float32 || weight.is_sparse || !weight.shape.SequenceEqual(operation.Difference.shape))
            throw new ArgumentException("Difference and target must have identical dense Float32 vector shapes.", nameof(weight));
        InferenceDevice.RequireSame(operation.Difference.device, weight, nameof(weight));
        var result = weight + operation.Difference;
        cancellationToken.ThrowIfCancellationRequested(); return result.MoveToOuterDisposeScope();
    }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
    public override void Dispose()
    {
        lock (shared.Gate) { if (disposed) return; disposed = true; if (--shared.Owners == 0) shared.Difference.Dispose(); }
    }
}
