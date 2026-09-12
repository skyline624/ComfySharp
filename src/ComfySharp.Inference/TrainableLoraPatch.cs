using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Owns two Float32 leaf parameters for ordinary LoRA training.
/// Callers must serialize forward/backward/updates. Borrowed parameters must not be disposed,
/// resized or moved; use a retained owner to keep them alive across an optimization step.</summary>
public sealed class TrainableLoraPatch : IDisposable
{
    private sealed class Shared(Tensor up, Tensor down, double alpha)
    {
        internal readonly object Gate = new();
        internal readonly Tensor Up = up, Down = down;
        internal readonly double Alpha = alpha;
        internal int Owners = 1;
    }
    private readonly Shared shared;
    private bool disposed;
    private TrainableLoraPatch(Shared shared) => this.shared = shared;

    public TrainableLoraPatch(Tensor up, Tensor down, double alpha)
    {
        ArgumentNullException.ThrowIfNull(up); ArgumentNullException.ThrowIfNull(down);
        if (!double.IsFinite(alpha)) throw new ArgumentOutOfRangeException(nameof(alpha));
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var noGrad = no_grad();
        foreach (var tensor in new[] { up, down })
            if (tensor.dtype != ScalarType.Float32 || !InferenceDevice.IsSupported(tensor.device_type) || tensor.is_sparse ||
                tensor.dim() != 2 || tensor.shape.Any(d => d <= 0) || !tensor.isfinite().all().item<bool>())
                throw new ArgumentException("Trainable LoRA factors require finite dense CPU/CUDA Float32 matrices.");
        InferenceDevice.RequireSame(up.device, down, nameof(down));
        if (up.shape[1] != down.shape[0]) throw new ArgumentException("LoRA factor ranks must agree.");
        var copiedUp = up.detach().clone().requires_grad_();
        var copiedDown = down.detach().clone().requires_grad_();
        shared = new(copiedUp, copiedDown, alpha);
        copiedUp.DetachFromDisposeScope(); copiedDown.DetachFromDisposeScope();
    }

    public Tensor Up { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Up; } } }
    public Tensor Down { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Down; } } }
    public double Alpha { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Alpha; } } }
    public TrainableLoraPatch Retain()
    {
        lock (shared.Gate) { ThrowIfDisposed(); shared.Owners = checked(shared.Owners + 1); return new(shared); }
    }
    internal Tensor Apply(Tensor weight, CancellationToken cancellationToken)
    {
        using var operation = Retain();
        return LoraMath.Apply(weight, operation.Up, operation.Down, alpha: operation.Alpha, cancellationToken: cancellationToken);
    }
    /// <summary>Independent frozen adapter snapshot for ordinary inference. Later training cannot mutate it.</summary>
    public LoraWeightPatch Snapshot()
    {
        using var operation = Retain();
        return new(operation.Up, operation.Down, alpha: operation.Alpha);
    }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
    public void Dispose()
    {
        lock (shared.Gate)
        {
            if (disposed) return; disposed = true;
            if (--shared.Owners == 0) { shared.Up.Dispose(); shared.Down.Dispose(); }
        }
    }
}
