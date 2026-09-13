using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Owns two Float32 leaf factors and optionally a trainable Float32 alpha for ordinary or bypass LoRA training.
/// Callers must serialize forward/backward/updates. Borrowed parameters must not be disposed,
/// resized or moved; use a retained owner to keep them alive across an optimization step.</summary>
public sealed class TrainableLoraPatch : TrainableWeightPatch
{
    private sealed class Shared(Tensor up, Tensor down, double alpha, Tensor? alphaParameter)
    {
        internal readonly object Gate = new();
        internal readonly Tensor Up = up, Down = down;
        internal readonly double Alpha = alpha;
        internal readonly Tensor? AlphaParameter = alphaParameter;
        internal int Owners = 1;
    }
    private readonly Shared shared;
    private bool disposed;
    private TrainableLoraPatch(Shared shared) => this.shared = shared;

    public TrainableLoraPatch(Tensor up, Tensor down, double alpha, bool trainAlpha = false)
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
        if (trainAlpha && !float.IsFinite((float)alpha)) throw new ArgumentOutOfRangeException(nameof(alpha));
        var alphaParameter = trainAlpha ? tensor((float)alpha, device: up.device).requires_grad_() : null;
        shared = new(copiedUp, copiedDown, alpha, alphaParameter);
        copiedUp.DetachFromDisposeScope(); copiedDown.DetachFromDisposeScope();
        alphaParameter?.DetachFromDisposeScope();
    }

    public Tensor Up { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Up; } } }
    public Tensor Down { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Down; } } }
    public double Alpha { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.AlphaParameter?.item<float>() ?? shared.Alpha; } } }
    public Tensor? AlphaParameter { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.AlphaParameter; } } }
    public override IReadOnlyList<Tensor> Parameters => AlphaParameter is { } alpha ? new[] { alpha, Up, Down } : new[] { Up, Down };
    public override TrainableLoraPatch Retain()
    {
        lock (shared.Gate) { ThrowIfDisposed(); shared.Owners = checked(shared.Owners + 1); return new(shared); }
    }
    internal override Tensor Apply(Tensor weight, CancellationToken cancellationToken)
    {
        using var operation = Retain();
        if (operation.AlphaParameter is { } alpha)
        {
            cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(weight);
            using var scope = NewDisposeScope();
            if (weight.dtype != ScalarType.Float32 || weight.is_sparse || weight.dim() < 2 ||
                weight.shape[0] != operation.Up.shape[0] || weight.numel() / weight.shape[0] != operation.Down.shape[1])
                throw new ArgumentException("Trainable-alpha LoRA target shape or dtype differs.", nameof(weight));
            InferenceDevice.RequireSame(operation.Up.device, weight, nameof(weight));
            var difference = operation.Up.matmul(operation.Down).reshape(weight.shape);
            var result = weight + (alpha / operation.Down.shape[0]) * difference;
            cancellationToken.ThrowIfCancellationRequested(); return result.MoveToOuterDisposeScope();
        }
        return LoraMath.Apply(weight, operation.Up, operation.Down, alpha: operation.Alpha, cancellationToken: cancellationToken);
    }
    /// <summary>Frozen LoraDiff.h for new two-factor Float32 SD adapters, preserving trainable alpha.</summary>
    internal override Tensor ApplyBypass(Tensor input, Tensor baseOutput, IReadOnlyList<long>? kernelSize, long stride, long padding)
    {
        using var operation = Retain(); using var scope = NewDisposeScope();
        if (input.dtype != ScalarType.Float32 || baseOutput.dtype != ScalarType.Float32 || input.is_sparse || baseOutput.is_sparse)
            throw new ArgumentException("Trainable SD bypass requires dense Float32 activations.");
        InferenceDevice.RequireSame(input.device, baseOutput, nameof(baseOutput));
        InferenceDevice.RequireSame(input.device, operation.Up, nameof(Up));
        Tensor result;
        if (kernelSize is null)
            result = nn.functional.linear(nn.functional.linear(input, operation.Down), operation.Up);
        else
        {
            if (input.dim() != 4 || kernelSize.Count != 2 || kernelSize.Any(v => v <= 0) || stride < 1 || padding < 0)
                throw new ArgumentException("Trainable SD bypass requires valid Conv2d geometry.");
            var down = operation.Down.reshape(operation.Down.shape[0], input.shape[1], kernelSize[0], kernelSize[1]);
            var up = operation.Up.reshape(operation.Up.shape[0], operation.Up.shape[1], 1, 1);
            result = nn.functional.conv2d(nn.functional.conv2d(input, down, strides: new[] { stride, stride }, padding: new[] { padding, padding }), up);
        }
        if (!result.shape.SequenceEqual(baseOutput.shape)) throw new ArgumentException("Trainable bypass output shape differs from the base module.");
        // LoraDiff.h computes a tensor scale, including alpha's gradient. Do not call item().
        var scale = operation.AlphaParameter is { } alpha ? alpha / operation.Down.shape[0]
            : tensor((float)operation.Alpha, device: input.device) / operation.Down.shape[0];
        return (baseOutput + result * (scale * 1.0)).MoveToOuterDisposeScope();
    }

    /// <summary>Independent frozen adapter snapshot. Later training cannot mutate it.</summary>
    public LoraWeightPatch Snapshot()
    {
        using var operation = Retain();
        return new(operation.Up, operation.Down, alpha: operation.Alpha);
    }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
    public override void Dispose()
    {
        lock (shared.Gate)
        {
            if (disposed) return; disposed = true;
            if (--shared.Owners == 0) { shared.Up.Dispose(); shared.Down.Dispose(); shared.AlphaParameter?.Dispose(); }
        }
    }
}
