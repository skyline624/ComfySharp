using System.Collections.ObjectModel;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Owned Float32 LohaDiff leaves, including its frozen first-order Tucker backward.
/// Parameters are borrowed; serialize forward, updates, capture and disposal as for other adapters.</summary>
public sealed class TrainableLohaPatch : TrainableWeightPatch
{
    private sealed class Shared(Dictionary<string, Tensor> tensors)
    {
        internal readonly object Gate = new();
        internal readonly IReadOnlyDictionary<string, Tensor> Named = new ReadOnlyDictionary<string, Tensor>(tensors);
        internal readonly IReadOnlyList<Tensor> Parameters = Array.AsReadOnly(tensors.Values.ToArray());
        internal int Owners = 1;
    }
    private readonly Shared shared;
    private bool disposed;
    private TrainableLohaPatch(Shared shared) => this.shared = shared;

    public TrainableLohaPatch(Tensor w1a, Tensor w1b, Tensor w2a, Tensor w2b,
        double alpha = 1, Tensor? t1 = null, Tensor? t2 = null)
    {
        ArgumentNullException.ThrowIfNull(w1a); ArgumentNullException.ThrowIfNull(w1b);
        ArgumentNullException.ThrowIfNull(w2a); ArgumentNullException.ThrowIfNull(w2b);
        if (!double.IsFinite(alpha) || !float.IsFinite((float)alpha)) throw new ArgumentOutOfRangeException(nameof(alpha));
        if ((t1 is null) != (t2 is null)) throw new ArgumentException("LoHa Tucker requires both cores.");
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var noGrad = no_grad();
        var originals = new Dictionary<string, Tensor>
        {
            ["hada_w1_a"] = w1a, ["hada_w1_b"] = w1b, ["hada_w2_a"] = w2a, ["hada_w2_b"] = w2b
        };
        if (t1 is not null) { originals.Add("hada_t1", t1); originals.Add("hada_t2", t2!); }
        foreach (var (name, value) in originals)
        {
            if (value.dtype != ScalarType.Float32 || !InferenceDevice.IsSupported(value.device_type) || value.is_sparse ||
                value.dim() < 2 || value.shape.Any(n => n <= 0) || !value.isfinite().all().item<bool>())
                throw new ArgumentException("Trainable LoHa requires finite dense Float32 factors: " + name);
            InferenceDevice.RequireSame(w1a.device, value, name);
        }
        ValidateGeometry(w1a.shape,w1b.shape,w2a.shape,w2b.shape,t1?.shape,t2?.shape);
        var snapshots = originals.ToDictionary(p => p.Key, p => p.Value.detach().clone().requires_grad_(), StringComparer.Ordinal);
        // Source setup enables requires_grad on alpha, but HadaWeight backward returns no alpha gradient.
        snapshots.Add("alpha", tensor((float)alpha, device: w1a.device).requires_grad_());
        shared = new(snapshots);
        foreach (var value in snapshots.Values) value.DetachFromDisposeScope();
    }

    internal static void ValidateGeometry(IReadOnlyList<long> a,IReadOnlyList<long> b,IReadOnlyList<long> c,IReadOnlyList<long> d,
        IReadOnlyList<long>? t1,IReadOnlyList<long>? t2,IReadOnlyList<long>? target=null)
    {
        if (new[] { a,b,c,d }.Any(value => value.Count != 2 || value.Any(n=>n<=0)))
            throw new ArgumentException("LoHa side factors must be matrices.");
        if((t1 is null)!=(t2 is null))throw new ArgumentException("LoHa Tucker requires both cores.");
        if (t1 is null)
        {
            if (a[1] != b[0] || c[1] != d[0] || a[0] != c[0] || b[1] != d[1])
                throw new ArgumentException("LoHa matrix products must have matching output shapes.");
        }
        else
        {
            if(t1.Count<2||t2!.Count<2||t1.Concat(t2).Any(n=>n<=0)||
                t1[0] != a[0] || t1[1] != b[0] || t2[0] != c[0] || t2[1] != d[0] ||
                a[1] != c[1] || b[1] != d[1] || !t1.Skip(2).SequenceEqual(t2.Skip(2)))
                throw new ArgumentException("LoHa Tucker core and factor shapes differ.");
            // The frozen backward returns each a-gradient using the opposite side's i dimension.
            if (a[0] != c[0])
                throw new NotSupportedException("Frozen LoHa Tucker backward cannot return valid a-gradients for different side ranks.");
        }
        long rows=a[t1 is null?0:1],spatial=t1 is null?1:t1.Skip(2).Aggregate(1L,(x,y)=>checked(x*y));
        if(target is not null&&(target.Count<2||target[0]!=rows||target.Aggregate(1L,(x,y)=>checked(x*y))!=checked(rows*b[1]*spatial)))
            throw new ArgumentException("LoHa reconstruction differs from the training target.");
    }

    public IReadOnlyDictionary<string, Tensor> NamedParameters { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Named; } } }
    public override IReadOnlyList<Tensor> Parameters { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Parameters; } } }
    public override TrainableLohaPatch Retain()
    {
        lock (shared.Gate) { ThrowIfDisposed(); shared.Owners = checked(shared.Owners + 1); return new(shared); }
    }
    public Tensor Apply(Tensor weight) => Apply(weight, default);
    internal override Tensor Apply(Tensor weight, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(weight);
        using var operation = Retain(); using var scope = NewDisposeScope();
        var p = operation.NamedParameters;
        var a = p["hada_w1_a"]; var b = p["hada_w1_b"]; var c = p["hada_w2_a"]; var d = p["hada_w2_b"];
        if (weight.is_sparse || weight.dim() < 2 || weight.dtype is not (ScalarType.Float32 or ScalarType.Float16 or ScalarType.BFloat16))
            throw new ArgumentException("LoHa target must be a dense floating-point weight.", nameof(weight));
        InferenceDevice.RequireSame(a.device, weight, nameof(weight));
        bool tucker = p.TryGetValue("hada_t1", out var t1);
        long rows = a.shape[tucker ? 1 : 0];
        long columns = b.shape[1];
        long spatial = tucker ? t1!.shape.Skip(2).Aggregate(1L, (x, y) => checked(x * y)) : 1;
        if (weight.shape[0] != rows || weight.numel() != checked(rows * columns * spatial))
            throw new ArgumentException("LoHa reconstruction does not match the target weight.", nameof(weight));
        Tensor first, second;
        if (!tucker) { first = a.matmul(b); second = c.matmul(d); }
        else
        {
            var t2 = p["hada_t2"];
            var aStopped = a.detach(); var cStopped = c.detach();
            first = einsum("ij...,jr,ip->pr...", t1!, b, aStopped);
            second = einsum("ij...,jr,ip->pr...", t2, d, cStopped);
            // Zero-valued terms supply the source's custom first-order a-gradients.
            // HadaWeightTucker.backward uses temp from side 2 for grad_w1u and side 1
            // for grad_w2u. Ordinary einsum autograd would silently change that behavior.
            first = first + einsum("ij...,jr,ip->pr...", t2.detach(), d.detach(), a - aStopped);
            second = second + einsum("ij...,jr,ip->pr...", t1!.detach(), b.detach(), c - cStopped);
        }
        var difference = first * second * (p["alpha"].detach() / b.shape[0]);
        var result = (weight.to_type(ScalarType.Float32) + difference.reshape(weight.shape)).to_type(weight.dtype);
        cancellationToken.ThrowIfCancellationRequested(); return result.MoveToOuterDisposeScope();
    }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
    public override void Dispose()
    {
        lock (shared.Gate)
        {
            if (disposed) return; disposed = true;
            if (--shared.Owners == 0) foreach (var value in shared.Parameters) value.Dispose();
        }
    }
}
