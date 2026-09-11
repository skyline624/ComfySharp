using TorchSharp;

namespace ComfySharp.Inference;

/// <summary>An independently disposable owner of an immutable shared CPU/Float32 parameter bank.</summary>
public sealed class ClipWeightSet : IDisposable
{
    private sealed class Bank(ClipTextConfig config, Dictionary<string, torch.Tensor> tensors)
    {
        internal readonly object Gate = new();
        internal readonly ClipTextConfig Config = config;
        internal readonly Dictionary<string, torch.Tensor> Tensors = tensors;
        internal int Owners = 1;
    }

    private readonly Bank bank;
    private bool disposed;
    private ClipWeightSet(Bank bank) => this.bank = bank;
    public ClipTextConfig Config => bank.Config;
    public bool HasProjection => bank.Tensors.ContainsKey(ClipWeightSchema.Projection);

    /// <summary>Takes ownership only after all validation succeeds. On success callers must neither mutate nor dispose
    /// the transferred tensor wrappers. The bank detaches them from ambient dispose scopes; no parameter copies are made.</summary>
    public static ClipWeightSet FromOwnedTensors(ClipTextConfig config,
        IReadOnlyDictionary<string, torch.Tensor> tensors, bool requireProjection = true)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var schema = ClipWeightSchema.Describe(config);
        var snapshot = new Dictionary<string, torch.Tensor>(tensors, StringComparer.Ordinal);
        foreach (var (name, tensor) in snapshot)
        {
            if (!schema.TryGetValue(name, out var shape)) throw new InvalidDataException($"Unknown canonical CLIP weight '{name}'.");
            if (tensor is null || tensor.IsInvalid) throw new InvalidDataException($"CLIP weight '{name}' is null or disposed.");
            ClipWeightSchema.CheckShape(name, tensor.shape, shape);
            if (tensor.dtype != torch.ScalarType.Float32 || tensor.device_type != DeviceType.CPU || tensor.is_sparse || !tensor.is_contiguous())
                throw new InvalidDataException($"CLIP weight '{name}' must be contiguous dense CPU/Float32.");
            if (tensor.requires_grad) throw new InvalidDataException($"CLIP weight '{name}' must be frozen (requires_grad=false).");
        }
        foreach (string name in schema.Keys)
            if ((requireProjection || name != ClipWeightSchema.Projection) && !snapshot.ContainsKey(name))
                throw new InvalidDataException($"Missing CLIP weight '{name}'.");
        var result = new ClipWeightSet(new(config, snapshot));
        foreach (var tensor in snapshot.Values) tensor.DetachFromDisposeScope();
        return result;
    }

    public ClipWeightSet Retain()
    {
        lock (bank.Gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var owner = new ClipWeightSet(bank);
            bank.Owners = checked(bank.Owners + 1);
            return owner;
        }
    }

    // Borrow only while holding a retained bank owner for the entire operation.
    internal torch.Tensor GetTensor(string name)
    {
        lock (bank.Gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return bank.Tensors.TryGetValue(name, out var tensor) ? tensor :
                throw new InvalidOperationException($"CLIP weight '{name}' is unavailable; projected pooling requires a projection weight.");
        }
    }

    public void Dispose()
    {
        lock (bank.Gate)
        {
            if (disposed) return;
            disposed = true;
            if (--bank.Owners != 0) return;
            foreach (var tensor in new HashSet<torch.Tensor>(bank.Tensors.Values, ReferenceEqualityComparer.Instance))
                tensor.Dispose();
        }
    }
}
