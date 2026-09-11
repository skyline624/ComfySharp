using TorchSharp;

namespace ComfySharp.Inference;

/// <summary>One retained owner. Tensor wrappers are borrowed only while an operation owns a lease.</summary>
internal sealed class CpuModelWeightBank : IDisposable
{
    private sealed class Shared(Dictionary<string, torch.Tensor> tensors)
    {
        internal readonly object Gate = new();
        internal readonly Dictionary<string, torch.Tensor> Tensors = tensors;
        internal int Owners = 1;
    }
    private readonly Shared shared;
    private bool disposed;
    private CpuModelWeightBank(Shared shared) => this.shared = shared;

    internal static CpuModelWeightBank Create(IReadOnlyDictionary<string, IReadOnlyList<long>> schema,
        IReadOnlyDictionary<string, torch.Tensor> tensors, Action<torch.Tensor>? normalized = null)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var snapshot = new Dictionary<string, torch.Tensor>(tensors, StringComparer.Ordinal);
        foreach (var (name, tensor) in snapshot)
        {
            if (!schema.TryGetValue(name, out var shape)) throw new InvalidDataException($"Unknown canonical weight '{name}'.");
            if (tensor is null || tensor.IsInvalid) throw new InvalidDataException($"Weight '{name}' is null or disposed.");
            ModelWeightSchemaBuilder.CheckShape(name, tensor.shape, shape);
            if (tensor.dtype != torch.ScalarType.Float32 || tensor.device_type != DeviceType.CPU || tensor.is_sparse || !tensor.is_contiguous())
                throw new InvalidDataException($"Weight '{name}' must be contiguous dense CPU/Float32.");
            if (tensor.requires_grad) throw new InvalidDataException($"Weight '{name}' must be frozen (requires_grad=false).");
        }
        foreach (string name in schema.Keys)
            if (!snapshot.ContainsKey(name)) throw new InvalidDataException($"Missing weight '{name}'.");
        // Preserve caller ownership until every required allocation succeeds. Managed-backed tensors
        // and offset views can select different CPU reduction kernels despite identical F32 bits.
        using var scope = torch.NewDisposeScope();
        using var noGrad = torch.no_grad();
        var replacements = new Dictionary<torch.Tensor, torch.Tensor>(ReferenceEqualityComparer.Instance);
        foreach (var tensor in new HashSet<torch.Tensor>(snapshot.Values, ReferenceEqualityComparer.Instance))
        {
            if (IsAligned(tensor)) continue;
            var clone = tensor.clone();
            if (!IsAligned(clone)) throw new NotSupportedException("The CPU allocator did not provide 64-byte aligned weight storage.");
            replacements.Add(tensor, clone);
            normalized?.Invoke(clone);
        }
        var canonical = snapshot.ToDictionary(p => p.Key,
            p => replacements.TryGetValue(p.Value, out var replacement) ? replacement : p.Value, StringComparer.Ordinal);
        var result = new CpuModelWeightBank(new Shared(canonical));
        foreach (var tensor in canonical.Values) tensor.DetachFromDisposeScope();
        // Commit: replaced wrappers were also transferred, and no longer own bank storage.
        foreach (var tensor in replacements.Keys) tensor.Dispose();
        return result;
    }

    internal static unsafe bool IsAligned(torch.Tensor tensor)
    {
        // Tensor.bytes is the supported contiguous CPU span; pin only to inspect its address.
        // No native address leaves this method and no payload byte is read or written.
        fixed (byte* address = tensor.bytes) return ((nuint)address & 63) == 0;
    }

    internal CpuModelWeightBank Retain()
    {
        lock (shared.Gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var owner = new CpuModelWeightBank(shared);
            shared.Owners = checked(shared.Owners + 1);
            return owner;
        }
    }

    internal torch.Tensor GetTensor(string name)
    {
        lock (shared.Gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return shared.Tensors.TryGetValue(name, out var tensor) ? tensor : throw new KeyNotFoundException($"Weight '{name}' is unavailable.");
        }
    }

    public void Dispose()
    {
        lock (shared.Gate)
        {
            if (disposed) return;
            disposed = true;
            if (--shared.Owners != 0) return;
            foreach (var tensor in new HashSet<torch.Tensor>(shared.Tensors.Values, ReferenceEqualityComparer.Instance)) tensor.Dispose();
        }
    }
}
