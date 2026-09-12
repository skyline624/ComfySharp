using TorchSharp;

namespace ComfySharp.Inference;

/// <summary>An independently disposable owner of an immutable shared Float32 bank on one CPU/CUDA device.</summary>
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
    public torch.Device Device
    {
        get { lock (bank.Gate) { ObjectDisposedException.ThrowIf(disposed, this); return bank.Tensors.Values.First().device; } }
    }

    public ClipWeightSet To(torch.Device device, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); device = InferenceDevice.Validate(device);
        using var source = Retain();
        if (InferenceDevice.Same(source.Device, device)) return source.Retain();
        using var scope = torch.NewDisposeScope(); using var noGrad = torch.no_grad();
        var copies = new Dictionary<torch.Tensor, torch.Tensor>(ReferenceEqualityComparer.Instance);
        var tensors = new Dictionary<string, torch.Tensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in source.bank.Tensors)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!copies.TryGetValue(tensor, out var copy)) { copy = tensor.to(device, copy: true); copies.Add(tensor, copy); }
            tensors.Add(name, copy);
        }
        cancellationToken.ThrowIfCancellationRequested();
        return FromOwnedTensors(Config, tensors, HasProjection);
    }

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
            if (tensor.dtype != torch.ScalarType.Float32 || !InferenceDevice.IsSupported(tensor.device_type) || tensor.is_sparse || !tensor.is_contiguous())
                throw new InvalidDataException($"CLIP weight '{name}' must be contiguous dense CPU or CUDA Float32.");
            if (tensor.requires_grad) throw new InvalidDataException($"CLIP weight '{name}' must be frozen (requires_grad=false).");
        }
        foreach (string name in schema.Keys)
            if ((requireProjection || name != ClipWeightSchema.Projection) && !snapshot.ContainsKey(name))
                throw new InvalidDataException($"Missing CLIP weight '{name}'.");
        var result = new ClipWeightSet(new(config, snapshot));
        var device = snapshot.Values.First().device;
        if (snapshot.Values.Any(t => !InferenceDevice.Same(device, t.device))) throw new InvalidDataException("A CLIP bank cannot mix devices.");
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

    public ClipWeightSet WithLora(IReadOnlyDictionary<string,LoraWeightPatch> patches,
        long maxPatchedWeightBytes=512L*1024*1024,CancellationToken cancellationToken=default)
    {
        cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(patches);
        if(maxPatchedWeightBytes<0)throw new ArgumentOutOfRangeException(nameof(maxPatchedWeightBytes));
        using var source=Retain();using var scope=torch.NewDisposeScope();using var noGrad=torch.no_grad();
        var selected=new Dictionary<string,LoraWeightPatch>(StringComparer.Ordinal);
        try
        {
            long bytes=0;
            foreach(var(name,patch) in patches)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ArgumentNullException.ThrowIfNull(patch);
                if(!source.bank.Tensors.TryGetValue(name,out var tensor))throw new InvalidDataException($"Unknown canonical CLIP patch target '{name}'.");
                bytes=checked(bytes+tensor.numel()*4);
                if(bytes>maxPatchedWeightBytes)throw new NotSupportedException("Patched resident weights exceed the configured byte allowance; temporary math tensors are additional.");
                selected.Add(name,patch.Retain());
            }
            var owned=new Dictionary<string,torch.Tensor>(StringComparer.Ordinal);
            foreach(var(name,tensor) in source.bank.Tensors)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if(selected.TryGetValue(name,out var patch))
                {using var result=patch.Apply(tensor,cancellationToken);owned.Add(name,result.alias());}
                else owned.Add(name,tensor.alias());
            }
            cancellationToken.ThrowIfCancellationRequested();return FromOwnedTensors(Config,owned,HasProjection);
        }
        finally{foreach(var patch in selected.Values)patch.Dispose();}
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
