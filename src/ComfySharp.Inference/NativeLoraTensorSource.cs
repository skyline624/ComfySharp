using System.Collections.ObjectModel;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Stable metadata and independent tensor reads for adapter loading. A load plan is
/// tied to this exact source instance. Sources and returned tensors have separate lifetimes.</summary>
public interface ILoraTensorSource
{
    IReadOnlyDictionary<string, SafeTensorInfo> Tensors { get; }
    Tensor ReadTensor(string name, CancellationToken cancellationToken = default);
}

/// <summary>Captures borrowed native adapter values into owned CPU storage, without serializing
/// a file. Callers serialize source mutation during capture. Metadata offsets are logical ranges.</summary>
public sealed class NativeLoraTensorSource : ILoraTensorSource, IDisposable
{
    private readonly object gate = new();
    private readonly Dictionary<string, Tensor> values;
    private readonly IReadOnlyDictionary<string, SafeTensorInfo> metadata;
    private bool disposed;
    public IReadOnlyDictionary<string, SafeTensorInfo> Tensors
    {
        get { lock (gate) { ObjectDisposedException.ThrowIf(disposed, this); return metadata; } }
    }

    public NativeLoraTensorSource(IReadOnlyDictionary<string, Tensor> borrowed,
        long maxSnapshotBytes = 512L * 1024 * 1024, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(borrowed); cancellationToken.ThrowIfCancellationRequested();
        if (maxSnapshotBytes < 0) throw new ArgumentOutOfRangeException(nameof(maxSnapshotBytes));
        if (borrowed.Count > 100_000) throw new NotSupportedException("Too many adapter tensors.");
        var infos = new Dictionary<string, SafeTensorInfo>(StringComparer.Ordinal); long offset = 0;
        foreach (var (key, value) in borrowed)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ArgumentNullException.ThrowIfNull(key); ArgumentNullException.ThrowIfNull(value);
            if (key == "__metadata__") throw new ArgumentException("Metadata cannot name an adapter tensor.", nameof(borrowed));
            if (value.is_sparse || value.dim() > 16) throw new NotSupportedException("Adapter snapshots require dense tensors of rank at most 16.");
            var (dtype, width) = SafeTensorWriter.Describe(value.dtype);
            long end = checked(offset + checked(value.numel() * width));
            if (end > maxSnapshotBytes || end - offset > int.MaxValue) throw new NotSupportedException("Adapter snapshot allowance exceeded.");
            infos.Add(key, new(dtype, Array.AsReadOnly(value.shape), offset, end)); offset = end;
        }
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var noGrad = no_grad();
        values = new(StringComparer.Ordinal);
        foreach (var (key, value) in borrowed)
        {
            cancellationToken.ThrowIfCancellationRequested(); using var iteration = NewDisposeScope();
            values.Add(key, value.detach().to(CPU, copy: true).contiguous().MoveToOuterDisposeScope());
        }
        cancellationToken.ThrowIfCancellationRequested();
        metadata = new ReadOnlyDictionary<string, SafeTensorInfo>(infos);
        foreach (var value in values.Values) value.DetachFromDisposeScope();
    }

    public Tensor ReadTensor(string name, CancellationToken cancellationToken = default)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this); cancellationToken.ThrowIfCancellationRequested();
            using var scope = NewDisposeScope();
            var result = values[name].clone();
            cancellationToken.ThrowIfCancellationRequested(); return result.MoveToOuterDisposeScope();
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return; disposed = true;
            foreach (var value in values.Values) value.Dispose();
        }
    }
}
