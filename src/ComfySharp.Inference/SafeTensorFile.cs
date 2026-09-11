using System.Buffers.Binary;
using System.Collections.ObjectModel;
using System.Text.Json;
using TorchSharp;

namespace ComfySharp.Inference;

public sealed record SafeTensorLimits(int MaxHeaderBytes = 16 * 1024 * 1024, int MaxTensors = 100_000,
    int MaxRank = 16, long MaxTensorBytes = 512 * 1024 * 1024);

public sealed record SafeTensorInfo(string DType, IReadOnlyList<long> Shape, long Start, long End);

/// <summary>Strict contiguous safetensors reader. Owns its file; returned tensors own copied storage.</summary>
public sealed class SafeTensorFile : IDisposable
{
    private readonly FileStream stream;
    private readonly long dataStart;
    private readonly object gate = new();
    private bool disposed;
    public IReadOnlyDictionary<string, SafeTensorInfo> Tensors { get; }
    public IReadOnlyDictionary<string, string> Metadata { get; }

    public SafeTensorFile(string path, SafeTensorLimits? limits = null)
    {
        limits ??= new();
        if (limits.MaxHeaderBytes < 2 || limits.MaxTensors < 1 || limits.MaxRank < 0 || limits.MaxTensorBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(limits));
        stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        try
        {
            Span<byte> prefix = stackalloc byte[8];
            stream.ReadExactly(prefix);
            var headerLength = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
            if (headerLength > (ulong)limits.MaxHeaderBytes || headerLength > (ulong)Math.Max(0, stream.Length - 8))
                throw new InvalidDataException("Header exceeds file bounds or configured limit.");
            byte[] header = new byte[(int)headerLength];
            stream.ReadExactly(header);
            if (header.Length == 0 || header[0] != '{') throw new InvalidDataException("Header must start with an object.");
            using var json = JsonDocument.Parse(header, new() { MaxDepth = 32 });
            if (json.RootElement.ValueKind != JsonValueKind.Object) throw new InvalidDataException("Header must be an object.");
            dataStart = 8 + (long)headerLength;
            var tensors = new Dictionary<string, SafeTensorInfo>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var entry in json.RootElement.EnumerateObject())
            {
                if (!names.Add(entry.Name)) throw new InvalidDataException("Duplicate header key.");
                if (entry.Name == "__metadata__")
                {
                    foreach (var pair in entry.Value.EnumerateObject())
                        if (pair.Value.ValueKind != JsonValueKind.String || !metadata.TryAdd(pair.Name, pair.Value.GetString()!))
                            throw new InvalidDataException("Metadata must contain unique string values.");
                    continue;
                }
                if (tensors.Count >= limits.MaxTensors) throw new InvalidDataException("Too many tensors.");
                var fields = new HashSet<string>(StringComparer.Ordinal);
                foreach (var field in entry.Value.EnumerateObject())
                    if (!fields.Add(field.Name) || field.Name is not ("dtype" or "shape" or "data_offsets"))
                        throw new InvalidDataException("Duplicate or unknown tensor field.");
                var dtype = entry.Value.GetProperty("dtype").GetString()!;
                int width = dtype switch { "BOOL" or "U8" or "I8" => 1, "I16" or "U16" or "F16" or "BF16" => 2,
                    "I32" or "U32" or "F32" => 4, "I64" or "U64" or "F64" => 8, _ => throw new InvalidDataException("Unsupported dtype.") };
                var shape = entry.Value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray();
                if (shape.Length > limits.MaxRank || shape.Any(d => d < 0)) throw new InvalidDataException("Invalid shape.");
                long elements = 1;
                foreach (long dimension in shape) elements = checked(elements * dimension);
                long bytes = checked(elements * width);
                var offsets = entry.Value.GetProperty("data_offsets").EnumerateArray().Select(v => v.GetInt64()).ToArray();
                if (offsets.Length != 2 || offsets[0] < 0 || offsets[1] < offsets[0] || offsets[1] > stream.Length - dataStart ||
                    offsets[1] - offsets[0] != bytes || bytes > limits.MaxTensorBytes)
                    throw new InvalidDataException("Invalid tensor byte range or configured size limit exceeded.");
                tensors.Add(entry.Name, new(dtype, Array.AsReadOnly(shape), offsets[0], offsets[1]));
            }
            long cursor = 0;
            foreach (var tensor in tensors.Values.OrderBy(t => t.Start).ThenBy(t => t.End))
            {
                if (tensor.Start != cursor) throw new InvalidDataException("Overlapping tensors or unindexed data.");
                cursor = tensor.End;
            }
            if (cursor != stream.Length - dataStart) throw new InvalidDataException("Trailing unindexed data.");
            Tensors = new ReadOnlyDictionary<string, SafeTensorInfo>(tensors);
            Metadata = new ReadOnlyDictionary<string, string>(metadata);
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or KeyNotFoundException or FormatException or OverflowException or EndOfStreamException)
        {
            stream.Dispose();
            throw new InvalidDataException("Malformed safetensors file.", e);
        }
        catch { stream.Dispose(); throw; }
    }

    /// <summary>Reads only F32 into independent CPU storage. Other validated dtypes are metadata-only.</summary>
    public torch.Tensor ReadFloat32(string name, CancellationToken cancellationToken = default)
        => ReadFloat32(name, cancellationToken, static (values, shape) => torch.tensor(values).reshape(shape));

    // Injectable materialization boundary enables deterministic cancellation/lifetime verification.
    internal torch.Tensor ReadFloat32(string name, CancellationToken cancellationToken,
        Func<float[], long[], torch.Tensor> materialize)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            cancellationToken.ThrowIfCancellationRequested();
            var info = Tensors[name];
            if (info.DType != "F32") throw new NotSupportedException("Tensor materialization currently supports F32 only.");
            var bytes = new byte[checked((int)(info.End - info.Start))];
            stream.Position = dataStart + info.Start;
            for (int offset = 0; offset < bytes.Length;)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int length = Math.Min(1024 * 1024, bytes.Length - offset);
                stream.ReadExactly(bytes.AsSpan(offset, length));
                offset += length;
            }
            var values = new float[bytes.Length / 4];
            for (int i = 0; i < values.Length; i++) values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
            cancellationToken.ThrowIfCancellationRequested();
            using var scope = torch.NewDisposeScope();
            var result = materialize(values, info.Shape.ToArray());
            cancellationToken.ThrowIfCancellationRequested();
            return result.MoveToOuterDisposeScope();
        }
    }

    public void Dispose() { lock (gate) { if (!disposed) { disposed = true; stream.Dispose(); } } }
}
