using System.Buffers.Binary;
using System.Collections.ObjectModel;
using System.Security.Cryptography;
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
    public long FileSizeBytes { get; }

    public SafeTensorFile(string path, SafeTensorLimits? limits = null)
    {
        limits ??= new();
        if (limits.MaxHeaderBytes < 2 || limits.MaxTensors < 1 || limits.MaxRank < 0 || limits.MaxTensorBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(limits));
        stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        try
        {
            FileSizeBytes = stream.Length;
            Span<byte> prefix = stackalloc byte[8];
            stream.ReadExactly(prefix);
            var headerLength = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
            if (headerLength > (ulong)limits.MaxHeaderBytes || headerLength > (ulong)Math.Max(0, FileSizeBytes - 8))
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
                int width = DTypeWidth(dtype);
                var shape = entry.Value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray();
                if (shape.Length > limits.MaxRank || shape.Any(d => d < 0)) throw new InvalidDataException("Invalid shape.");
                long elements = 1;
                foreach (long dimension in shape) elements = checked(elements * dimension);
                long bytes = checked(elements * width);
                var offsets = entry.Value.GetProperty("data_offsets").EnumerateArray().Select(v => v.GetInt64()).ToArray();
                if (offsets.Length != 2 || offsets[0] < 0 || offsets[1] < offsets[0] || offsets[1] > FileSizeBytes - dataStart ||
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
            if (cursor != FileSizeBytes - dataStart) throw new InvalidDataException("Trailing unindexed data.");
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

    /// <summary>Explicitly hashes the complete file using the same open handle whose header was parsed.
    /// Reads payload bytes but does not materialize tensors or initialize native Torch code.</summary>
    public string ComputeSha256(CancellationToken cancellationToken = default)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            cancellationToken.ThrowIfCancellationRequested();
            if (stream.Length != FileSizeBytes) throw new InvalidDataException("File length changed after header validation.");
            long position = stream.Position;
            try
            {
                stream.Position = 0;
                using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                byte[] buffer = new byte[1024 * 1024];
                long remaining = FileSizeBytes;
                while (remaining > 0)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int count = stream.Read(buffer, 0, (int)Math.Min(buffer.Length, remaining));
                    if (count == 0) throw new InvalidDataException("File was truncated during hashing.");
                    hash.AppendData(buffer, 0, count);
                    remaining -= count;
                }
                cancellationToken.ThrowIfCancellationRequested();
                if (stream.Length != FileSizeBytes) throw new InvalidDataException("File length changed during hashing.");
                return Convert.ToHexStringLower(hash.GetHashAndReset());
            }
            finally { stream.Position = position; }
        }
    }

    /// <summary>Whether this dtype can be copied into a CPU tensor by this reader. Does not initialize native code
    /// and does not assess model architecture, operators, or weight compatibility.</summary>
    public static bool SupportsTensorDType(string dtype) => dtype is
        "BOOL" or "U8" or "I8" or "I16" or "I32" or "I64" or "F16" or "BF16" or "F32" or "F64";

    private static int DTypeWidth(string dtype) => dtype switch
    {
        "BOOL" or "U8" or "I8" => 1,
        "I16" or "U16" or "F16" or "BF16" => 2,
        "I32" or "U32" or "F32" => 4,
        "I64" or "U64" or "F64" => 8,
        _ => throw new InvalidDataException($"Unsupported safetensors dtype '{dtype}'.")
    };

    private static torch.ScalarType TensorDType(string dtype) => dtype switch
    {
        "BOOL" => torch.ScalarType.Bool,
        "U8" => torch.ScalarType.Byte,
        "I8" => torch.ScalarType.Int8,
        "I16" => torch.ScalarType.Int16,
        "I32" => torch.ScalarType.Int32,
        "I64" => torch.ScalarType.Int64,
        "F16" => torch.ScalarType.Float16,
        "BF16" => torch.ScalarType.BFloat16,
        "F32" => torch.ScalarType.Float32,
        "F64" => torch.ScalarType.Float64,
        _ => throw new NotSupportedException($"Dtype '{dtype}' is valid metadata but cannot be materialized: " +
            "the pinned TorchSharp API has no matching scalar type.")
    };

    /// <summary>Copies a validated tensor into independent CPU storage without floating-point conversion.
    /// Supports BOOL, U8, I8/I16/I32/I64, F16/BF16/F32/F64. U16/U32/U64 remain metadata-only.
    /// The caller owns the result, including after this reader is disposed. Reads are bounded to Int32.MaxValue bytes.</summary>
    public torch.Tensor ReadTensor(string name, CancellationToken cancellationToken = default)
        => ReadTensor(name, cancellationToken, static (shape, dtype) => torch.empty(shape, dtype: dtype, device: torch.CPU));

    // An allocation boundary permits deterministic verification of cancellation after native allocation.
    internal torch.Tensor ReadTensor(string name, CancellationToken cancellationToken,
        Func<long[], torch.ScalarType, torch.Tensor> allocate, Action? chunkRead = null)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            cancellationToken.ThrowIfCancellationRequested();
            var info = Tensors[name];
            var dtype = TensorDType(info.DType);
            long length = checked(info.End - info.Start);
            if (length > int.MaxValue)
                throw new NotSupportedException("CPU tensor reads are limited to Int32.MaxValue bytes per tensor.");
            if (checked(dataStart + info.End) > stream.Length)
                throw new InvalidDataException("Tensor data was truncated after its header was validated.");
            cancellationToken.ThrowIfCancellationRequested();
            NativeRuntimeBootstrap.Initialize();
            using var scope = torch.NewDisposeScope();
            var result = allocate(info.Shape.ToArray(), dtype);
            cancellationToken.ThrowIfCancellationRequested();
            stream.Position = checked(dataStart + info.Start);
            // torch.empty owns CPU storage. Read directly into it; never share file-backed or managed buffers.
            // Skip bytes access for empty tensors because their native data pointer may be null.
            for (int offset = 0; offset < (int)length;)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int chunkLength = Math.Min(1024 * 1024, (int)length - offset);
                var chunk = result.bytes.Slice(offset, chunkLength);
                stream.ReadExactly(chunk);
                NormalizeByteOrder(chunk, DTypeWidth(info.DType), BitConverter.IsLittleEndian);
                offset += chunkLength;
                chunkRead?.Invoke();
            }
            cancellationToken.ThrowIfCancellationRequested();
            return result.MoveToOuterDisposeScope();
        }
    }

    // Safetensors payloads are always little endian; Torch tensor storage uses native byte order.
    internal static void NormalizeByteOrder(Span<byte> bytes, int elementWidth, bool isLittleEndian)
    {
        if (elementWidth is not (1 or 2 or 4 or 8) || bytes.Length % elementWidth != 0)
            throw new ArgumentException("Invalid element width or byte count.");
        if (!isLittleEndian && elementWidth != 1)
            for (int offset = 0; offset < bytes.Length; offset += elementWidth)
                bytes.Slice(offset, elementWidth).Reverse();
    }

    /// <summary>Compatibility API that reads only F32 into independent CPU storage. Use ReadTensor for other supported dtypes.</summary>
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
            NativeRuntimeBootstrap.Initialize();
            using var scope = torch.NewDisposeScope();
            var result = materialize(values, info.Shape.ToArray());
            cancellationToken.ThrowIfCancellationRequested();
            return result.MoveToOuterDisposeScope();
        }
    }

    public void Dispose() { lock (gate) { if (!disposed) { disposed = true; stream.Dispose(); } } }
}
