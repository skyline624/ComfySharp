using System.Buffers.Binary;
using System.Text.Json;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Serializes an independent CPU snapshot of borrowed tensors without changing their dtype.
/// The destination is borrowed; use an atomic file store for publication. Parameter updates must
/// be serialized against snapshot capture by the caller.</summary>
public static class SafeTensorWriter
{
    internal static (string Name, int Width) Describe(ScalarType dtype) => dtype switch
    {
        ScalarType.Bool => ("BOOL", 1), ScalarType.Byte => ("U8", 1), ScalarType.Int8 => ("I8", 1),
        ScalarType.Int16 => ("I16", 2), ScalarType.Int32 => ("I32", 4), ScalarType.Int64 => ("I64", 8),
        ScalarType.Float16 => ("F16", 2), ScalarType.BFloat16 => ("BF16", 2),
        ScalarType.Float32 => ("F32", 4), ScalarType.Float64 => ("F64", 8),
        _ => throw new NotSupportedException($"Safetensors export does not support {dtype}.")
    };

    public static void Write(Stream destination, IReadOnlyDictionary<string, Tensor> tensors,
        long maxSnapshotBytes = 512L * 1024 * 1024, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(destination); ArgumentNullException.ThrowIfNull(tensors);
        cancellationToken.ThrowIfCancellationRequested();
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Safetensors writing requires a little-endian platform.");
        if (!destination.CanWrite) throw new ArgumentException("The destination must be writable.", nameof(destination));
        if (maxSnapshotBytes < 0) throw new ArgumentOutOfRangeException(nameof(maxSnapshotBytes));
        if (tensors.Count > 100_000) throw new NotSupportedException("Too many tensors for the safetensors writer.");
        var entries = new List<(string Key, Tensor Value, string DType, int Width, long Bytes)>();
        long total = 0;
        foreach (var (key, value) in tensors)
        {
            cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(key); ArgumentNullException.ThrowIfNull(value);
            if (key == "__metadata__") throw new ArgumentException("The reserved metadata key cannot name a tensor.", nameof(tensors));
            if (value.shape.Length > 16) throw new NotSupportedException("Tensor rank exceeds the safe reader's limit.");
            if (!value.is_contiguous()) throw new ArgumentException($"Tensor '{key}' must be contiguous before saving.", nameof(tensors));
            var (dtype, width) = Describe(value.dtype);
            long bytes = checked(value.numel() * width); total = checked(total + bytes);
            if (bytes > int.MaxValue || total > maxSnapshotBytes) throw new NotSupportedException("Safetensors snapshots exceed the configured byte allowance.");
            entries.Add((key, value, dtype, width, bytes));
        }
        // Larger element widths first keep all payload offsets naturally aligned.
        var ordered = entries.OrderByDescending(e => e.Width).ThenBy(e => e.Key, StringComparer.Ordinal).ToArray();
        var header = new Dictionary<string, object>(StringComparer.Ordinal); long offset = 0;
        foreach (var e in ordered)
        {
            header.Add(e.Key, new { dtype = e.DType, shape = e.Value.shape, data_offsets = new[] { offset, offset + e.Bytes } });
            offset += e.Bytes;
        }
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
        int length = checked((json.Length + 7) / 8 * 8);
        if (length > 16 * 1024 * 1024) throw new NotSupportedException("Safetensors header exceeds the safe reader's limit.");
        byte[] padded = new byte[length]; padded.AsSpan().Fill(32); json.CopyTo(padded, 0);
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        var snapshots = new List<(Tensor Tensor, long Bytes)>();
        foreach (var e in ordered)
        {
            cancellationToken.ThrowIfCancellationRequested();
            snapshots.Add((e.Value.detach().to(CPU, copy: true), e.Bytes));
        }
        cancellationToken.ThrowIfCancellationRequested();
        Span<byte> prefix = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)length);
        destination.Write(prefix); destination.Write(padded);
        foreach (var (snapshot, bytes) in snapshots)
            for (int start = 0; start < bytes;)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int count = (int)Math.Min(1024 * 1024, bytes - start);
                destination.Write(snapshot.bytes.Slice(start, count)); start += count;
            }
        cancellationToken.ThrowIfCancellationRequested();
    }
}
