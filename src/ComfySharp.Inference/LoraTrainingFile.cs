using System.Buffers.Binary;
using System.Text.Json;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Exports ordinary trained LoRA factors to a new safetensors file.
/// Keys are explicit loader prefixes (without suffixes), not filesystem paths.
/// The caller serializes parameter updates against export; factor snapshots own their CPU storage.</summary>
public static class LoraTrainingFile
{
    public static void SaveNew(string path, IReadOnlyDictionary<string, TrainableLoraPatch> aliases,
        long maxFactorBytes = 512L * 1024 * 1024, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(aliases);
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Safetensors export currently requires a little-endian platform.");
        if (maxFactorBytes < 0) throw new ArgumentOutOfRangeException(nameof(maxFactorBytes));
        if (aliases.Count is < 1 or > 10000) throw new ArgumentException("Export requires 1-10000 LoRA prefixes.", nameof(aliases));
        string destination = Path.GetFullPath(path);
        if (File.Exists(destination) || Directory.Exists(destination)) throw new IOException("Adapter destination already exists.");
        string directory = Path.GetDirectoryName(destination)!;
        if (!Directory.Exists(directory)) throw new DirectoryNotFoundException("Adapter destination directory does not exist.");
        var owners = new Dictionary<string, TrainableLoraPatch>(StringComparer.Ordinal);
        try
        {
            long bytes = 0;
            foreach (var (prefix, patch) in aliases)
            {
                cancellationToken.ThrowIfCancellationRequested(); ArgumentException.ThrowIfNullOrWhiteSpace(prefix); ArgumentNullException.ThrowIfNull(patch);
                bytes = checked(bytes + (patch.Up.numel() + patch.Down.numel()) * 4 + 8);
                if (bytes > maxFactorBytes) throw new NotSupportedException("LoRA export snapshots exceed the configured factor allowance.");
                owners.Add(prefix, patch.Retain());
            }
            NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var noGrad = no_grad();
            var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            foreach (var (prefix, patch) in owners)
            {
                cancellationToken.ThrowIfCancellationRequested();
                tensors.Add(prefix + ".lora_up.weight", patch.Up.detach().to(CPU, copy: true).contiguous());
                tensors.Add(prefix + ".lora_down.weight", patch.Down.detach().to(CPU, copy: true).contiguous());
                // F64 keeps the explicit alpha value rather than silently narrowing it on export.
                tensors.Add(prefix + ".alpha", tensor(patch.Alpha, dtype: ScalarType.Float64));
            }
            var header = new Dictionary<string, object>(StringComparer.Ordinal); long offset = 0;
            foreach (var (name, tensor) in tensors)
            {
                if (!tensor.isfinite().all().item<bool>()) throw new ArithmeticException("Cannot save nonfinite LoRA parameters.");
                long end = checked(offset + tensor.numel() * (tensor.dtype == ScalarType.Float64 ? 8 : 4));
                header.Add(name, new { dtype = tensor.dtype == ScalarType.Float64 ? "F64" : "F32", shape = tensor.shape, data_offsets = new[] { offset, end } }); offset = end;
            }
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
            int headerLength = checked((json.Length + 7) / 8 * 8);
            if (headerLength > 16 * 1024 * 1024) throw new NotSupportedException("LoRA header exceeds the safe reader's default allowance.");
            byte[] padded = new byte[headerLength]; padded.AsSpan().Fill(32); json.CopyTo(padded, 0);
            string temporary = Path.Combine(directory, ".comfysharp-lora-" + Guid.NewGuid().ToString("N") + ".tmp");
            bool created = false;
            try
            {
                using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                {
                    created = true;
                    Span<byte> size = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(size, (ulong)headerLength);
                    stream.Write(size); stream.Write(padded);
                    foreach (var tensor in tensors.Values)
                        for (int start = 0; start < tensor.bytes.Length; start += 1024 * 1024)
                        {
                            cancellationToken.ThrowIfCancellationRequested();
                            stream.Write(tensor.bytes.Slice(start, Math.Min(1024 * 1024, tensor.bytes.Length - start)));
                        }
                    stream.Flush(flushToDisk: true);
                }
                cancellationToken.ThrowIfCancellationRequested();
                File.Move(temporary, destination, overwrite: false);
                created = false;
            }
            finally { if (created) File.Delete(temporary); }
        }
        finally { foreach (var owner in owners.Values) owner.Dispose(); }
    }
}
