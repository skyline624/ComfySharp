using System.Buffers.Binary;
using System.Text.Json;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Exports trained LoRA/LoHa factors and additive weight/bias differences to a new safetensors file.
/// Keys are explicit loader prefixes (without suffixes), not filesystem paths.
/// The caller serializes parameter updates against export; factor snapshots own their CPU storage.</summary>
public static class LoraTrainingFile
{
    public static void SaveNew(string path, IReadOnlyDictionary<string, TrainableLoraPatch> aliases,
        long maxFactorBytes = 512L * 1024 * 1024, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(aliases);
        SaveEntries(path,aliases.Select(p=>new Entry(p.Key,p.Value,false)).ToArray(),maxFactorBytes,cancellationToken);
    }

    /// <summary>Exports canonical weight/bias targets using a source component prefix (for example diffusion_model.).
    /// Bias targets become module.diff_b and one-dimensional weights become module.diff.</summary>
    public static void SaveTargetsNew<T>(string path,IReadOnlyDictionary<string,T> targets,string componentPrefix="diffusion_model.",
        long maxFactorBytes=512L*1024*1024,CancellationToken cancellationToken=default) where T:TrainableWeightPatch
    {
        ArgumentNullException.ThrowIfNull(targets);ArgumentNullException.ThrowIfNull(componentPrefix);
        var entries=new List<Entry>();
        foreach(var (name,patch) in targets)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(name);ArgumentNullException.ThrowIfNull(patch);
            bool bias=name.EndsWith(".bias",StringComparison.Ordinal);
            if(!bias&&!name.EndsWith(".weight",StringComparison.Ordinal))throw new ArgumentException("Export targets must end in .weight or .bias.",nameof(targets));
            if(bias&&patch is not TrainableDifferencePatch)throw new ArgumentException("A bias target requires an additive adapter.",nameof(targets));
            entries.Add(new(componentPrefix+name[..^(bias?5:7)],patch,bias));
        }
        SaveEntries(path,entries,maxFactorBytes,cancellationToken);
    }
    private sealed record Entry(string Prefix,TrainableWeightPatch Patch,bool Bias);
    private static void SaveEntries(string path,IReadOnlyList<Entry> entries,long maxFactorBytes,CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Safetensors export currently requires a little-endian platform.");
        if (maxFactorBytes < 0) throw new ArgumentOutOfRangeException(nameof(maxFactorBytes));
        if (entries.Count is < 1 or > 10000) throw new ArgumentException("Export requires 1-10000 adapter targets.", nameof(entries));
        string destination = Path.GetFullPath(path);
        if (File.Exists(destination) || Directory.Exists(destination)) throw new IOException("Adapter destination already exists.");
        string directory = Path.GetDirectoryName(destination)!;
        if (!Directory.Exists(directory)) throw new DirectoryNotFoundException("Adapter destination directory does not exist.");
        var owners = new List<Entry>();
        try
        {
            long bytes = 0;
            foreach (var (prefix, patch,bias) in entries)
            {
                cancellationToken.ThrowIfCancellationRequested(); ArgumentException.ThrowIfNullOrWhiteSpace(prefix); ArgumentNullException.ThrowIfNull(patch);
                bytes = checked(bytes + (patch switch
                {
                    TrainableLoraPatch lora=>checked((lora.Up.numel()+lora.Down.numel())*4+(lora.AlphaParameter is null?8:4)),
                    TrainableDifferencePatch diff=>checked(diff.Difference.numel()*4),
                    TrainableLohaPatch loha=>checked(loha.Parameters.Sum(p=>p.numel())*4),
                    TrainableLokrPatch lokr=>checked(lokr.Parameters.Sum(p=>p.numel())*4),
                    TrainableOftPatch oft=>checked(oft.Parameters.Sum(p=>p.numel())*4),
                    _=>throw new NotSupportedException("Unknown trainable adapter kind.")
                }));
                if (bytes > maxFactorBytes) throw new NotSupportedException("LoRA export snapshots exceed the configured factor allowance.");
                owners.Add(new(prefix,patch.Retain(),bias));
            }
            NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var noGrad = no_grad();
            var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            foreach (var (prefix, patch,bias) in owners)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if(patch is TrainableDifferencePatch diff)
                    tensors.Add(prefix+(bias?".diff_b":".diff"),diff.Difference.detach().to(CPU,copy:true).contiguous());
                else if(patch is TrainableLoraPatch lora)
                {
                    tensors.Add(prefix + ".lora_up.weight", lora.Up.detach().to(CPU, copy: true).contiguous());
                    tensors.Add(prefix + ".lora_down.weight", lora.Down.detach().to(CPU, copy: true).contiguous());
                    // Preserve learned Float32 alpha bits; explicit double alpha retains its existing F64 encoding.
                    tensors.Add(prefix + ".alpha", lora.AlphaParameter is { } alpha
                        ?alpha.detach().to(CPU,copy:true).contiguous():tensor(lora.Alpha,dtype:ScalarType.Float64));
                }
                else if(patch is TrainableLohaPatch loha)
                    foreach(var (key,value) in loha.NamedParameters)
                        tensors.Add(prefix+"."+key,value.detach().to(CPU,copy:true).contiguous());
                else if(patch is TrainableLokrPatch lokr)
                    foreach(var (key,value) in lokr.NamedParameters)
                        tensors.Add(prefix+"."+key,value.detach().to(CPU,copy:true).contiguous());
                else if(patch is TrainableOftPatch oft)
                    foreach(var (key,value) in oft.NamedParameters)
                        tensors.Add(prefix+"."+key,value.detach().to(CPU,copy:true).contiguous());
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
        finally { foreach (var owner in owners) owner.Patch.Dispose(); }
    }
}
