using System.Runtime.InteropServices;
using System.Security.Cryptography;
using ComfySharp.Inference;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

internal sealed record SyntheticTensorPlan(string Name, long[] Shape, long Bytes, SdSyntheticRecipe.Descriptor Recipe);
internal sealed record SyntheticTensorRecord(string Name, long[] Shape, long Bytes, string Sha256);
internal sealed record SyntheticWeightPlan(IReadOnlyList<SyntheticTensorPlan> Tensors, long ResidentBytes, long LargestTensorBytes)
{
    public int TensorCount => Tensors.Count;
}
internal sealed record SyntheticBuildOptions(long MaximumWeightBytes, int ChunkElements = 262_144)
{
    internal void Validate()
    {
        if (MaximumWeightBytes <= 0) throw new ArgumentOutOfRangeException(nameof(MaximumWeightBytes));
        if (ChunkElements is < 1 or > 262_144) throw new ArgumentOutOfRangeException(nameof(ChunkElements));
    }
}
internal sealed class SyntheticBank<T>(T weights, IReadOnlyList<SyntheticTensorRecord> parameters) : IDisposable where T : IDisposable
{
    internal T Weights { get; } = weights;
    internal IReadOnlyList<SyntheticTensorRecord> Parameters { get; } = parameters;
    public void Dispose() => Weights.Dispose();
}

/// <summary>One native destination per parameter. No whole managed tensor or second bank.</summary>
internal static class SdSyntheticWeightBuilder
{
    internal static SyntheticWeightPlan DescribeUnet(SdUnetConfig config) => Describe(UnetWeightSchema.Describe(config));

    internal static SyntheticWeightPlan Describe(IReadOnlyDictionary<string, IReadOnlyList<long>> schema)
    {
        ArgumentNullException.ThrowIfNull(schema);
        if (schema.Count == 0) throw new ArgumentException("A synthetic schema cannot be empty.", nameof(schema));
        var entries = new List<SyntheticTensorPlan>(schema.Count);
        long total = 0, largest = 0;
        foreach (var (name, shape) in schema.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            var entry = TensorPlan(name, shape, parameter: true);
            total = checked(total + entry.Bytes);
            largest = Math.Max(largest, entry.Bytes);
            entries.Add(entry);
        }
        return new(entries.AsReadOnly(), total, largest);
    }

    private static SyntheticTensorPlan TensorPlan(string name, IReadOnlyList<long> shape, bool parameter)
    {
        var recipe = SdSyntheticRecipe.Describe(name, shape, parameter);
        long bytes = checked(recipe.Elements * sizeof(float));
        if (bytes > int.MaxValue) throw new NotSupportedException("A synthetic tensor exceeds the CPU span byte limit.");
        return new(name, shape.ToArray(), bytes, recipe);
    }

    internal static SyntheticBank<UnetWeightSet> CreateUnet(SdUnetConfig config, SyntheticBuildOptions options,
        CancellationToken cancellationToken = default)
        => Create(DescribeUnet(config), options, values => UnetWeightSet.FromOwnedTensors(config, values), cancellationToken);

    // Allocation and chunk notifications are test seams only. The normal tool uses the
    // native CPU allocator and never supplies callbacks carrying a borrowed Tensor.
    internal static SyntheticBank<T> Create<T>(SyntheticWeightPlan plan, SyntheticBuildOptions options,
        Func<IReadOnlyDictionary<string, Tensor>, T> transfer, CancellationToken cancellationToken = default,
        Func<long[], Tensor>? allocate = null, Action<string, long>? chunkWritten = null) where T : class, IDisposable
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(transfer);
        options.Validate();
        if (plan.ResidentBytes > options.MaximumWeightBytes) throw new InvalidOperationException("Synthetic weight budget exceeded.");
        cancellationToken.ThrowIfCancellationRequested();
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Synthetic evidence requires a little-endian host.");
        NativeRuntimeBootstrap.Initialize();
        var owned = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        T? committed = null;
        try
        {
            using var noGrad = no_grad();
            var records = new List<SyntheticTensorRecord>(plan.TensorCount);
            foreach (var entry in plan.Tensors)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var scope = NewDisposeScope();
                var tensor = Materialize(entry, options.ChunkElements, cancellationToken, allocate, chunkWritten, out var record);
                records.Add(record);
                owned.Add(entry.Name, tensor);
                tensor.DetachFromDisposeScope();
            }
            cancellationToken.ThrowIfCancellationRequested();
            committed = transfer(owned) ?? throw new InvalidDataException("The weight transfer returned no owner.");
            owned.Clear();
            cancellationToken.ThrowIfCancellationRequested();
            var result = new SyntheticBank<T>(committed, records.AsReadOnly());
            committed = null;
            return result;
        }
        finally
        {
            committed?.Dispose();
            foreach (var tensor in owned.Values) tensor.Dispose();
        }
    }

    internal static Tensor CreateInput(string name, IReadOnlyList<long> shape, int chunkElements,
        CancellationToken cancellationToken, out SyntheticTensorRecord record)
    {
        var plan = TensorPlan(name, shape, parameter: false);
        new SyntheticBuildOptions(plan.Bytes, chunkElements).Validate();
        cancellationToken.ThrowIfCancellationRequested();
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Synthetic evidence requires a little-endian host.");
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        return Materialize(plan, chunkElements, cancellationToken, null, null, out record).DetachFromDisposeScope();
    }

    private static Tensor Materialize(SyntheticTensorPlan plan, int chunkElements, CancellationToken token,
        Func<long[], Tensor>? allocate, Action<string, long>? chunkWritten, out SyntheticTensorRecord record)
    {
        token.ThrowIfCancellationRequested();
        var tensor = allocate is null ? empty(plan.Shape, dtype: ScalarType.Float32, device: CPU) : allocate(plan.Shape);
        try
        {
        token.ThrowIfCancellationRequested();
        if (tensor is null || tensor.IsInvalid || tensor.dtype != ScalarType.Float32 || tensor.device_type != DeviceType.CPU
            || tensor.is_sparse || !tensor.is_contiguous() || tensor.requires_grad || !tensor.shape.SequenceEqual(plan.Shape))
            throw new InvalidDataException("The synthetic allocator returned an incompatible tensor.");
        if (!IsAligned(tensor)) throw new NotSupportedException("The synthetic CPU allocation must be aligned to 64 bytes.");
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        for (long start = 0; start < plan.Recipe.Elements;)
        {
            token.ThrowIfCancellationRequested();
            int count = checked((int)Math.Min(chunkElements, plan.Recipe.Elements - start));
            var bytes = tensor.bytes.Slice(checked((int)(start * 4)), checked(count * 4));
            SdSyntheticRecipe.Fill(MemoryMarshal.Cast<byte, float>(bytes), start, plan.Recipe);
            hash.AppendData(bytes);
            start += count;
            chunkWritten?.Invoke(plan.Name, start);
        }
        token.ThrowIfCancellationRequested();
        record = new(plan.Name, plan.Shape.ToArray(), plan.Bytes, Convert.ToHexStringLower(hash.GetHashAndReset()));
        return tensor;
        }
        catch
        {
            // Allocator ownership transfers on return, even when a test allocator has
            // detached its wrapper. DisposeScope alone would not cover that case.
            tensor?.Dispose();
            throw;
        }
    }

    internal static unsafe bool IsAligned(Tensor tensor)
    {
        // Same supported contiguous CPU-span check as the inference weight bank. No
        // pointer escapes, and the caller retains tensor ownership throughout the check.
        fixed (byte* address = tensor.bytes) return ((nuint)address & 63) == 0;
    }
}
