using TorchSharp;

namespace ComfySharp.Inference;

public enum SdModelWeightTransform { Identity, Reshape }

public sealed record SdModelWeightMapping(string SourceName, string CanonicalName, SdModelWeightTransform Transform,
    IReadOnlyList<long> SourceShape, IReadOnlyList<long> CanonicalShape, string SourceDType);

internal sealed class SdModelWeightPlan
{
    internal SafeTensorFile Source { get; }
    internal IReadOnlyList<SdModelWeightMapping> Mappings { get; }
    internal long SourceBytes { get; }
    internal long ResidentBytes { get; }
    internal long TemporaryBytes { get; }
    internal SdModelWeightPlan(SafeTensorFile source, List<SdModelWeightMapping> mappings)
    {
        Source = source; Mappings = mappings.AsReadOnly();
        foreach (var mapping in mappings)
        {
            long elements = mapping.CanonicalShape.Aggregate(1L, (n, d) => checked(n * d));
            long bytes = checked(elements * (mapping.SourceDType == "F32" ? 4 : 2));
            SourceBytes = checked(SourceBytes + bytes);
            ResidentBytes = checked(ResidentBytes + elements * 4);
            // Conservative read + conversion allowance; pure reshape adds no payload copy.
            // Each native allocation is checked below, so bank transfer cannot clone the bank.
            TemporaryBytes = Math.Max(TemporaryBytes, checked(bytes + elements * 4));
        }
    }
}

internal static class SdModelWeightLoading
{
    internal static SdModelWeightPlan Inspect(SafeTensorFile file, ModelWeightSchemaBuilder definition,
        string prefix, bool diffusers, bool nestedQuantAliases)
    {
        ArgumentNullException.ThrowIfNull(file);
        var mappings = new List<SdModelWeightMapping>();
        var claimed = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (source, info) in file.Tensors)
        {
            if (!source.StartsWith(prefix, StringComparison.Ordinal)) continue;
            string local = source[prefix.Length..];
            if (nestedQuantAliases)
            {
                if (local.StartsWith("encoder.quant_conv.", StringComparison.Ordinal)) local = local["encoder.".Length..];
                if (local.StartsWith("decoder.post_quant_conv.", StringComparison.Ordinal)) local = local["decoder.".Length..];
            }
            string canonical = local;
            bool reshape = false;
            if (diffusers)
            {
                if (!definition.Diffusers.TryGetValue(local, out var mapped))
                    throw new InvalidDataException($"Unknown weight '{source}' in the explicitly selected Diffusers layout.");
                (canonical, reshape) = mapped;
            }
            if (!definition.Shapes.TryGetValue(canonical, out var expected))
                throw new InvalidDataException($"Unknown weight '{source}' in the selected component namespace.");
            if (!claimed.TryAdd(canonical, source))
                throw new InvalidDataException($"Weight alias collision: '{claimed[canonical]}' and '{source}' both map to '{canonical}'.");
            if (info.DType is not ("F32" or "F16" or "BF16"))
                throw new InvalidDataException($"Weight '{source}' uses unsupported storage dtype '{info.DType}'; expected F32/F16/BF16.");
            if (info.End - info.Start > int.MaxValue)
                throw new NotSupportedException($"Weight '{source}' exceeds the native reader's per-tensor byte limit.");
            IReadOnlyList<long> sourceShape = reshape ? new long[] { expected[0], expected[1] } : expected;
            ModelWeightSchemaBuilder.CheckShape(source, info.Shape, sourceShape);
            mappings.Add(new(source, canonical, reshape ? SdModelWeightTransform.Reshape : SdModelWeightTransform.Identity,
                Array.AsReadOnly(info.Shape.ToArray()), Array.AsReadOnly(expected.ToArray()), info.DType));
        }
        foreach (string name in definition.Shapes.Keys)
            if (!claimed.ContainsKey(name)) throw new InvalidDataException($"Missing required weight '{name}' for the selected component/layout.");
        return new(file, mappings);
    }

    internal static T Load<T>(SafeTensorFile file, SdModelWeightPlan plan, CancellationToken cancellationToken,
        Func<IReadOnlyDictionary<string, torch.Tensor>, T> transfer,
        Action<IReadOnlyDictionary<string, torch.Tensor>>? materialized = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        if (!ReferenceEquals(file, plan.Source))
            throw new ArgumentException("A weight plan must be loaded from the same open safetensors reader that was inspected.", nameof(file));
        cancellationToken.ThrowIfCancellationRequested();
        NativeRuntimeBootstrap.Initialize();
        var owned = new Dictionary<string, torch.Tensor>(StringComparer.Ordinal);
        try
        {
            using var noGrad = torch.no_grad();
            foreach (var mapping in plan.Mappings)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var scope = torch.NewDisposeScope();
                var raw = file.ReadTensor(mapping.SourceName, cancellationToken);
                var converted = raw.dtype == torch.ScalarType.Float32 ? raw : raw.to_type(torch.ScalarType.Float32);
                var canonical = mapping.Transform == SdModelWeightTransform.Reshape ? converted.reshape(mapping.CanonicalShape.ToArray()) : converted;
                if (!CpuModelWeightBank.IsAligned(canonical))
                    throw new NotSupportedException("The CPU allocator did not provide 64-byte aligned weight storage.");
                owned.Add(mapping.CanonicalName, canonical);
                canonical.DetachFromDisposeScope();
                materialized?.Invoke(owned);
            }
            cancellationToken.ThrowIfCancellationRequested();
            T result = transfer(owned);
            owned.Clear();
            return result;
        }
        finally { foreach (var tensor in owned.Values) tensor.Dispose(); }
    }
}
