using System.Text.RegularExpressions;
using TorchSharp;

namespace ComfySharp.Inference;

public enum ClipCheckpointLayout { Canonical, ClipL, ClipG, Sd1, SdxlL, SdxlG, OpenClip }
public enum ClipWeightTransform { Identity, Transpose, QkvSlice }

public sealed record ClipWeightMapping(string SourceName, string CanonicalName, ClipWeightTransform Transform,
    int SliceStart, IReadOnlyList<long> SourceShape, IReadOnlyList<long> CanonicalShape, string SourceDType);

/// <summary>Validated metadata only. Estimates exclude graph activations, allocator overhead and runtime libraries.</summary>
public sealed class ClipWeightPlan
{
    internal SafeTensorFile Source { get; }
    public ClipTextConfig Config { get; }
    public ClipCheckpointLayout Layout { get; }
    public bool RequireProjection { get; }
    public bool HasProjection { get; }
    public IReadOnlyList<ClipWeightMapping> Mappings { get; }
    public IReadOnlyList<string> IgnoredEncoderKeys { get; }
    public long ResidentBytes { get; }
    /// <summary>Conservative additional peak for one source read, its Float32 conversion and transformed copies.</summary>
    public long TemporaryBytes { get; }
    public long EstimatedPeakWeightBytes => checked(ResidentBytes + TemporaryBytes);

    internal ClipWeightPlan(SafeTensorFile source, ClipTextConfig config, ClipCheckpointLayout layout,
        bool requireProjection, List<ClipWeightMapping> mappings, List<string> ignored)
    {
        Source = source; Config = config; Layout = layout; RequireProjection = requireProjection;
        Mappings = mappings.AsReadOnly(); IgnoredEncoderKeys = ignored.AsReadOnly();
        HasProjection = mappings.Any(m => m.CanonicalName == ClipWeightSchema.Projection);
        static long Elements(IReadOnlyList<long> shape) => shape.Aggregate(1L, (n, d) => checked(n * d));
        ResidentBytes = mappings.Sum(m => checked(Elements(m.CanonicalShape) * 4));
        TemporaryBytes = mappings.GroupBy(m => m.SourceName).Max(group =>
        {
            var first = group.First();
            long elements = Elements(first.SourceShape);
            return checked(elements * (first.SourceDType == "F32" ? 4 : 6) +
                group.Where(m => m.Transform != ClipWeightTransform.Identity).Sum(m => Elements(m.CanonicalShape) * 4));
        });
    }
}

/// <summary>Explicit, metadata-validated safetensors adapters for frozen CPU/Float32 CLIP inference.</summary>
public static partial class ClipCheckpointLoader
{
    public static ClipWeightPlan Inspect(SafeTensorFile file, ClipTextConfig config, ClipCheckpointLayout layout,
        bool requireProjection = true)
    {
        ArgumentNullException.ThrowIfNull(file);
        if (!Enum.IsDefined(layout)) throw new ArgumentOutOfRangeException(nameof(layout));
        if (!requireProjection && layout is ClipCheckpointLayout.ClipG or ClipCheckpointLayout.SdxlG)
            throw new ArgumentException("The explicitly selected G encoder requires its pooled projection.", nameof(requireProjection));
        var schema = ClipWeightSchema.Describe(config);
        var mappings = new List<ClipWeightMapping>();
        var ignored = new List<string>();
        var claimed = new Dictionary<string, string>(StringComparer.Ordinal);
        string prefix = layout switch
        {
            ClipCheckpointLayout.ClipL => "clip_l.", ClipCheckpointLayout.ClipG => "clip_g.",
            ClipCheckpointLayout.Sd1 => "cond_stage_model.",
            ClipCheckpointLayout.SdxlL => "conditioner.embedders.0.",
            ClipCheckpointLayout.SdxlG => "conditioner.embedders.1.model.", _ => ""
        };
        bool openClip = layout is ClipCheckpointLayout.OpenClip or ClipCheckpointLayout.SdxlG;
        foreach (var (source, info) in file.Tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (!source.StartsWith(prefix, StringComparison.Ordinal)) continue;
            string local = source[prefix.Length..];
            if (layout is ClipCheckpointLayout.ClipL or ClipCheckpointLayout.ClipG or ClipCheckpointLayout.Sd1 or ClipCheckpointLayout.SdxlL)
            {
                if (local == "logit_scale") { ignored.Add(source); continue; }
                if (!local.StartsWith("transformer.", StringComparison.Ordinal))
                    throw new InvalidDataException($"Unknown weight '{source}' inside selected CLIP namespace '{prefix}'.");
                local = local["transformer.".Length..];
                if (layout == ClipCheckpointLayout.Sd1 &&
                    (local.StartsWith("embeddings.", StringComparison.Ordinal) || local.StartsWith("encoder.", StringComparison.Ordinal) || local.StartsWith("final_layer_norm.", StringComparison.Ordinal)))
                    local = "text_model." + local;
            }
            if (local is "logit_scale" or "text_model.embeddings.position_ids") { ignored.Add(source); continue; }
            if (schema.TryGetValue(local, out var directShape))
            {
                Add(local, ClipWeightTransform.Identity, 0, directShape); continue;
            }
            if (local == "text_projection")
            {
                Add(ClipWeightSchema.Projection, ClipWeightTransform.Transpose, 0, schema[ClipWeightSchema.Projection]); continue;
            }
            if (openClip)
            {
                string? target = local switch
                {
                    "positional_embedding" => "text_model.embeddings.position_embedding.weight",
                    "token_embedding.weight" => "text_model.embeddings.token_embedding.weight",
                    "ln_final.weight" => "text_model.final_layer_norm.weight",
                    "ln_final.bias" => "text_model.final_layer_norm.bias", _ => null
                };
                var match = OpenClipBlock().Match(local);
                if (match.Success)
                {
                    string block = "text_model.encoder.layers." + match.Groups[1].Value + ".";
                    string suffix = match.Groups[2].Value;
                    if (suffix is "attn.in_proj_weight" or "attn.in_proj_bias")
                    {
                        bool weight = suffix.EndsWith("weight", StringComparison.Ordinal);
                        long[] sourceShape = weight ? [3L * config.HiddenSize, config.HiddenSize] : [3L * config.HiddenSize];
                        foreach (var (part, index) in new[] { ("q", 0), ("k", 1), ("v", 2) })
                            Add(block + "self_attn." + part + "_proj." + (weight ? "weight" : "bias"),
                                ClipWeightTransform.QkvSlice, checked(index * config.HiddenSize), sourceShape);
                        continue;
                    }
                    foreach (var (from, to) in new[] { ("ln_1.", "layer_norm1."), ("ln_2.", "layer_norm2."),
                        ("mlp.c_fc.", "mlp.fc1."), ("mlp.c_proj.", "mlp.fc2."), ("attn.out_proj.", "self_attn.out_proj.") })
                        if (suffix.StartsWith(from, StringComparison.Ordinal)) target = block + to + suffix[from.Length..];
                }
                if (target is not null && schema.TryGetValue(target, out var targetShape))
                {
                    Add(target, ClipWeightTransform.Identity, 0, targetShape); continue;
                }
            }
            throw new InvalidDataException($"Unknown weight '{source}' inside selected CLIP namespace '{prefix}'. Quantization, adapters and other architectures are unsupported.");

            void Add(string canonical, ClipWeightTransform transform, int slice, IReadOnlyList<long> sourceShape)
            {
                if (!schema.TryGetValue(canonical, out var expected)) throw new InvalidDataException($"Unexpected CLIP layer or parameter '{source}'.");
                if (!claimed.TryAdd(canonical, source))
                    throw new InvalidDataException($"CLIP alias collision: '{claimed[canonical]}' and '{source}' both map to '{canonical}'.");
                if (info.DType is not ("F32" or "F16" or "BF16"))
                    throw new InvalidDataException($"CLIP weight '{source}' uses unsupported storage dtype '{info.DType}'; only F32/F16/BF16 are supported.");
                ClipWeightSchema.CheckShape(source, info.Shape, sourceShape);
                mappings.Add(new(source, canonical, transform, slice, Array.AsReadOnly(info.Shape.ToArray()),
                    Array.AsReadOnly(expected.ToArray()), info.DType));
            }
        }
        foreach (string name in schema.Keys)
            if ((requireProjection || name != ClipWeightSchema.Projection) && !claimed.ContainsKey(name))
                throw new InvalidDataException($"Missing CLIP weight '{name}' for explicitly selected layout {layout}.");
        return new(file, config, layout, requireProjection, mappings, ignored);
    }

    public static ClipWeightSet Load(SafeTensorFile file, ClipWeightPlan plan, CancellationToken cancellationToken = default)
        => Load(file, plan, cancellationToken, null);

    internal static ClipWeightSet Load(SafeTensorFile file, ClipWeightPlan plan, CancellationToken cancellationToken,
        Action<IReadOnlyDictionary<string, torch.Tensor>>? materialized)
    {
        ArgumentNullException.ThrowIfNull(file); ArgumentNullException.ThrowIfNull(plan);
        if (!ReferenceEquals(file, plan.Source)) throw new ArgumentException("The CLIP plan must be loaded from the same open safetensors reader inspected.", nameof(file));
        cancellationToken.ThrowIfCancellationRequested();
        NativeRuntimeBootstrap.Initialize();
        var owned = new Dictionary<string, torch.Tensor>(StringComparer.Ordinal);
        try
        {
            using var noGrad = torch.no_grad();
            foreach (var source in plan.Mappings.GroupBy(m => m.SourceName))
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var scope = torch.NewDisposeScope();
                var raw = file.ReadTensor(source.Key, cancellationToken);
                var converted = raw.dtype == torch.ScalarType.Float32 ? raw : raw.to_type(torch.ScalarType.Float32);
                foreach (var mapping in source)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var canonical = mapping.Transform switch
                    {
                        ClipWeightTransform.Identity => converted,
                        ClipWeightTransform.Transpose => converted.transpose(0, 1).contiguous(),
                        ClipWeightTransform.QkvSlice => converted.narrow(0, mapping.SliceStart, plan.Config.HiddenSize).clone(),
                        _ => throw new InvalidOperationException("Unknown CLIP weight transform.")
                    };
                    owned.Add(mapping.CanonicalName, canonical);
                    canonical.DetachFromDisposeScope();
                    materialized?.Invoke(owned);
                }
            }
            cancellationToken.ThrowIfCancellationRequested();
            var result = ClipWeightSet.FromOwnedTensors(plan.Config, owned, plan.RequireProjection);
            owned.Clear();
            return result;
        }
        finally { foreach (var tensor in owned.Values) tensor.Dispose(); }
    }

    [GeneratedRegex(@"^transformer\.resblocks\.([0-9]+)\.(.+)$", RegexOptions.CultureInvariant)]
    private static partial Regex OpenClipBlock();
}
