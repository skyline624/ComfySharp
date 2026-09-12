namespace ComfySharp.Inference;

/// <summary>Frozen training factory selection. Deliberately distinct from inference aliases,
/// alpha keys and difference loading. Metadata only; supplied source is borrowed.</summary>
internal sealed class SdLoraResumePlan
{
    internal sealed record Factors(string Up, string Down, long Rank);
    internal Dictionary<string, Factors> Targets { get; } = new(StringComparer.Ordinal);
    internal Dictionary<string, string> Alphas { get; } = new(StringComparer.Ordinal);
    internal IReadOnlyList<string> IgnoredKeys { get; private set; } = Array.Empty<string>();

    internal static SdLoraResumePlan Inspect(ILoraTensorSource source, IReadOnlyDictionary<string,IReadOnlyList<long>> schema,
        CancellationToken token)
    {
        var plan = new SdLoraResumePlan(); var consumed = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (name, shape) in schema)
        {
            token.ThrowIfCancellationRequested();
            if (shape.Count < 2) continue; // Source always constructs fresh BiasDiff leaves.
            string prefix = "diffusion_model." + name[..^7];
            string alpha = prefix + ".weight.alpha";
            if (source.Tensors.TryGetValue(alpha, out var info))
            {
                if (!SafeTensorFile.SupportsTensorDType(info.DType) || info.Shape.Aggregate(1L,(a,b)=>checked(a*b)) != 1)
                    throw new InvalidDataException("Resume alpha must contain one readable scalar: " + alpha);
                plan.Alphas.Add(name,alpha); consumed.Add(alpha);
            }
            bool selected = false;
            foreach (var format in LoraFileLoader.Formats)
            {
                string up = prefix + format.Up, down = prefix + format.Down;
                if (!source.Tensors.TryGetValue(up, out var u)) continue;
                if (!source.Tensors.TryGetValue(down, out var d)) throw new InvalidDataException("Resume factor has no matching down matrix: " + up);
                if (format.Mid && source.Tensors.ContainsKey(prefix + ".lora_mid.weight"))
                    throw new NotSupportedException("Training resume of LoCon mid factors remains unported: " + prefix);
                long columns = shape.Skip(1).Aggregate(1L,(a,b)=>checked(a*b));
                if (!SafeTensorFile.SupportsTensorDType(u.DType) || !SafeTensorFile.SupportsTensorDType(d.DType) ||
                    u.Shape.Count != 2 || d.Shape.Count != 2 || u.Shape[0] != shape[0] || d.Shape[1] != columns ||
                    u.Shape[1] < 1 || u.Shape[1] != d.Shape[0])
                    throw new InvalidDataException("Resume matrices must match the SD target and share a positive rank: " + prefix);
                plan.Targets.Add(name,new(up,down,d.Shape[0])); consumed.Add(up); consumed.Add(down);
                selected = true; break;
            }
            // Later adapter loaders would claim these formats; never silently replace them by fresh LoRA.
            bool otherAlgorithm = new[]{".hada_w1_a", ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w2_a", ".a1.weight"}
                .Any(suffix => source.Tensors.ContainsKey(prefix + suffix)) ||
                source.Tensors.TryGetValue(prefix + ".oft_blocks",out var blocks) && blocks.Shape.Count is 3 or 4;
            if (!selected && otherAlgorithm)
                throw new NotSupportedException("Training resume requires an unported adapter algorithm: " + prefix);
        }
        plan.IgnoredKeys = Array.AsReadOnly(source.Tensors.Keys.Where(k=>!consumed.Contains(k)).Order(StringComparer.Ordinal).ToArray());
        return plan;
    }
}
