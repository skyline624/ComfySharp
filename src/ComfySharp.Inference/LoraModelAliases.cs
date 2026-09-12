using ComfySharp.Tokenization;

namespace ComfySharp.Inference;

/// <summary>Frozen LoRA names for the implemented plain SD U-Net and a single CLIP encoder.
/// Includes one-dimensional weights for additive normalization adapters.</summary>
public static class LoraModelAliases
{
    public static IReadOnlyList<LoraAlias> ForUnet(SdUnetConfig config, string component = "model")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(component);
        var schema = UnetWeightSchema.Define(config);
        var aliases = new Dictionary<string, LoraTarget>(StringComparer.Ordinal);
        foreach (var (weight, shape) in schema.Shapes)
        {
            if (!Eligible(weight, shape)) continue;
            string stem = weight[..^7];
            var target = new LoraTarget(component, weight, shape);
            aliases["lora_unet_" + stem.Replace('.', '_')] = target;
            aliases["diffusion_model." + stem] = target;
        }
        foreach (var (diffusers, mapping) in schema.Diffusers)
        {
            var shape = schema.Shapes[mapping.Canonical];
            if (!Eligible(mapping.Canonical, shape)) continue;
            string stem = diffusers[..^7];
            var target = new LoraTarget(component, mapping.Canonical, shape);
            aliases["lora_unet_" + stem.Replace('.', '_')] = target;
            aliases["lycoris_" + stem.Replace('.', '_')] = target;
            string processor = stem.Replace(".to_", ".processor.to_", StringComparison.Ordinal);
            if (processor.EndsWith(".to_out.0", StringComparison.Ordinal)) processor = processor[..^2];
            aliases[processor] = target;
            aliases["unet." + processor] = target;
        }
        return Snapshot(aliases);
    }

    /// <summary>Single encoder mapping. SDXL's composite L+G map needs joint encoder context
    /// and must not be constructed by concatenating two calls to this method.</summary>
    public static IReadOnlyList<LoraAlias> ForClip(ClipTextConfig config, ClipProfile profile,
        bool includeProjection = true, string component = "clip")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(component);
        if (!Enum.IsDefined(profile)) throw new ArgumentOutOfRangeException(nameof(profile));
        var schema = ClipWeightSchema.Describe(config, includeProjection);
        bool giant = profile == ClipProfile.SdXlG;
        string wrapper = giant ? "clip_g" : "clip_l";
        var aliases = new Dictionary<string, LoraTarget>(StringComparer.Ordinal);
        foreach (var (weight, shape) in schema)
            if (Eligible(weight, shape))
                aliases[$"text_encoders.{wrapper}.transformer.{weight[..^7]}"] = new(component, weight, shape);
        // Preserve LORA_CLIP_MAP insertion order, independent of the graph's declaration order.
        string[] paths = ["mlp.fc1", "mlp.fc2", "self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj", "self_attn.out_proj"];
        for (int layer = 0; layer < Math.Min(config.LayerCount, 32); layer++)
            foreach (string path in paths)
            {
                string stem = $"text_model.encoder.layers.{layer}.{path}";
                string weight = stem + ".weight";
                var target = new LoraTarget(component, weight, schema[weight]);
                string tail = $"text_model_encoder_layers_{layer}_{path.Replace('.', '_')}";
                aliases["lora_te_" + tail] = target;
                if (!giant) aliases["lora_te1_" + tail] = target;
                aliases["text_encoder." + stem] = target;
                if (giant) aliases["lora_prior_te_" + tail] = target;
            }
        if (includeProjection)
        {
            var target = new LoraTarget(component, ClipWeightSchema.Projection, schema[ClipWeightSchema.Projection]);
            if (giant) aliases["lora_prior_te_text_projection"] = target;
            aliases[giant ? "lora_te2_text_projection" : "lora_te1_text_projection"] = target;
        }
        return Snapshot(aliases);
    }

    private static bool Eligible(string weight, IReadOnlyList<long> shape) =>
        weight.EndsWith(".weight", StringComparison.Ordinal) && shape.Count >= 1;
    private static IReadOnlyList<LoraAlias> Snapshot(Dictionary<string, LoraTarget> aliases) =>
        Array.AsReadOnly(aliases.Select(p => new LoraAlias(p.Key, p.Value)).ToArray());
}
