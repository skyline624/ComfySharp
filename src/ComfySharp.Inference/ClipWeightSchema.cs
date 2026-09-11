using System.Collections.ObjectModel;

namespace ComfySharp.Inference;

/// <summary>Exact learned parameter names and PyTorch [out,in] shapes of the text transformer.</summary>
public static class ClipWeightSchema
{
    public const string Projection = "text_projection.weight";

    public static IReadOnlyDictionary<string, IReadOnlyList<long>> Describe(ClipTextConfig config, bool includeProjection = true)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        var shapes = new Dictionary<string, IReadOnlyList<long>>(StringComparer.Ordinal);
        void Add(string name, params long[] shape) => shapes.Add(name, Array.AsReadOnly(shape));
        long h = config.HiddenSize, m = config.IntermediateSize;
        Add("text_model.embeddings.token_embedding.weight", ClipTextConfig.VocabularySize, h);
        Add("text_model.embeddings.position_embedding.weight", ClipTextConfig.MaxPositions, h);
        for (int i = 0; i < config.LayerCount; i++)
        {
            string prefix = $"text_model.encoder.layers.{i}.";
            foreach (string norm in new[] { "layer_norm1", "layer_norm2" })
            {
                Add(prefix + norm + ".weight", h);
                Add(prefix + norm + ".bias", h);
            }
            foreach (string projection in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
            {
                Add(prefix + "self_attn." + projection + ".weight", h, h);
                Add(prefix + "self_attn." + projection + ".bias", h);
            }
            Add(prefix + "mlp.fc1.weight", m, h);
            Add(prefix + "mlp.fc1.bias", m);
            Add(prefix + "mlp.fc2.weight", h, m);
            Add(prefix + "mlp.fc2.bias", h);
        }
        Add("text_model.final_layer_norm.weight", h);
        Add("text_model.final_layer_norm.bias", h);
        if (includeProjection) Add(Projection, h, h);
        return new ReadOnlyDictionary<string, IReadOnlyList<long>>(shapes);
    }

    internal static void CheckShape(string name, IReadOnlyList<long> actual, IReadOnlyList<long> expected)
    {
        if (!actual.SequenceEqual(expected))
            throw new InvalidDataException($"CLIP weight '{name}' has shape [{string.Join(',', actual)}]; expected [{string.Join(',', expected)}].");
    }
}
