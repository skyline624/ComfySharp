using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Inference;

namespace ComfySharp.Nodes.Tensor;

/// <summary>Local safetensors adapters on implemented SD U-Net and standalone CLIP graphs.</summary>
public static class LoraNodes
{
    public static void Register(NodeRegistry registry, string? modelsDirectory)
    {
        ArgumentNullException.ThrowIfNull(registry);
        var files = new CheckpointFiles(modelsDirectory, "loras");
        foreach (var schema in Schemas(files.Names())) registry.Register(new Node(schema, files));
    }

    public static IReadOnlyList<NodeSchema> Schemas(IReadOnlyList<string> names)
    {
        InputSchema File() => new("lora_name", "COMBO", Options: new() { ["options"] = new JsonArray(names.Select(n => (JsonNode?)JsonValue.Create(n)).ToArray()) });
        InputSchema Strength(string name) => new(name, "FLOAT", Options: new() { ["default"] = 1.0, ["min"] = -100.0, ["max"] = 100.0, ["step"] = .01 });
        const string description = "Local safetensors LoRA/LoCon/DoRA for plain SD U-Net and standalone CLIP in Float32. Shared loras directory; no downloads or copies. Unclaimed keys are reported; no matching factors is an error. Other adapters, quantization, reshape_weight and composite encoders remain unavailable.";
        return [
            new("LoraLoader", "Load LoRA", "model/loaders", [new("model", "MODEL"), new("clip", "CLIP"), File(), Strength("strength_model"), Strength("strength_clip")],
                [new("MODEL"), new("CLIP")], PythonModule: "nodes", Description: description),
            new("LoraLoaderModelOnly", "Load LoRA (Model Only)", "model/loaders", [new("model", "MODEL"), File(), Strength("strength_model")],
                [new("MODEL")], PythonModule: "nodes", Description: description)
        ];
    }

    private sealed class Node(NodeSchema schema, CheckpointFiles files) : IRuntimeNode
    {
        public NodeSchema Schema => schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            bool withClip = schema.ClassType == "LoraLoader";
            double Strength(string key)
            {
                double value = PythonValues.Float(inputs[key].ToJson());
                if (!double.IsFinite(value) || value is < -100 or > 100) throw new ArgumentOutOfRangeException(key, "LoRA strength must be finite and in [-100,100].");
                return value;
            }
            double modelStrength = Strength("strength_model"), clipStrength = withClip ? Strength("strength_clip") : 0;
            if (modelStrength == 0 && clipStrength == 0)
                return ValueTask.FromResult(new NodeExecutionOutput(withClip
                    ? [context.Retain(inputs["model"]), context.Retain(inputs["clip"])] : [context.Retain(inputs["model"])]));
            string name = inputs["lora_name"].ToJson()!.GetValue<string>();
            string path = files.Resolve(name);
            var model = inputs["model"].GetNative<SdUnet>();
            var clip = withClip ? inputs["clip"].GetNative<ComfyClipEncoder>() : null;
            var aliases = LoraModelAliases.ForUnet(model.Config).ToList();
            if (clip is not null) aliases.AddRange(LoraModelAliases.ForClip(clip.Config, clip.Profile, clip.HasProjection));
            using var file = new SafeTensorFile(path);
            var plan = LoraFileLoader.Inspect(file, aliases, allowUnclaimedKeys: true, cancellationToken: cancellationToken);
            var strengths = new Dictionary<string, double> { ["model"] = modelStrength };
            if (clip is not null) strengths["clip"] = clipStrength;
            using var adapter = LoraFileLoader.Load(file, plan, strengths, cancellationToken);
            // Publish outputs only after both graph patches have completed. Originals stay unchanged.
            using var modelOutput = modelStrength == 0 ? model.Retain() : adapter.ApplyTo(model, cancellationToken: cancellationToken);
            using var clipOutput = clip is null ? null : clipStrength == 0 ? clip.Retain() : adapter.ApplyTo(clip, cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            RuntimeValue[] outputs = clipOutput is null ? [context.Own(modelOutput.Retain())]
                : [context.Own(modelOutput.Retain()), context.Own(clipOutput.Retain())];
            return ValueTask.FromResult(new NodeExecutionOutput(outputs, new JsonObject
            {
                ["comfysharp_lora"] = new JsonArray(new JsonObject
                {
                    ["name"] = name, ["matched_targets"] = plan.Bindings.Count,
                    ["unclaimed_tensors"] = new JsonArray(plan.UnclaimedKeys.Select(k => (JsonNode?)JsonValue.Create(k)).ToArray()),
                    ["shadowed_prefixes"] = new JsonArray(plan.ShadowedPrefixes.Select(k => (JsonNode?)JsonValue.Create(k)).ToArray())
                })
            }));
        }
    }
}
