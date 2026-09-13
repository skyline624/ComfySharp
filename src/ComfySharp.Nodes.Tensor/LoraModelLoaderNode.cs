using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Inference;
using System.Text.Json.Nodes;

namespace ComfySharp.Nodes.Tensor;

/// <summary>Applies an in-memory LORA_MODEL to a plain SD model using ordinary weight patches
/// or retained linear/Conv2d forward adapters.</summary>
public sealed class LoraModelLoaderNode(long maxPatchedWeightBytes = 4L * 1024 * 1024 * 1024) : IRuntimeNode
{
    public static NodeSchema Description { get; } = new("LoraModelLoader", "Load LoRA Model", "model/loaders",
        [new("model", "MODEL", Options: new() { ["tooltip"] = "The diffusion model the LoRA will be applied to." }),
         new("lora", "LORA_MODEL", Options: new() { ["tooltip"] = "The LoRA model to apply to the diffusion model." }),
         new("strength_model", "FLOAT", Options: new() { ["default"] = 1.0, ["min"] = -100.0, ["max"] = 100.0,
             ["tooltip"] = "How strongly to modify the diffusion model. This value can be negative." }),
         new("bypass", "BOOLEAN", Options: new() { ["default"] = false,
             ["tooltip"] = "When enabled, applies LoRA in bypass mode without modifying base model weights. Useful for training and when model weights are offloaded." })],
        [new("MODEL", "model")], Experimental: true, PythonModule: "comfy_extras.nodes_train", V3ObjectInfo: true, OmitEmptyOptionalInputs: true,
        Description: "Applies an in-memory adapter to a plain Float32 SD model, using ordinary patches or linear/Conv2d bypass adapters. Quantized models and other model families remain unavailable.");
    public NodeSchema Schema => Description;

    public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
        IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        double strength = PythonValues.Float(inputs["strength_model"].ToJson());
        if (!double.IsFinite(strength) || strength is < -100 or > 100) throw new ArgumentOutOfRangeException("strength_model");
        if (strength == 0) return ValueTask.FromResult(new NodeExecutionOutput([context.Retain(inputs["model"])]));
        bool useBypass = inputs.TryGetValue("bypass", out var bypass) && bypass.ToJson()!.GetValue<bool>();
        if (inputs["lora"].Kind != RuntimeValueKind.Map) throw new ArgumentException("LORA_MODEL must be a runtime map of tensors.");
        var model = inputs["model"].GetNative<SdUnet>();
        var tensors = inputs["lora"].Properties.ToDictionary(p => p.Key, p => p.Value.GetNative<TorchSharp.torch.Tensor>(), StringComparer.Ordinal);
        using var source = new NativeLoraTensorSource(tensors, cancellationToken: cancellationToken);
        var plan = LoraFileLoader.Inspect(source, LoraModelAliases.ForUnet(model.Config), allowUnclaimedKeys: true, cancellationToken: cancellationToken,
            mode: useBypass ? LoraLoadMode.Bypass : LoraLoadMode.Weights);
        using var adapter = LoraFileLoader.Load(source, plan, new Dictionary<string, double> { ["model"] = strength }, cancellationToken);
        var modified = useBypass ? adapter.ApplyBypassTo(model, maxPatchedWeightBytes: maxPatchedWeightBytes, cancellationToken: cancellationToken)
            : adapter.ApplyTo(model, maxPatchedWeightBytes: maxPatchedWeightBytes, cancellationToken: cancellationToken);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult(new NodeExecutionOutput([context.Own(modified)], new JsonObject
            {
                ["comfysharp_lora"] = new JsonArray(new JsonObject
                {
                    ["source"] = "LORA_MODEL", ["bypass"] = useBypass, ["matched_targets"] = plan.Bindings.Count,
                    ["unclaimed_tensors"] = new JsonArray(plan.UnclaimedKeys.Select(k => (JsonNode?)JsonValue.Create(k)).ToArray()),
                    ["shadowed_prefixes"] = new JsonArray(plan.ShadowedPrefixes.Select(k => (JsonNode?)JsonValue.Create(k)).ToArray())
                })
            }));
        }
        catch { modified.Dispose(); throw; }
    }
}
