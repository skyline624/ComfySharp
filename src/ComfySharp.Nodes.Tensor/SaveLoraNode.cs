using System.Globalization;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Inference;

namespace ComfySharp.Nodes.Tensor;

/// <summary>Frozen SaveLoRA contract. LORA_MODEL is a runtime map of detached tensor values,
/// distinct from the MODEL graph that consumes an adapter.</summary>
public sealed class SaveLoraNode(IStreamingFileStore store, TimeProvider? timeProvider = null) : IRuntimeNode
{
    public static NodeSchema Description { get; } = new("SaveLoRA", "Save LoRA Weights", "model/merging",
        [new("lora", "LORA_MODEL", Options: new() { ["tooltip"] = "The LoRA model to save. Do not use the model with LoRA layers." }),
         new("prefix", "STRING", Options: new() { ["default"] = "loras/ComfyUI_trained_lora", ["tooltip"] = "The prefix to use for the saved LoRA file." }),
         new("steps", "INT", Required: false, Options: new() { ["tooltip"] = "Optional: The number of steps the LoRA has been trained for, used to name the saved file." })],
        [], OutputNode: true, Experimental: true, SearchAliases: ["export lora"], PythonModule: "comfy_extras.nodes_train",
        Description: "Saves a LORA_MODEL tensor map to local safetensors, retaining dtype and values. Publishes the completed file atomically after checking cancellation.",
        V3ObjectInfo: true, OmitEmptyOptionalInputs: true);
    public NodeSchema Schema => Description;

    public async ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
        IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (inputs["lora"].Kind != RuntimeValueKind.Map) throw new ArgumentException("LORA_MODEL requires a runtime map of tensors, not a patched MODEL or JSON object.");
        var tensors = inputs["lora"].Properties.ToDictionary(p => p.Key, p => p.Value.GetNative<TorchSharp.torch.Tensor>(), StringComparer.Ordinal);
        string prefix = inputs["prefix"].ToJson()!.GetValue<string>();
        var plan = ImageFileNaming.Prepare(store, "output", prefix, 0, 0, (timeProvider ?? TimeProvider.System).GetLocalNow(), cancellationToken);
        string suffix = "";
        if (inputs.TryGetValue("steps", out var steps) && steps.ToJson() is { } value)
            suffix = PythonValues.Integer(value).ToString(CultureInfo.InvariantCulture) + "_steps_";
        var file = new ImageFileDescriptor(plan.Filename + "_" + suffix + ImageFileNaming.FormatCounter(plan.Counter) + "_.safetensors", plan.Subfolder, "output");
        await store.WriteAtomicAsync(file, (stream, token) => SafeTensorWriter.Write(stream, tensors, cancellationToken: token), cancellationToken);
        return new([]);
    }
}
