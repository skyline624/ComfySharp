using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Inference;

namespace ComfySharp.Nodes.Tensor;

/// <summary>Frozen image primitives. The initial execution profile is CPU/Float32 NHWC RGB/RGBA.</summary>
public static class ImageNodes
{
    public static void Register(NodeRegistry registry)
    {
        registry.Register(new ImageNode(new("EmptyImage", "Empty Image", "image",
            [Int("width", 512, 1, 16384, 1), Int("height", 512, 1, 16384, 1),
             Int("batch_size", 1, 1, 4096), Int("color", 0, 0, 0xffffff, 1, "color")],
            [new("IMAGE")], PythonModule: "nodes")));
        registry.Register(new ImageNode(new("ImageInvert", "Invert Image Colors", "image/color",
            [new("image", "IMAGE")], [new("IMAGE")], SearchAliases: ["reverse colors"],
            PythonModule: "nodes", EssentialsCategory: "Image Tools")));
        registry.Register(new ImageNode(new("RepeatImageBatch", "Repeat Image Batch", "image/batch",
            [new("image", "IMAGE"), Int("amount", 1, 1, 4096)], [new("IMAGE")],
            SearchAliases: ["duplicate image", "clone image"], PythonModule: "comfy_extras.nodes_images", V3ObjectInfo: true)));
        registry.Register(new ImageNode(new("ImageFromBatch", "Get Image from Batch", "image/batch",
            [new("image", "IMAGE"), Int("batch_index", 0, -16384, 16384), Int("length", 1, 1, 4096)], [new("IMAGE")],
            SearchAliases: ["select image", "pick from batch", "extract image"], PythonModule: "comfy_extras.nodes_images", V3ObjectInfo: true)));
        registry.Register(new ImageNode(new("ImageBatch", "Batch Images (DEPRECATED)", "image/batch",
            [new("image1", "IMAGE"), new("image2", "IMAGE")], [new("IMAGE")],
            SearchAliases: ["combine images", "merge images", "stack images"], PythonModule: "nodes", Deprecated: true)));
    }

    private static InputSchema Int(string name, int value, int min, int max, int? step = null, string? display = null)
    {
        var options = new JsonObject { ["default"] = value, ["min"] = min, ["max"] = max };
        if (step.HasValue) options["step"] = step.Value;
        if (display is not null) options["display"] = display;
        return new(name, "INT", Options: options);
    }

    private sealed class ImageNode(NodeSchema schema) : IRuntimeNode
    {
        public NodeSchema Schema { get; } = schema;

        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            long I(string name) => checked((long)PythonValues.Integer(inputs[name].ToJson()));
            TorchSharp.torch.Tensor Image(string name = "image")
            {
                if (inputs[name].Kind != RuntimeValueKind.Native)
                    throw new ArgumentException("IMAGE requires a native NHWC tensor.");
                return inputs[name].GetNative<TorchSharp.torch.Tensor>();
            }
            NativeRuntimeBootstrap.Initialize();
            // Helpers return a detached, independently owned tensor.
            var result = Schema.ClassType switch
            {
                "EmptyImage" => ImageOperations.EmptyImage(I("width"), I("height"), I("batch_size"), checked((int)I("color")), cancellationToken),
                "ImageInvert" => ImageOperations.Invert(Image(), cancellationToken),
                "RepeatImageBatch" => ImageOperations.RepeatBatch(Image(), I("amount"), cancellationToken),
                "ImageFromBatch" => ImageOperations.FromBatch(Image(), I("batch_index"), I("length"), cancellationToken),
                "ImageBatch" => ImageOperations.Batch(Image("image1"), Image("image2"), cancellationToken),
                _ => throw new InvalidOperationException("Unregistered image operation.")
            };
            RuntimeValue owned;
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                owned = context.Own(result);
            }
            catch { result.Dispose(); throw; }
            result.DetachFromDisposeScope();
            return ValueTask.FromResult(new NodeExecutionOutput([owned]));
        }
    }
}
