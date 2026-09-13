using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Inference;

namespace ComfySharp.Nodes.Tensor;

/// <summary>SaveImage and PreviewImage with an explicitly configured local image store.</summary>
public static class ImageFileNodes
{
    private const string PreviewAlphabet = "abcdefghijklmnopqrstupvxyz";
    public static IReadOnlyList<NodeSchema> Schemas { get; } = Array.AsReadOnly(new[] { Describe(false), Describe(true) });
    internal static IRuntimeNode CreatePreview(IImageFileStore store,TimeProvider? time=null,bool disableMetadata=false)
        =>new FileNode(store,true,time??TimeProvider.System,disableMetadata);
    public static void Register(NodeRegistry registry, IImageFileStore store, TimeProvider? timeProvider = null,
        bool disableMetadata = false)
    {
        ArgumentNullException.ThrowIfNull(store);
        registry.Register(new FileNode(store, false, timeProvider ?? TimeProvider.System, disableMetadata));
        registry.Register(new FileNode(store, true, timeProvider ?? TimeProvider.System, disableMetadata));
    }

    private static NodeSchema Describe(bool preview) => new(
        preview ? "PreviewImage" : "SaveImage", preview ? "Preview Image" : "Save Image", "image",
        preview ? [new("images", "IMAGE")] :
        [new("images", "IMAGE", Options: new() { ["tooltip"] = "The images to save." }),
         new("filename_prefix", "STRING", Options: new()
         {
             ["default"] = "ComfyUI",
             ["tooltip"] = "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."
         })],
        [new("IMAGE", "images")], OutputNode: true,
        Description: preview ? "Preview the images without saving them to the ComfyUI output directory." : "Saves the input images to your ComfyUI output directory.",
        SearchAliases: preview ? ["preview", "preview image", "show image", "view image", "display image", "image viewer"]
            : ["save", "save image", "export image", "output image", "write image", "download"],
        PythonModule: "nodes", EssentialsCategory: "Basics",
        HiddenInputs: [new("prompt", "PROMPT"), new("extra_pnginfo", "EXTRA_PNGINFO")], OmitEmptyOptionalInputs: true);

    private sealed class FileNode(IImageFileStore store, bool preview, TimeProvider time, bool disableMetadata) : IRuntimeNodeFactory
    {
        private readonly Lazy<string> suffix = new(() => preview
            ? "_temp_" + string.Concat(Enumerable.Range(0, 5).Select(_ => PreviewAlphabet[RandomNumberGenerator.GetInt32(PreviewAlphabet.Length)])) : "");
        public NodeSchema Schema => Schemas[preview ? 1 : 0];
        public IRuntimeNode CreateInstance() => new FileNode(store, preview, time, disableMetadata);

        public async ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string prefix = (preview ? "ComfyUI" : inputs["filename_prefix"].ToJson()!.GetValue<string>()) + suffix.Value;
            if (inputs["images"].Kind != RuntimeValueKind.Native)
                throw new ArgumentException("IMAGE requires a native NHWC tensor.");
            var images = inputs["images"].GetNative<TorchSharp.torch.Tensor>();
            long[] shape = images.shape;
            if (shape.Length != 4 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0)
                throw new ArgumentException("Saving images requires a positive NHWC batch.");
            string type = preview ? "temp" : "output";
            var plan = ImageFileNaming.Prepare(store, type, prefix, checked((int)shape[2]), checked((int)shape[1]), time.GetLocalNow(), cancellationToken);
            var metadata = new List<PngText>();
            if (!disableMetadata)
            {
                if (inputs.TryGetValue("prompt", out var prompt) && prompt.ToJson() is { } original)
                    metadata.Add(new("prompt", original.ToJsonString()));
                if (inputs.TryGetValue("extra_pnginfo", out var extra) && extra.ToJson() is { } pngInfo)
                {
                    if (pngInfo is not JsonObject entries) throw new ArgumentException("extra_pnginfo must be an object or null.");
                    foreach (var (key, value) in entries)
                        metadata.Add(new(key, value?.ToJsonString() ?? "null"));
                }
            }
            var files = new JsonArray();
            for (long index = 0; index < shape[0]; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                byte[] png = ImagePngEncoder.EncodeFrame(images, index, metadata, preview ? 1 : 4, cancellationToken);
                var file = plan.Frame(type, index);
                await store.WriteAsync(file, png, cancellationToken);
                files.Add(JsonSerializer.SerializeToNode(file));
            }
            // The image input remains borrowed; the engine retains a separate lease for this output slot.
            return new([inputs["images"]], new JsonObject { ["images"] = files });
        }
    }
}
