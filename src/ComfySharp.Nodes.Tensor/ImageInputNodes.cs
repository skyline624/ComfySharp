using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Inference;
using static TorchSharp.torch;

namespace ComfySharp.Nodes.Tensor;

public static class ImageInputNodes
{
    public static NodeSchema Describe(IReadOnlyList<string> names) => new("LoadImage", "Load Image", "image",
        [new("image", "COMBO", Options: new() { ["options"] = new JsonArray(names.Select(n => (JsonNode?)JsonValue.Create(n)).ToArray()), ["image_upload"] = true })],
        [new("IMAGE"), new("MASK")], PythonModule: "nodes", EssentialsCategory: "Basics",
        Description: "Native image decoding with RGB frames and inverse-alpha MASK. Codec coverage and source numerical parity remain under qualification.");
    public static void Register(NodeRegistry registry, IImageInputService service) => registry.Register(new Node(service));

    private sealed class Node(IImageInputService service) : IRuntimeNode
    {
        public NodeSchema Schema => Describe(service.Names());
        public async ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (decoded, hash) = await service.ReadAsync(inputs["image"].ToJson()!.GetValue<string>(), cancellationToken);
            cancellationToken.ThrowIfCancellationRequested(); NativeRuntimeBootstrap.Initialize();
            using var scope = NewDisposeScope();
            using var noGrad = no_grad();
            int pixels = checked(decoded.Frames * decoded.Width * decoded.Height);
            if (decoded.Frames <= 0 || decoded.Width <= 0 || decoded.Height <= 0 || decoded.Rgb.Length != checked(pixels * 3) || decoded.Alpha is { } alpha && alpha.Length != pixels)
                throw new InvalidDataException("Image decoder returned inconsistent dimensions.");
            var image = tensor(decoded.Rgb, dtype: ScalarType.Float32, device: CPU).reshape(decoded.Frames, decoded.Height, decoded.Width, 3);
            var mask = decoded.Alpha is null ? zeros(new long[] { decoded.Frames, 64, 64 }, dtype: ScalarType.Float32, device: CPU)
                : 1.0 - tensor(decoded.Alpha, dtype: ScalarType.Float32, device: CPU).reshape(decoded.Frames, decoded.Height, decoded.Width);
            var values = new[] { context.Own(image), context.Own(mask) }; image.DetachFromDisposeScope(); mask.DetachFromDisposeScope();
            return new(values, new JsonObject { ["comfysharp_image_source"] = new JsonArray(new JsonObject { ["sha256"] = hash, ["frames"] = decoded.Frames }) });
        }
    }
}
