using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;
using ComfySharp.Tokenization;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraNodeTests
{
    private static NodeRegistry Registry(string? root = null) { var registry = new NodeRegistry(); LoraNodes.Register(registry, root); return registry; }

    [Fact]
    public void Schema_preserves_upstream_input_output_order_and_strength_controls()
    {
        var info = Registry().ToObjectInfo();
        Assert.Equal(new[] { "model", "clip", "lora_name", "strength_model", "strength_clip" }, info["LoraLoader"]!["input_order"]!["required"]!.AsArray().Select(v => v!.GetValue<string>()));
        Assert.Equal(new[] { "MODEL", "CLIP" }, info["LoraLoader"]!["output"]!.AsArray().Select(v => v!.GetValue<string>()));
        Assert.Equal(new[] { "model", "lora_name", "strength_model" }, info["LoraLoaderModelOnly"]!["input_order"]!["required"]!.AsArray().Select(v => v!.GetValue<string>()));
        Assert.Equal(-100, info["LoraLoader"]!["input"]!["required"]!["strength_clip"]![1]!["min"]!.GetValue<double>());
        Assert.Empty(info["LoraLoader"]!["input"]!["required"]!["lora_name"]![0]!.AsArray());
    }

    [Theory]
    [InlineData("LoraLoader")]
    [InlineData("LoraLoaderModelOnly")]
    public async Task Zero_strength_retains_inputs_without_reading_filename_and_cancellation_precedes_inputs(string type)
    {
        var registry = Registry(); Assert.True(registry.TryGet(type, out var node));
        using var input = new RuntimeNodeContext(); using var output = new RuntimeNodeContext();
        var values = new Dictionary<string, RuntimeValue> { ["model"] = input.Json(JsonValue.Create("model marker")), ["clip"] = input.Json(JsonValue.Create("clip marker")),
            ["strength_model"] = input.Json(JsonValue.Create(0)), ["strength_clip"] = input.Json(JsonValue.Create(0)) };
        var result = await node.ExecuteAsync(output, values, default); input.Dispose();
        Assert.Equal(type == "LoraLoader" ? 2 : 1, result.Result.Count);
        Assert.Equal("model marker", result.Result[0].ToJson()!.GetValue<string>());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await node.ExecuteAsync(output, new Dictionary<string, RuntimeValue>(), new(true)));
    }

    [Fact]
    public async Task Shared_file_patches_both_graphs_and_chains_model_only_without_mutating_originals()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        string root = Path.Combine(Path.GetTempPath(), "comfysharp-lora-nodes-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var scope = NewDisposeScope(); using var input = new RuntimeNodeContext(); using var first = new RuntimeNodeContext(); using var second = new RuntimeNodeContext();
            Directory.CreateDirectory(Path.Combine(root, "loras", "nested"));
            string path = Path.Combine(root, "loras", "nested", "adapter.safetensors");
            Write(path, new() { ["lora_unet_conv_out.lora_up.weight"] = ([4, 1], .1f), ["lora_unet_conv_out.lora_down.weight"] = ([1, 32, 3, 3], .01f),
                ["lora_te1_text_projection.lora_up.weight"] = ([4, 1], .1f), ["lora_te1_text_projection.lora_down.weight"] = ([1, 4], .1f) });
            byte[] originalFile = File.ReadAllBytes(path);
            var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var original = new SdUnet(bank);
            var clipConfig = new ClipTextConfig(4, 8, 2, 1, ClipActivation.QuickGelu);
            using var clipBank = ClipWeightSet.FromOwnedTensors(clipConfig, ClipWeightSchema.Describe(clipConfig).ToDictionary(p => p.Key, p => full(p.Value.ToArray(), .1f)));
            using var clipGraph = new ClipTextEncoder(clipBank); using var originalClip = new ComfyClipEncoder(clipGraph, ClipProfile.Sd1L);
            var registry = Registry(root); Assert.True(registry.TryGet("LoraLoader", out var loader)); Assert.True(registry.TryGet("LoraLoaderModelOnly", out var modelOnly));
            var values = new Dictionary<string, RuntimeValue> { ["model"] = input.Own(original.Retain()), ["clip"] = input.Own(originalClip.Retain()),
                ["lora_name"] = input.Json(JsonValue.Create("nested/adapter.safetensors")), ["strength_model"] = input.Json(JsonValue.Create(1)), ["strength_clip"] = input.Json(JsonValue.Create(-1)) };
            var result = await loader.ExecuteAsync(first, values, default);
            Assert.Equal(2, result.Ui!["comfysharp_lora"]![0]!["matched_targets"]!.GetValue<int>());
            Assert.Empty(result.Ui["comfysharp_lora"]![0]!["unclaimed_tensors"]!.AsArray());
            var x = ones(new long[] { 1, 4, 4, 5 }); var t = tensor(new[] { 500f }); var conditioning = ones(new long[] { 1, 3, 16 });
            using var baseline = original.Forward(x, t, conditioning);
            using var patched = result.Result[0].GetNative<SdUnet>().Forward(x, t, conditioning);
            Assert.False(baseline.data<float>().ToArray().SequenceEqual(patched.data<float>().ToArray()));
            using var again = original.Forward(x, t, conditioning); Assert.Equal(baseline.data<float>().ToArray(), again.data<float>().ToArray());
            var tokenizer = new ComfyClipTokenizer(ClipTokenizer.CreateDefault(), ClipProfile.Sd1L); var tokens = tokenizer.Tokenize("cat");
            using var clipBefore = originalClip.Encode(tokens, new() { ProjectPooled = true });
            using var clipAfter = result.Result[1].GetNative<ComfyClipEncoder>().Encode(tokens, new() { ProjectPooled = true });
            Assert.False(clipBefore.Pooled.data<float>().ToArray().SequenceEqual(clipAfter.Pooled.data<float>().ToArray()));
            values["model"] = result.Result[0];
            var chained = await modelOnly.ExecuteAsync(second, values, default);
            Assert.Equal(2, chained.Ui!["comfysharp_lora"]![0]!["unclaimed_tensors"]!.AsArray().Count);
            first.Dispose(); input.Dispose(); original.Dispose(); bank.Dispose();
            using var twice = chained.Result[0].GetNative<SdUnet>().Forward(x, t, conditioning);
            Assert.False(patched.data<float>().ToArray().SequenceEqual(twice.data<float>().ToArray()));
            Assert.Equal(originalFile, File.ReadAllBytes(path));
            Assert.Single(Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories));
        }
        finally { set_num_threads(threads); if (Directory.Exists(root)) Directory.Delete(root, true); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Lora_category_is_separate_and_rejects_escaping_category_or_file_names()
    {
        Assert.Throws<ArgumentException>(() => new CheckpointFiles(Path.GetTempPath(), "../loras"));
        var files = new CheckpointFiles(Path.GetTempPath(), "loras");
        Assert.Throws<ArgumentException>(() => files.Resolve("../checkpoints/weights.safetensors"));
        Assert.Throws<ArgumentException>(() => files.Resolve("weights.pt"));
    }

    private static void Write(string path, Dictionary<string, (long[] Shape, float Value)> factors)
    {
        var header = new Dictionary<string, object>(); long offset = 0;
        foreach (var (name, factor) in factors)
        {
            long end = offset + factor.Shape.Aggregate(1L, (n, d) => n * d) * 4;
            header[name] = new { dtype = "F32", shape = factor.Shape, data_offsets = new[] { offset, end } }; offset = end;
        }
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(header); using var stream = File.Create(path); using var writer = new BinaryWriter(stream);
        writer.Write((ulong)json.Length); writer.Write(json);
        foreach (var factor in factors.Values)
            for (long i = 0; i < factor.Shape.Aggregate(1L, (n, d) => n * d); i++) writer.Write(factor.Value);
    }
}
