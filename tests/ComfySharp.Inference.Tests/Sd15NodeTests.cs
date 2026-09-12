using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class Sd15NodeTests
{
    private static NodeRegistry Registry(string? directory = null)
    {
        var registry = new NodeRegistry(); Sd15Nodes.Register(registry, new CheckpointFiles(directory), cpuThreads: 1); return registry;
    }

    [Fact]
    public void Schemas_keep_component_output_order_and_unsigned_seed_range_without_model_files()
    {
        var info = Registry().ToObjectInfo(); Assert.Equal(6, info.Count);
        Assert.Equal(new[] { "MODEL", "CLIP", "VAE" }, info["CheckpointLoaderSimple"]!["output"]!.AsArray().Select(n => n!.GetValue<string>()));
        Assert.Empty(info["CheckpointLoaderSimple"]!["input"]!["required"]!["ckpt_name"]![0]!.AsArray());
        Assert.Equal(ulong.MaxValue, info["KSampler"]!["input"]!["required"]!["seed"]![1]!["max"]!.GetValue<ulong>());
        Assert.Contains("uni_pc_bh2", info["KSampler"]!["input"]!["required"]!["sampler_name"]![0]!.AsArray().Select(n => n!.GetValue<string>()));
        Assert.False(info["CLIPTextEncode"]!["output_is_list"]![0]!.GetValue<bool>());
    }

    [Fact]
    public async Task Empty_latent_preserves_source_floor_dimensions_metadata_and_releases_native_storage()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var engine = new EngineService(Registry()))
        using (var result = await engine.ExecuteValuesAsync(JsonNode.Parse("""
            {"latent":{"class_type":"EmptyLatentImage","inputs":{"width":35,"height":47,"batch_size":2}}}
            """)!.AsObject(), ["latent"]))
        {
            Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
            var latent = result.Outputs["latent"][0][0].Properties;
            var samples = latent["samples"].GetNative<Tensor>();
            Assert.Equal(new long[] { 2, 4, 5, 4 }, samples.shape);
            Assert.Equal(ScalarType.Float32, samples.dtype);
            using var nonzero = samples.count_nonzero(); Assert.Equal(0, nonzero.item<long>());
            Assert.Equal(8, latent["downscale_ratio_spacial"].ToJson()!.GetValue<int>());
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public async Task Empty_latent_refuses_excessive_allocation_before_allocating()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using var engine = new EngineService(Registry());
        using var result = await engine.ExecuteValuesAsync(JsonNode.Parse("""
            {"latent":{"class_type":"EmptyLatentImage","inputs":{"width":16384,"height":16384,"batch_size":4096}}}
            """)!.AsObject(), ["latent"]);
        Assert.Equal("error", result.Status);
        Assert.Contains(result.Diagnostics, d => d.Message.Contains("256 MiB", StringComparison.Ordinal));
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory]
    [InlineData("heunpp2", "karras", 1.0)]
    [InlineData("euler", "normal", 1.0)]
    public async Task Unported_sampler_modes_fail_explicitly_without_accessing_model_inputs(string sampler, string scheduler, double denoise)
    {
        var registry = Registry(); Assert.True(registry.TryGet("KSampler", out var node));
        using var context = new RuntimeNodeContext();
        var inputs = new Dictionary<string, RuntimeValue>
        {
            ["sampler_name"] = context.Json(JsonValue.Create(sampler)), ["scheduler"] = context.Json(JsonValue.Create(scheduler)),
            ["denoise"] = context.Json(JsonValue.Create(denoise))
        };
        var error = await Assert.ThrowsAsync<NotSupportedException>(async () => await node.ExecuteAsync(context, inputs, default));
        Assert.Contains("Euler/Karras", error.Message);
    }

    [Fact]
    public async Task Cancellation_precedes_model_paths_and_native_allocation()
    {
        var registry = Registry(); using var context = new RuntimeNodeContext();
        foreach (var node in registry.Nodes)
            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
                await node.ExecuteAsync(context, new Dictionary<string, RuntimeValue>(), new CancellationToken(true)));
    }

    [Theory]
    [InlineData("../outside.safetensors")]
    [InlineData("nested/../../outside.safetensors")]
    [InlineData("/outside.safetensors")]
    [InlineData("C:/outside.safetensors")]
    [InlineData("nested\\outside.safetensors")]
    [InlineData("nested//outside.safetensors")]
    [InlineData(".. /outside.safetensors")]
    [InlineData(".../outside.safetensors")]
    [InlineData("model.ckpt")]
    public void Catalogue_rejects_escaping_or_unsupported_names(string name)
    {
        var files = new CheckpointFiles(Path.GetTempPath());
        Assert.Throws<ArgumentException>(() => files.Resolve(name));
    }

    [Fact]
    public void Catalogue_reads_existing_nested_files_without_creating_or_changing_weights()
    {
        string directory = Path.Combine(Path.GetTempPath(), "comfysharp-model-catalog-" + Guid.NewGuid().ToString("N"));
        try
        {
            var files = new CheckpointFiles(directory); Assert.Empty(files.Names()); Assert.False(Directory.Exists(directory));
            Directory.CreateDirectory(Path.Combine(directory, "checkpoints", "nested"));
            string path = Path.Combine(directory, "checkpoints", "nested", "model.safetensors");
            File.WriteAllBytes(path, [1, 2, 3]); File.WriteAllText(Path.Combine(directory, "checkpoints", "ignored.txt"), "ignored");
            Assert.Equal(new[] { "nested/model.safetensors" }, files.Names());
            Assert.Equal(path, files.Resolve("nested/model.safetensors"));
            Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(path));
            Assert.Equal(2, Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories).Count());
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
    }
}
