using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;
using Xunit;
using static ComfySharp.Testing.PngFixtureReader;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class ImageFileNodeTests
{
    private static JsonObject Graph(string prefix = "album/image_%width%x%height%")
    {
        var graph = JsonNode.Parse("""
            {"red":{"class_type":"EmptyImage","inputs":{"width":2,"height":1,"batch_size":1,"color":16711680}},
             "blue":{"class_type":"EmptyImage","inputs":{"width":2,"height":1,"batch_size":1,"color":255}},
             "batch":{"class_type":"ImageBatch","inputs":{"image1":["red",0],"image2":["blue",0]}},
             "save":{"class_type":"SaveImage","inputs":{"images":["batch",0],"filename_prefix":"unused"}},
             "preview":{"class_type":"PreviewImage","inputs":{"images":["save",0]}}}
            """)!.AsObject();
        graph["save"]!["inputs"]!["filename_prefix"] = prefix; return graph;
    }
    private static EngineService Engine(MemoryFiles files, bool disableMetadata = false)
    {
        var registry = TensorNodes.CreateRegistry(); ImageFileNodes.Register(registry, files, disableMetadata: disableMetadata);
        return new(registry);
    }

    [Fact]
    public async Task Saves_each_frame_with_known_pixels_metadata_and_the_same_borrowed_image_output()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        var files = new MemoryFiles(); using var engine = Engine(files); var prompt = Graph();
        var extra = JsonNode.Parse("""{"extra_pnginfo":{"workflow":{"version":1,"unknown":["猫",null]},"prompt":{"shadow":true},"note":null}}""")!.AsObject();
        using (var result = await engine.ExecuteValuesAsync(prompt, ["batch", "save", "preview"], extraData: extra))
        {
            Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
            var batch = result.Outputs["batch"][0][0].GetNative<Tensor>();
            Assert.Same(batch, result.Outputs["save"][0][0].GetNative<Tensor>());
            Assert.Same(batch, result.Outputs["preview"][0][0].GetNative<Tensor>());
            Assert.Equal(new long[] { 2, 1, 2, 3 }, batch.shape);
            Assert.Equal(4, files.Writes.Count);
            foreach (string type in new[] { "output", "temp" })
            {
                var saved = files.Writes.Where(p => p.File.Type == type).ToArray(); Assert.Equal(2, saved.Length);
                for (int index = 0; index < saved.Length; index++)
                {
                    var decoded = Read(saved[index].Png); Assert.Equal(2, decoded.Width); Assert.Equal(1, decoded.Height);
                    Assert.Equal(index == 0 ? new byte[] { 255,0,0,255,0,0 } : new byte[] { 0,0,255,0,0,255 }, decoded.Pixels);
                    Assert.Equal(new[] { "prompt", "workflow", "prompt", "note" }, decoded.Text.Select(t => t.Keyword));
                    Assert.True(JsonNode.DeepEquals(prompt, JsonNode.Parse(decoded.Text[0].Text)));
                    Assert.True(JsonNode.DeepEquals(extra["extra_pnginfo"]!["workflow"], JsonNode.Parse(decoded.Text[1].Text)));
                    Assert.True(JsonNode.Parse(decoded.Text[2].Text)!["shadow"]!.GetValue<bool>());
                    Assert.Equal("null", decoded.Text[3].Text);
                }
            }
            Assert.Equal("image_2x1_00001_.png", files.Writes[0].File.Filename);
            Assert.Equal("album", files.Writes[0].File.Subfolder);
            Assert.Matches("^ComfyUI_temp_[abcdefghijklmnopqrstupvxyz]{5}_00001_\\.png$", files.Writes[2].File.Filename);
            Assert.Empty(files.Writes[2].File.Subfolder);
            Assert.Equal(2, result.UiOutputs["save"]["images"]!.AsArray().Count);
            Assert.Equal("output", result.UiOutputs["save"]["images"]![0]!["type"]!.GetValue<string>());
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public async Task Preview_prefix_is_stable_for_the_cached_node_and_its_counter_advances()
    {
        var files = new MemoryFiles(); using var engine = Engine(files); var prompt = Graph();
        Assert.Equal("success", (await engine.ExecuteUiAsync(prompt)).Status);
        string first = files.Writes.First(p => p.File.Type == "temp").File.Filename;
        string stem = first[..^11]; // remove _00001_.png
        Assert.Equal("success", (await engine.ExecuteUiAsync(prompt)).Status);
        var previews = files.Writes.Where(p => p.File.Type == "temp").ToArray();
        Assert.Equal(4, previews.Length);
        Assert.Equal(stem + "_00003_.png", previews[2].File.Filename);
        Assert.Equal(stem + "_00004_.png", previews[3].File.Filename);
    }

    [Fact]
    public async Task Explicit_metadata_disable_omits_text_even_when_extra_metadata_is_not_an_object()
    {
        var files = new MemoryFiles(); using var engine = Engine(files, disableMetadata: true);
        var result = await engine.ExecuteUiAsync(Graph(), extraData: new JsonObject { ["extra_pnginfo"] = "ignored by explicit setting" });
        Assert.Equal("success", result.Status); Assert.Equal(4, files.Writes.Count);
        Assert.All(files.Writes, file => Assert.Empty(Read(file.Png).Text));
    }

    [Fact]
    public async Task Invalid_metadata_reports_failure_without_fabricating_files_or_ui_outputs()
    {
        var files = new MemoryFiles(); using var engine = Engine(files);
        var result = await engine.ExecuteUiAsync(Graph(), extraData: new JsonObject { ["extra_pnginfo"] = 1 });
        Assert.Equal("error", result.Status); Assert.Empty(files.Writes); Assert.Empty(result.Outputs);
        Assert.Contains(result.Diagnostics, d => d.Message.Contains("extra_pnginfo", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Write_failure_keeps_previous_file_and_releases_the_image_without_successful_ui()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        var files = new MemoryFiles { FailWrite = 2 }; using var engine = Engine(files);
        var result = await engine.ExecuteUiAsync(Graph(), ["save"]);
        Assert.Equal("error", result.Status); Assert.Single(files.Writes); Assert.Empty(result.Outputs);
        Assert.Contains(result.Diagnostics, d => d.Message.Contains("test write failure", StringComparison.Ordinal));
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public async Task Cancellation_after_a_write_leaves_that_file_and_releases_native_inputs()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; using var cancel = new CancellationTokenSource();
        var files = new MemoryFiles { AfterWrite = cancel.Cancel }; using var engine = Engine(files);
        var result = await engine.ExecuteUiAsync(Graph(), ["save"], cancellationToken: cancel.Token);
        Assert.Equal("cancelled", result.Status); Assert.Single(files.Writes); Assert.Empty(result.Outputs);
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory]
    [InlineData("SaveImage")]
    [InlineData("PreviewImage")]
    public void Schemas_preserve_hidden_inputs_and_image_output_without_an_optional_group(string type)
    {
        using var engine = Engine(new MemoryFiles()); var schema = engine.Registry.ToObjectInfo()[type]!;
        Assert.Equal("images", schema["output_name"]![0]!.GetValue<string>());
        Assert.Equal("IMAGE", schema["output"]![0]!.GetValue<string>());
        Assert.True(schema["output_node"]!.GetValue<bool>());
        Assert.Equal("{\"prompt\":\"PROMPT\",\"extra_pnginfo\":\"EXTRA_PNGINFO\"}", schema["input"]!["hidden"]!.ToJsonString());
        Assert.Equal("[\"prompt\",\"extra_pnginfo\"]", schema["input_order"]!["hidden"]!.ToJsonString());
        Assert.False(schema["input"]!.AsObject().ContainsKey("optional"));
        Assert.False(schema["input_order"]!.AsObject().ContainsKey("optional"));
        if (type == "SaveImage") Assert.Equal("ComfyUI", schema["input"]!["required"]!["filename_prefix"]![1]!["default"]!.GetValue<string>());
    }

    private sealed class MemoryFiles : IImageFileStore
    {
        public List<(ImageFileDescriptor File, byte[] Png)> Writes { get; } = [];
        public int FailWrite { get; init; }
        public Action? AfterWrite { get; init; }
        public IReadOnlyList<string> PrepareDirectory(string type, string subfolder, CancellationToken cancellationToken = default) =>
            Writes.Where(p => p.File.Type == type && p.File.Subfolder == subfolder).Select(p => p.File.Filename).Distinct().ToArray();
        public async ValueTask WriteAsync(ImageFileDescriptor file, ReadOnlyMemory<byte> png, CancellationToken cancellationToken = default)
        {
            await Task.Yield(); cancellationToken.ThrowIfCancellationRequested();
            if (FailWrite != 0 && Writes.Count + 1 == FailWrite) throw new IOException("test write failure");
            Writes.Add((file, png.ToArray())); AfterWrite?.Invoke();
        }
    }
}
