using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Nodes;
using ComfySharp.Host;
using ComfySharp.Workflow;
using Microsoft.AspNetCore.Mvc.Testing;

namespace ComfySharp.Host.Tests;

/// <summary>Actual HTTP schema/admission and compiled image-to-text routes, not a source numerical oracle.</summary>
public sealed class ImagePrimitiveHostTests
{
    [Theory]
    [InlineData("EmptyImage", "Empty Image", "image", "nodes")]
    [InlineData("ImageInvert", "Invert Image Colors", "image/color", "nodes")]
    [InlineData("RepeatImageBatch", "Repeat Image Batch", "image/batch", "comfy_extras.nodes_images")]
    [InlineData("ImageFromBatch", "Get Image from Batch", "image/batch", "comfy_extras.nodes_images")]
    public async Task ActualObjectInfoPreservesRequiredImageSchemas(string type, string display, string category, string module)
    {
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        var info = (await client.GetFromJsonAsync<JsonObject>($"/object_info/{type}"))![type]!;
        Assert.Equal(type, info["name"]!.GetValue<string>()); Assert.Equal(display, info["display_name"]!.GetValue<string>());
        Assert.Equal(category, info["category"]!.GetValue<string>()); Assert.Equal(module, info["python_module"]!.GetValue<string>());
        var required = type switch
        {
            "EmptyImage" => """{"width":["INT",{"default":512,"min":1,"max":16384,"step":1}],"height":["INT",{"default":512,"min":1,"max":16384,"step":1}],"batch_size":["INT",{"default":1,"min":1,"max":4096}],"color":["INT",{"default":0,"min":0,"max":16777215,"step":1,"display":"color"}]}""",
            "ImageInvert" => """{"image":["IMAGE",{}]}""",
            "RepeatImageBatch" => """{"image":["IMAGE",{}],"amount":["INT",{"default":1,"min":1,"max":4096}]}""",
            "ImageFromBatch" => """{"image":["IMAGE",{}],"batch_index":["INT",{"default":0,"min":-16384,"max":16384}],"length":["INT",{"default":1,"min":1,"max":4096}]}""",
            _ => throw new InvalidOperationException()
        };
        var expected = JsonNode.Parse(required)!.AsObject();
        Assert.True(JsonNode.DeepEquals(expected, info["input"]!["required"]));
        Assert.Equal(expected.Select(p => p.Key), Strings(info["input_order"]!["required"]));
        Assert.Equal(new[] { "IMAGE" }, Strings(info["output"])); Assert.Equal(new[] { "IMAGE" }, Strings(info["output_name"]));
        Assert.False(info["output_is_list"]![0]!.GetValue<bool>());
        Assert.False(info["output_node"]!.GetValue<bool>()); Assert.False(info["is_input_list"]!.GetValue<bool>());
        if (type is "RepeatImageBatch" or "ImageFromBatch")
        {
            Assert.Null(info["input"]!["optional"]); Assert.False(info["api_node"]!.GetValue<bool>());
            Assert.False(info["has_intermediate_output"]!.GetValue<bool>());
            Assert.Equal(type == "RepeatImageBatch" ? new[] { "duplicate image", "clone image" } : new[] { "select image", "pick from batch", "extract image" }, Strings(info["search_aliases"]));
        }
        else Assert.Empty(info["input"]!["optional"]!.AsObject());
        if (type == "ImageInvert")
        {
            Assert.Equal("Image Tools", info["essentials_category"]!.GetValue<string>());
            Assert.Equal(new[] { "reverse colors" }, Strings(info["search_aliases"]));
        }
    }

    [Theory]
    [InlineData("width-zero", "empty", "width", "invalid_input_type")]
    [InlineData("height-too-large", "empty", "height", "invalid_input_type")]
    [InlineData("color-too-large", "empty", "color", "invalid_input_type")]
    [InlineData("width-object", "empty", "width", "invalid_input_type")]
    [InlineData("amount-zero", "repeat", "amount", "invalid_input_type")]
    [InlineData("index-too-negative", "extract", "batch_index", "invalid_input_type")]
    [InlineData("length-zero", "extract", "length", "invalid_input_type")]
    [InlineData("missing-image", "invert", "image", "required_input_missing")]
    [InlineData("string-image-link", "invert", "image", "type_mismatch")]
    public async Task InvalidImagePromptsAreRejectedBeforeQueue(string scenario, string nodeId, string inputName, string code)
    {
        var prompt = Prompt(); var inputs = prompt[nodeId]!["inputs"]!.AsObject();
        inputs[inputName] = scenario switch
        {
            "width-zero" or "amount-zero" or "length-zero" => JsonValue.Create(0),
            "height-too-large" => JsonValue.Create(16385),
            "color-too-large" => JsonValue.Create(0x1000000),
            "index-too-negative" => JsonValue.Create(-16385),
            "width-object" => new JsonObject(),
            "string-image-link" => new JsonArray("text", 0),
            "missing-image" => null,
            _ => throw new InvalidOperationException()
        };
        if (scenario == "missing-image") inputs.Remove(inputName);
        if (scenario == "string-image-link") prompt["text"] = new JsonObject { ["class_type"] = "PrimitiveString", ["inputs"] = new JsonObject { ["value"] = "not an IMAGE" } };
        string before = prompt.ToJsonString();
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        using var response = await client.PostAsJsonAsync("/prompt", Body(prompt, "preview"));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var error = (await response.Content.ReadFromJsonAsync<JsonObject>())!;
        var diagnostic = Assert.Single(error["node_errors"]![nodeId]!["errors"]!.AsArray())!;
        Assert.Equal(code, diagnostic["type"]!.GetValue<string>());
        Assert.Equal(inputName, diagnostic["extra_info"]!["input_name"]!.GetValue<string>());
        var queue = await client.GetFromJsonAsync<JsonObject>("/queue");
        Assert.Empty(queue!["queue_pending"]!.AsArray()); Assert.Empty(queue["queue_running"]!.AsArray());
        Assert.Empty((await client.GetFromJsonAsync<JsonObject>("/history"))!);
        Assert.Equal(before, prompt.ToJsonString());
    }

    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public async Task CompiledImageChainCompletesWithTextPreviewAndNoImageFilePayload(double version)
    {
        var document = WorkflowDocument.Parse(new JsonObject
        {
            ["version"] = version,
            ["nodes"] = new JsonArray(Node(1, "EmptyImage", new(1, 1, 2, 0xff0000), []),
                Node(2, "RepeatImageBatch", new(3), ["image"]), Node(3, "ImageFromBatch", new(-1, 10), ["image"]),
                Node(4, "ImageInvert", new(), ["image"]), Node(5, "PreviewAny", new(), ["source"])),
            ["links"] = new JsonArray()
        }.ToJsonString());
        for (int id = 1; id < 5; id++) document.Connect(new(id.ToString()), 0, new((id + 1).ToString()), 0);
        string before = document.ToJson(); var compiled = PromptCompiler.Compile(document);
        Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
        Assert.Equal(new[] { "EmptyImage", "RepeatImageBatch", "ImageFromBatch", "ImageInvert", "PreviewAny" },
            compiled.Prompt!.Select(p => p.Value!["class_type"]!.GetValue<string>()));
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        using var response = await client.PostAsJsonAsync("/prompt", Body(compiled.Prompt!, "5"), timeout.Token);
        response.EnsureSuccessStatusCode();
        string jobId = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
        while (true)
        {
            var job = await client.GetFromJsonAsync<JsonObject>($"/api/jobs/{jobId}", timeout.Token);
            string status = job!["status"]!.GetValue<string>();
            if (status == "completed") break;
            Assert.DoesNotContain(status, new[] { "failed", "cancelled" }); await Task.Delay(10, timeout.Token);
        }
        var history = (await client.GetFromJsonAsync<JsonObject>($"/history/{jobId}", timeout.Token))![jobId]!;
        var preview = Assert.Single(history["outputs"]!.AsObject()); Assert.Equal("5", preview.Key);
        Assert.Equal("text", Assert.Single(preview.Value!.AsObject()).Key);
        Assert.Equal(new[] { "tensor([[[[0., 1., 1.]]]])" }, Strings(preview.Value!["text"]));
        Assert.Empty(history["status"]!["messages"]!.AsArray()); Assert.Equal(before, document.ToJson());
    }

    private static JsonObject Prompt() => JsonNode.Parse("""
        {"empty":{"class_type":"EmptyImage","inputs":{"width":1,"height":1,"batch_size":2,"color":16711680}},
         "repeat":{"class_type":"RepeatImageBatch","inputs":{"image":["empty",0],"amount":3}},
         "extract":{"class_type":"ImageFromBatch","inputs":{"image":["repeat",0],"batch_index":-1,"length":10}},
         "invert":{"class_type":"ImageInvert","inputs":{"image":["extract",0]}},
         "preview":{"class_type":"PreviewAny","inputs":{"source":["invert",0]}}}
        """)!.AsObject();
    private static JsonObject Node(int id, string type, JsonArray widgets, string[] inputs) => new()
    {
        ["id"] = id, ["type"] = type, ["widgets_values"] = widgets,
        ["inputs"] = new JsonArray(inputs.Select(name => (JsonNode)new JsonObject { ["name"] = name, ["type"] = name == "source" ? "*" : "IMAGE", ["link"] = null }).ToArray()),
        ["outputs"] = type == "PreviewAny" ? new JsonArray() : new JsonArray(new JsonObject { ["name"] = "IMAGE", ["type"] = "IMAGE", ["links"] = new JsonArray() })
    };
    private static JsonObject Body(JsonObject prompt, string target) => new() { ["prompt"] = prompt.DeepClone(), ["partial_execution_targets"] = new JsonArray(target) };
    private static string[] Strings(JsonNode? array) => array!.AsArray().Select(v => v!.GetValue<string>()).ToArray();
}
