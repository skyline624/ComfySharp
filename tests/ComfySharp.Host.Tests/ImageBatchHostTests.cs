using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Nodes;
using ComfySharp.Host;
using ComfySharp.Workflow;
using Microsoft.AspNetCore.Mvc.Testing;

namespace ComfySharp.Host.Tests;

/// <summary>Real schema, admission and compiled HTTP routes; not an independent interpolation oracle.</summary>
public sealed class ImageBatchHostTests
{
    [Fact]
    public async Task ObjectInfoPreservesDeprecatedImageBatchAndTwoRequiredImagePorts()
    {
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        var info = (await client.GetFromJsonAsync<JsonObject>("/object_info/ImageBatch"))!["ImageBatch"]!;
        Assert.Equal("ImageBatch", info["name"]!.GetValue<string>());
        Assert.Equal("Batch Images (DEPRECATED)", info["display_name"]!.GetValue<string>());
        Assert.True(info["deprecated"]!.GetValue<bool>());
        Assert.Equal("image/batch", info["category"]!.GetValue<string>());
        Assert.Equal("nodes", info["python_module"]!.GetValue<string>());
        Assert.Equal(new[] { "combine images", "merge images", "stack images" }, Strings(info["search_aliases"]));
        Assert.Equal(new[] { "image1", "image2" }, Strings(info["input_order"]!["required"]));
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse("""{"image1":["IMAGE",{}],"image2":["IMAGE",{}]}"""), info["input"]!["required"]));
        Assert.Empty(info["input"]!["optional"]!.AsObject());
        Assert.Equal(new[] { "IMAGE" }, Strings(info["output"])); Assert.Equal(new[] { "IMAGE" }, Strings(info["output_name"]));
        Assert.False(info["output_is_list"]![0]!.GetValue<bool>());
        Assert.False(info["is_input_list"]!.GetValue<bool>()); Assert.False(info["output_node"]!.GetValue<bool>());
    }

    [Theory]
    [InlineData("image1", false, "required_input_missing")]
    [InlineData("image2", false, "required_input_missing")]
    [InlineData("image1", true, "type_mismatch")]
    [InlineData("image2", true, "type_mismatch")]
    public async Task MissingOrStringLinkedImageIsRejectedBeforeQueue(string inputName, bool wrongLink, string code)
    {
        var compiled = PromptCompiler.Compile(Document(1.0));
        Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
        var prompt = compiled.Prompt!;
        var inputs = prompt["3"]!["inputs"]!.AsObject();
        if (wrongLink)
        {
            prompt["text"] = new JsonObject { ["class_type"] = "PrimitiveString", ["inputs"] = new JsonObject { ["value"] = "not an image" } };
            inputs[inputName] = new JsonArray("text", 0);
        }
        else inputs.Remove(inputName);
        string before = prompt.ToJsonString();
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        using var response = await client.PostAsJsonAsync("/prompt", Body(prompt));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var error = (await response.Content.ReadFromJsonAsync<JsonObject>())!;
        var diagnostic = Assert.Single(error["node_errors"]!["3"]!["errors"]!.AsArray())!;
        Assert.Equal(code, diagnostic["type"]!.GetValue<string>());
        Assert.Equal(inputName, diagnostic["extra_info"]!["input_name"]!.GetValue<string>());
        var queue = await client.GetFromJsonAsync<JsonObject>("/queue");
        Assert.Empty(queue!["queue_pending"]!.AsArray()); Assert.Empty(queue["queue_running"]!.AsArray());
        Assert.Empty((await client.GetFromJsonAsync<JsonObject>("/history"))!); Assert.Equal(before, prompt.ToJsonString());
    }

    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public async Task CompiledUnequalSizeBatchesReturnTheSecondBatchLastImageAsText(double version)
    {
        var document = Document(version); string before = document.ToJson();
        var compiled = PromptCompiler.Compile(document); Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
        Assert.True(JsonNode.DeepEquals(new JsonObject { ["image1"] = new JsonArray("1", 0), ["image2"] = new JsonArray("2", 0) }, compiled.Prompt!["3"]!["inputs"]));
        Assert.Equal(2, compiled.Prompt["1"]!["inputs"]!["batch_size"]!.GetValue<int>());
        Assert.Equal(2, compiled.Prompt["2"]!["inputs"]!["width"]!.GetValue<int>());
        Assert.Equal(-1, compiled.Prompt["4"]!["inputs"]!["batch_index"]!.GetValue<int>());
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        using var response = await client.PostAsJsonAsync("/prompt", Body(compiled.Prompt!), timeout.Token);
        response.EnsureSuccessStatusCode();
        string id = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
        while (true)
        {
            var job = await client.GetFromJsonAsync<JsonObject>($"/api/jobs/{id}", timeout.Token);
            string status = job!["status"]!.GetValue<string>(); if (status == "completed") break;
            Assert.DoesNotContain(status, new[] { "failed", "cancelled" }); await Task.Delay(10, timeout.Token);
        }
        var history = (await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token))![id]!;
        var preview = Assert.Single(history["outputs"]!.AsObject()); Assert.Equal("5", preview.Key);
        Assert.Equal("text", Assert.Single(preview.Value!.AsObject()).Key);
        Assert.Equal(new[] { "tensor([[[[0., 1., 0.]]]])" }, Strings(preview.Value!["text"]));
        Assert.Empty(history["status"]!["messages"]!.AsArray()); Assert.Equal(before, document.ToJson());
    }

    private static WorkflowDocument Document(double version)
    {
        var document = WorkflowDocument.Parse(new JsonObject
        {
            ["version"] = version,
            ["nodes"] = new JsonArray(Node(1, "EmptyImage", new(1, 1, 2, 0xff0000), []),
                Node(2, "EmptyImage", new(2, 2, 1, 0x00ff00), []), Node(3, "ImageBatch", new(), ["image1", "image2"]),
                Node(4, "ImageFromBatch", new(-1, 1), ["image"]), Node(5, "PreviewAny", new(), ["source"])),
            ["links"] = new JsonArray()
        }.ToJsonString());
        document.Connect(new("1"), 0, new("3"), 0); document.Connect(new("2"), 0, new("3"), 1);
        document.Connect(new("3"), 0, new("4"), 0); document.Connect(new("4"), 0, new("5"), 0);
        return document;
    }
    private static JsonObject Node(int id, string type, JsonArray widgets, string[] names) => new()
    {
        ["id"] = id, ["type"] = type, ["widgets_values"] = widgets,
        ["inputs"] = new JsonArray(names.Select(name => (JsonNode)new JsonObject { ["name"] = name, ["type"] = name == "source" ? "*" : "IMAGE", ["link"] = null }).ToArray()),
        ["outputs"] = type == "PreviewAny" ? new JsonArray() : new JsonArray(new JsonObject { ["name"] = "IMAGE", ["type"] = "IMAGE", ["links"] = new JsonArray() })
    };
    private static JsonObject Body(JsonObject prompt) => new() { ["prompt"] = prompt.DeepClone(), ["partial_execution_targets"] = new JsonArray("5") };
    private static string[] Strings(JsonNode? array) => array!.AsArray().Select(v => v!.GetValue<string>()).ToArray();
}
