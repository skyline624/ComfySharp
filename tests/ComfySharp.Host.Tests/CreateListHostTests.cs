using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Nodes;
using ComfySharp.Host;
using Microsoft.AspNetCore.Mvc.Testing;

namespace ComfySharp.Host.Tests;

public sealed class CreateListHostTests
{
    [Fact]
    public async Task RealHostProjectsTheTemplateAndExecutesAFlatCreateListPrompt()
    {
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        var info = await client.GetFromJsonAsync<JsonObject>("/object_info/CreateList");
        Assert.Equal("COMFY_AUTOGROW_V3", info!["CreateList"]!["input"]!["required"]!["inputs"]![0]!.GetValue<string>());
        Assert.True(info["CreateList"]!["is_input_list"]!.GetValue<bool>());
        using var response = await client.PostAsJsonAsync("/prompt", JsonNode.Parse("""
            {"prompt":{"a":{"class_type":"PrimitiveString","inputs":{"value":"alpha"}},
                       "b":{"class_type":"PrimitiveString","inputs":{"value":"🌍"}},
                       "list":{"class_type":"CreateList","inputs":{"inputs.input2":["b",0],"inputs.input0":["a",0]}},
                       "length":{"class_type":"StringLength","inputs":{"string":["list",0]}},
                       "preview":{"class_type":"PreviewAny","inputs":{"source":["length",0]}}},
             "partial_execution_targets":["preview"]}
            """));
        response.EnsureSuccessStatusCode();
        string id = (await response.Content.ReadFromJsonAsync<JsonObject>())!["prompt_id"]!.GetValue<string>();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        while (true)
        {
            var job = await client.GetFromJsonAsync<JsonObject>($"/api/jobs/{id}", timeout.Token);
            string state = job!["status"]!.GetValue<string>();
            if (state == "completed") break;
            Assert.DoesNotContain(state, new[] { "failed", "cancelled" });
            await Task.Delay(10, timeout.Token);
        }
        var history = await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token);
        Assert.Equal(new[] { "5", "1" }, history![id]!["outputs"]!["preview"]!["text"]!.AsArray().Select(v => v!.GetValue<string>()));
        Assert.Equal("preview", Assert.Single(history[id]!["outputs"]!.AsObject()).Key);
    }

    [Theory]
    [InlineData("inputs.input10")]
    [InlineData("inputs.input01")]
    [InlineData("inputs")]
    public async Task ExtraDynamicNamesProduceStructured400BeforeEnqueue(string name)
    {
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        var body = JsonNode.Parse("""
            {"prompt":{"list":{"class_type":"CreateList","inputs":{"inputs.input0":1}},
                       "preview":{"class_type":"PreviewAny","inputs":{"source":["list",0]}}},
             "partial_execution_targets":["preview"]}
            """ )!.AsObject();
        body["prompt"]!["list"]!["inputs"]![name] = 2;
        using var response = await client.PostAsJsonAsync("/prompt", body);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var detail = await response.Content.ReadFromJsonAsync<JsonObject>();
        Assert.Contains("unsupported_dynamic_input", detail!.ToJsonString());
        var queue = await client.GetFromJsonAsync<JsonObject>("/queue");
        Assert.Empty(queue!["queue_pending"]!.AsArray());
    }
}
