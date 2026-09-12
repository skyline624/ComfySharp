using System.Net.Http.Json;
using System.Text.Json.Nodes;
using ComfySharp.Host;
using ComfySharp.Workflow;
using Microsoft.AspNetCore.Mvc.Testing;

namespace ComfySharp.Host.Tests;

public sealed class PromptLiteralTests
{
    [Theory]
    [InlineData("{\"__value__\":1,\"sibling\":2}", "1")]
    [InlineData("{\"__value__\":[],\"sibling\":[1]}", "[]")]
    [InlineData("{\"__value__\":{\"__value__\":[\"absent\",0]},\"sibling\":true}", "{\"__value__\":[\"absent\",0]}")]
    [InlineData("{\"__value__\":{\"__value__\":1,\"sibling\":2}}", "{\"__value__\":1,\"sibling\":2}")]
    [InlineData("{\"__VALUE__\":1,\"nested\":{\"__value__\":2}}", "{\"__VALUE__\":1,\"nested\":{\"__value__\":2}}")]
    [InlineData("[]", "[]")]
    [InlineData("[\"absent\",0]", "[\"absent\",0]")]
    public async Task CurrentCompilerAndHttpUnwrapOnceIncludingExplicitEnvelopesAndObjectCollisions(string literal, string expectedValue)
    {
        var document = Document(literal);
        string before = document.ToJson();
        var compilation = PromptCompiler.Compile(document, Definitions());
        Assert.True(compilation.Success, string.Join("; ", compilation.Diagnostics));
        string promptBefore = compilation.Prompt!.ToJsonString();
        string preview = await ExecutePreview(compilation.Prompt);
        // Existing contract: the compiler wraps arrays only. An object with __value__ is
        // consumed as an envelope, losing outer siblings; an explicit extra envelope protects
        // an intended dictionary. This records C# behavior, not upstream HTTP parity.
        // Compare JSON contents, not pretty-print whitespace or a new Python repr contract.
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(expectedValue), JsonNode.Parse(preview)), preview);
        Assert.Equal(before, document.ToJson());
        Assert.Equal(promptBefore, compilation.Prompt.ToJsonString());
    }

    [Fact]
    public async Task ConnectedWidgetExecutesTheRealProducerInsteadOfThePersistedReservedObject()
    {
        var root = Document("{\"__value__\":1,\"sibling\":2}").Snapshot();
        root["nodes"]!.AsArray().Add(JsonNode.Parse("""
            {"id":2,"type":"PrimitiveString","widgets_values":["connected"],
             "outputs":[{"name":"STRING","type":"STRING","links":[]}]}
            """));
        var document = WorkflowDocument.Parse(root.ToJsonString());
        document.Connect(new("2"), 0, new("1"), 0);
        var compilation = PromptCompiler.Compile(document, Definitions());
        Assert.True(compilation.Success, string.Join("; ", compilation.Diagnostics));
        Assert.True(JsonNode.DeepEquals(new JsonArray("2", 0), compilation.Prompt!["1"]!["inputs"]!["source"]));
        Assert.Equal("connected", await ExecutePreview(compilation.Prompt));
    }

    private static IReadOnlyDictionary<string, NodeDefinition> Definitions() =>
        new Dictionary<string, NodeDefinition>(PromptCompiler.BaseDefinitions)
        {
            // Explicit fixture frontend layout only. No node is registered or substituted:
            // the HTTP host supplies its actual PreviewAny and PrimitiveString implementations.
            ["PreviewAny"] = new("PreviewAny", [new("source")])
        };

    private static WorkflowDocument Document(string literal)
    {
        var root = JsonNode.Parse("""
            {"version":1,"nodes":[{"id":1,"type":"PreviewAny",
              "inputs":[{"name":"source","type":"*","link":null}],"outputs":[]}],"links":[]}
            """)!.AsObject();
        root["nodes"]![0]!["widgets_values"] = new JsonArray(JsonNode.Parse(literal));
        return WorkflowDocument.Parse(root.ToJsonString());
    }

    private static async Task<string> ExecutePreview(JsonObject prompt)
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var response = await client.PostAsJsonAsync("/prompt", new JsonObject
        {
            ["prompt"] = prompt.DeepClone(), ["partial_execution_targets"] = new JsonArray("1")
        }, timeout.Token);
        response.EnsureSuccessStatusCode();
        string id = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
        while (true)
        {
            var job = await client.GetFromJsonAsync<JsonObject>($"/api/jobs/{id}", timeout.Token);
            string state = job!["status"]!.GetValue<string>();
            if (state == "completed") break;
            Assert.DoesNotContain(state, new[] { "failed", "cancelled" });
            await Task.Delay(10, timeout.Token);
        }
        var history = await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token);
        var outputs = history![id]!["outputs"]!.AsObject();
        Assert.Equal("1", Assert.Single(outputs).Key);
        return Assert.Single(outputs["1"]!["text"]!.AsArray())!.GetValue<string>();
    }
}
