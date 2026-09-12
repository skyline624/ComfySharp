using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Nodes;
using ComfySharp.Host;
using ComfySharp.Workflow;
using Microsoft.AspNetCore.Mvc.Testing;

namespace ComfySharp.Host.Tests;

/// <summary>Actual compiler/HTTP integration properties, separate from the prospective source oracle.</summary>
public sealed class CaseConverterHostTests
{
    [Fact]
    public async Task ObjectInfoPreservesTheV3ComboAndRequiredSourceSchema()
    {
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        var info = (await client.GetFromJsonAsync<JsonObject>("/object_info/CaseConverter"))!["CaseConverter"]!;
        Assert.Equal("CaseConverter", info["name"]!.GetValue<string>());
        Assert.Equal("Convert Text Case", info["display_name"]!.GetValue<string>());
        Assert.Equal("text", info["category"]!.GetValue<string>());
        Assert.Equal("comfy_extras.nodes_string", info["python_module"]!.GetValue<string>());
        Assert.Equal(new[] { "case converter", "text case", "uppercase", "lowercase", "capitalize" }, Strings(info["search_aliases"]));
        Assert.Equal(new[] { "string", "mode" }, Strings(info["input_order"]!["required"]));
        Assert.Equal(new[] { "string", "mode" }, info["input"]!["required"]!.AsObject().Select(p => p.Key));
        Assert.Null(info["input"]!["optional"]);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse("""["STRING",{"multiline":true}]"""), info["input"]!["required"]!["string"]));
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse("""["COMBO",{"multiselect":false,"options":["UPPERCASE","lowercase","Capitalize","Title Case"]}]"""), info["input"]!["required"]!["mode"]));
        Assert.Null(info["input"]!["required"]!["mode"]![1]!["default"]);
        Assert.Equal(new[] { "STRING" }, Strings(info["output"])); Assert.Equal(new[] { "STRING" }, Strings(info["output_name"]));
        Assert.False(info["output_is_list"]![0]!.GetValue<bool>());
        Assert.False(info["output_node"]!.GetValue<bool>()); Assert.False(info["is_input_list"]!.GetValue<bool>());
        Assert.False(info["api_node"]!.GetValue<bool>()); Assert.False(info["has_intermediate_output"]!.GetValue<bool>());
    }

    [Theory]
    [InlineData("UPPERCASE", "Straße", "STRASSE")]
    [InlineData("lowercase", "İ", "i\u0307")]
    [InlineData("Capitalize", "ǳABC", "ǲabc")]
    [InlineData("Title Case", "they're 2fast", "They'Re 2Fast")]
    [InlineData("Title Case", "AΣA", "Aσa")]
    [InlineData("Capitalize", "1ABC", "1abc")]
    [InlineData("lowercase", "É", "é")]
    [InlineData("UPPERCASE", "e\u0301", "E\u0301")]
    [InlineData("UPPERCASE", "", "")]
    [InlineData("lowercase", "", "")]
    [InlineData("Capitalize", "", "")]
    [InlineData("Title Case", "", "")]
    public async Task CompiledModeRunsTheActualNodeAndPreview(string mode, string input, string expected)
    {
        Assert.Equal(new[] { expected }, await Execute(Document(input, mode)));
    }

    [Fact]
    public async Task RealStringConnectionOverridesTheSavedLiteralWidget()
    {
        var document = Document("saved widget", "UPPERCASE", connected: true);
        var compiled = PromptCompiler.Compile(document); Assert.True(compiled.Success);
        Assert.True(JsonNode.DeepEquals(new JsonArray("2", 0), compiled.Prompt!["10"]!["inputs"]!["string"]));
        Assert.Equal(new[] { "STRASSE" }, await Execute(document));
    }

    [Fact]
    public async Task WildcardCreateListMapsTheComboAndRepeatsTheShorterTextColumn()
    {
        // STRING -> COMBO is incompatible; the real CreateList wildcard output is admitted.
        var prompt = JsonNode.Parse("""
            {"modes":{"class_type":"CreateList","inputs":{"inputs.input0":"UPPERCASE","inputs.input1":"lowercase","inputs.input2":"Capitalize","inputs.input3":"Title Case"}},
             "texts":{"class_type":"CreateList","inputs":{"inputs.input0":"bC","inputs.input1":"ǳABC"}},
             "10":{"class_type":"CaseConverter","inputs":{"string":["texts",0],"mode":["modes",0]}},
             "20":{"class_type":"PreviewAny","inputs":{"source":["10",0]}}}
            """)!.AsObject();
        Assert.Equal(new[] { "BC", "ǳabc", "ǲabc", "ǲabc" }, await Execute(prompt));
    }

    [Theory]
    [InlineData("missing-string", "required_input_missing")]
    [InlineData("missing-mode", "required_input_missing")]
    [InlineData("invalid-mode", "invalid_input_type")]
    [InlineData("string-to-combo", "type_mismatch")]
    public async Task InvalidPromptsAreRefusedBeforeQueue(string scenario, string code)
    {
        var inputs = new JsonObject { ["string"] = "text", ["mode"] = "UPPERCASE" };
        var prompt = new JsonObject
        {
            ["10"] = new JsonObject { ["class_type"] = "CaseConverter", ["inputs"] = inputs },
            ["20"] = new JsonObject { ["class_type"] = "PreviewAny", ["inputs"] = new JsonObject { ["source"] = new JsonArray("10", 0) } }
        };
        if (scenario == "missing-string") inputs.Remove("string");
        if (scenario == "missing-mode") inputs.Remove("mode");
        if (scenario == "invalid-mode") inputs["mode"] = "uppercase";
        if (scenario == "string-to-combo")
        {
            prompt["2"] = new JsonObject { ["class_type"] = "PrimitiveString", ["inputs"] = new JsonObject { ["value"] = "UPPERCASE" } };
            inputs["mode"] = new JsonArray("2", 0);
        }
        string before = prompt.ToJsonString();
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        using var response = await client.PostAsJsonAsync("/prompt", Body(prompt));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains(code, (await response.Content.ReadFromJsonAsync<JsonObject>())!.ToJsonString());
        var queue = await client.GetFromJsonAsync<JsonObject>("/queue");
        Assert.Empty(queue!["queue_pending"]!.AsArray()); Assert.Empty(queue["queue_running"]!.AsArray());
        Assert.Equal(before, prompt.ToJsonString());
    }

    private static WorkflowDocument Document(string text, string mode, bool connected = false)
    {
        var nodes = new JsonArray(new JsonObject
        {
            ["id"] = 10, ["type"] = "CaseConverter", ["widgets_values"] = new JsonArray(text, mode),
            ["inputs"] = new JsonArray(new JsonObject { ["name"] = "string", ["type"] = "STRING", ["link"] = null }),
            ["outputs"] = new JsonArray(new JsonObject { ["name"] = "STRING", ["type"] = "STRING", ["links"] = new JsonArray() })
        }, new JsonObject
        {
            ["id"] = 20, ["type"] = "PreviewAny", ["widgets_values"] = new JsonArray(), ["outputs"] = new JsonArray(),
            ["inputs"] = new JsonArray(new JsonObject { ["name"] = "source", ["type"] = "*", ["link"] = null })
        });
        if (connected) nodes.Add(new JsonObject
        {
            ["id"] = 2, ["type"] = "PrimitiveString", ["widgets_values"] = new JsonArray("Straße"),
            ["outputs"] = new JsonArray(new JsonObject { ["name"] = "STRING", ["type"] = "STRING", ["links"] = new JsonArray() })
        });
        var document = WorkflowDocument.Parse(new JsonObject { ["version"] = 1, ["nodes"] = nodes, ["links"] = new JsonArray() }.ToJsonString());
        document.Connect(new("10"), 0, new("20"), 0);
        if (connected) document.Connect(new("2"), 0, new("10"), 0);
        return document;
    }
    private static async Task<string[]> Execute(WorkflowDocument document)
    {
        string before = document.ToJson(); var compiled = PromptCompiler.Compile(document);
        Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
        var output = await Execute(compiled.Prompt!); Assert.Equal(before, document.ToJson()); return output;
    }
    private static async Task<string[]> Execute(JsonObject prompt)
    {
        string before = prompt.ToJsonString();
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var response = await client.PostAsJsonAsync("/prompt", Body(prompt), timeout.Token); response.EnsureSuccessStatusCode();
        string id = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
        while (true)
        {
            var job = await client.GetFromJsonAsync<JsonObject>($"/api/jobs/{id}", timeout.Token);
            string status = job!["status"]!.GetValue<string>();
            if (status == "completed") break;
            Assert.DoesNotContain(status, new[] { "failed", "cancelled" }); await Task.Delay(10, timeout.Token);
        }
        var history = (await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token))![id]!;
        Assert.Equal("20", Assert.Single(history["outputs"]!.AsObject()).Key);
        Assert.Empty(history["status"]!["messages"]!.AsArray()); Assert.Equal(before, prompt.ToJsonString());
        return Strings(history["outputs"]!["20"]!["text"]);
    }
    private static JsonObject Body(JsonObject prompt) => new() { ["prompt"] = prompt.DeepClone(), ["partial_execution_targets"] = new JsonArray("20") };
    private static string[] Strings(JsonNode? array) => array!.AsArray().Select(v => v!.GetValue<string>()).ToArray();
}
