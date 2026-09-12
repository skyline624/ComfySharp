using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Nodes;
using ComfySharp.Host;
using ComfySharp.Workflow;
using Microsoft.AspNetCore.Mvc.Testing;

namespace ComfySharp.Host.Tests;

/// <summary>Real Host/compiler integration contracts; no source-output oracle is synthesized here.</summary>
public sealed class StringFormatHostTests
{
    [Fact]
    public async Task ObjectInfoPreservesNamesSchemaAndExplicitlyDeclaresThePartialProfile()
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        var all = await client.GetFromJsonAsync<JsonObject>("/object_info");
        Assert.Equal(29, all!.Count);
        var info = (await client.GetFromJsonAsync<JsonObject>("/object_info/StringFormat"))!["StringFormat"]!;
        Assert.True(JsonNode.DeepEquals(info, all["StringFormat"]));
        Assert.Equal("Format Text", info["display_name"]!.GetValue<string>());
        Assert.Equal("text", info["category"]!.GetValue<string>());
        Assert.Equal("comfy_extras.nodes_string", info["python_module"]!.GetValue<string>());
        Assert.Equal(new[] { "string", "format" }, Strings(info["search_aliases"]));
        Assert.Contains("python-format-text-v1", info["description"]!.GetValue<string>());
        Assert.Contains("partial", info["description"]!.GetValue<string>(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal(new[] { "values", "f_string" }, Strings(info["input_order"]!["required"]));
        Assert.Null(info["input"]!["optional"]);
        var required = info["input"]!["required"]!;
        Assert.Equal("COMFY_AUTOGROW_V3", required["values"]![0]!.GetValue<string>());
        var template = required["values"]![1]!["template"]!;
        Assert.Equal(0, template["min"]!.GetValue<int>());
        Assert.Equal("abcdefghijklmnopqrstuvwxyz".Select(c => c.ToString()), Strings(template["names"]));
        Assert.Null(template["max"]);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse("""{"required":{"value":["*",{}]}}"""), template["input"]));
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse("""["STRING",{"default":"{a}","multiline":true}]"""), required["f_string"]));
        Assert.Equal(new[] { "STRING" }, Strings(info["output"]));
        Assert.Equal(new[] { "STRING" }, Strings(info["output_name"]));
        Assert.False(info["output_is_list"]![0]!.GetValue<bool>());
        Assert.False(info["is_input_list"]!.GetValue<bool>());
        Assert.False(info["output_node"]!.GetValue<bool>());
    }

    [Fact]
    public async Task PersistedDefaultFormatsARealStringProducerThroughPreview()
    {
        var document = Document(Node(1, "PrimitiveString", ["value"], "alpha"),
            Node(20, "StringFormat", ["values.a"], "{a}"), Preview());
        document.Connect(new("1"), 0, new("20"), 0);
        document.Connect(new("20"), 0, new("30"), 0);
        var history = await Execute(document, "completed");
        Assert.Equal(new[] { "alpha" }, PreviewText(history));
    }

    [Fact]
    public async Task ConstantFormatRequiresNoAutogrowValues()
    {
        var document = Document(Node(20, "StringFormat", [], "constant"), Preview());
        document.Connect(new("20"), 0, new("30"), 0);
        Assert.Equal(new[] { "constant" }, PreviewText(await Execute(document, "completed")));
    }

    [Fact]
    public async Task ConnectedFormatOverridesPersistedWidgetAndSparseNamedPortsRemainFlat()
    {
        var document = Document(Node(1, "PrimitiveString", ["value"], "alpha"),
            Node(2, "PrimitiveString", ["value"], "{z}/{z}"),
            Node(20, "StringFormat", ["f_string", "values.z"], "unused {missing}"), Preview());
        document.Connect(new("1"), 0, new("20"), 1);
        document.Connect(new("2"), 0, new("20"), 0);
        document.Connect(new("20"), 0, new("30"), 0);
        var prompt = Compile(document);
        Assert.True(JsonNode.DeepEquals(new JsonArray("2", 0), prompt["20"]!["inputs"]!["f_string"]));
        Assert.True(JsonNode.DeepEquals(new JsonArray("1", 0), prompt["20"]!["inputs"]!["values.z"]));
        Assert.Null(prompt["20"]!["inputs"]!["values"]);
        Assert.Equal(new[] { "alpha/alpha" }, PreviewText(await Execute(document, "completed")));
    }

    [Fact]
    public async Task TwoRealCreateListsMapFormatsAndRepeatTheLastValue()
    {
        var document = Document(Node(1, "PrimitiveString", ["value"], "alpha"),
            Node(2, "PrimitiveString", ["value"], "beta"),
            Node(3, "PrimitiveString", ["value"], "one:{a}"),
            Node(4, "PrimitiveString", ["value"], "two:{a}"),
            Node(5, "PrimitiveString", ["value"], "three:{a}"),
            Node(10, "CreateList", ["inputs.input0", "inputs.input1"]),
            Node(11, "CreateList", ["inputs.input0", "inputs.input1", "inputs.input2"]),
            Node(20, "StringFormat", ["values.a", "f_string"], "unused"), Preview());
        document.Connect(new("1"), 0, new("10"), 0); document.Connect(new("2"), 0, new("10"), 1);
        document.Connect(new("3"), 0, new("11"), 0); document.Connect(new("4"), 0, new("11"), 1);
        document.Connect(new("5"), 0, new("11"), 2);
        document.Connect(new("10"), 0, new("20"), 0); document.Connect(new("11"), 0, new("20"), 1);
        document.Connect(new("20"), 0, new("30"), 0);
        Assert.Equal(new[] { "one:alpha", "two:beta", "three:beta" }, PreviewText(await Execute(document, "completed")));
    }

    [Theory]
    [InlineData("{a!r}", "unsupported_format_feature")]
    [InlineData("{a[0]}", "unsupported_format_feature")]
    [InlineData("{missing}", "format_missing_field")]
    [InlineData("{", "format_syntax")]
    public async Task UnsupportedOrMalformedFormatsFailExecutionWithoutPreview(string format, string category)
    {
        var document = Document(Node(1, "PrimitiveString", ["value"], "alpha"),
            Node(20, "StringFormat", ["values.a"], format), Preview());
        document.Connect(new("1"), 0, new("20"), 0); document.Connect(new("20"), 0, new("30"), 0);
        var history = await Execute(document, "failed");
        Assert.Empty(history["outputs"]!.AsObject());
        var messages = history["status"]!["messages"]!.AsArray();
        Assert.Contains(messages, m => m!["Code"]?.GetValue<string>() == "execution_error" &&
            m["Message"]!.GetValue<string>().Contains(category, StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("values.A", "unsupported_dynamic_input")]
    [InlineData("values.aa", "unsupported_dynamic_input")]
    [InlineData(null, "required_input_missing")]
    public async Task InvalidFlatNamesOrMissingRequiredFormatAreRejectedBeforeQueue(string? extraName, string code)
    {
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        var inputs = new JsonObject();
        if (extraName is not null) { inputs["f_string"] = "constant"; inputs[extraName] = "x"; }
        var prompt = new JsonObject
        {
            ["20"] = new JsonObject { ["class_type"] = "StringFormat", ["inputs"] = inputs },
            ["30"] = new JsonObject { ["class_type"] = "PreviewAny", ["inputs"] = new JsonObject { ["source"] = new JsonArray("20", 0) } }
        };
        using var response = await client.PostAsJsonAsync("/prompt", Body(prompt));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains(code, (await response.Content.ReadFromJsonAsync<JsonObject>())!.ToJsonString());
        var queue = await client.GetFromJsonAsync<JsonObject>("/queue");
        Assert.Empty(queue!["queue_pending"]!.AsArray()); Assert.Empty(queue["queue_running"]!.AsArray());
    }

    private static string[] Strings(JsonNode? array) => array!.AsArray().Select(x => x!.GetValue<string>()).ToArray();
    private static string[] PreviewText(JsonObject history)
    {
        Assert.Equal("30", Assert.Single(history["outputs"]!.AsObject()).Key);
        return Strings(history["outputs"]!["30"]!["text"]);
    }
    private static JsonObject Node(int id, string type, string[] ports, params string[] widgets) => new()
    {
        ["id"] = id, ["type"] = type,
        ["inputs"] = new JsonArray(ports.Select(p => (JsonNode?)new JsonObject { ["name"] = p, ["type"] = "*", ["link"] = null }).ToArray()),
        ["outputs"] = new JsonArray(new JsonObject { ["name"] = "STRING", ["type"] = "*", ["links"] = new JsonArray() }),
        ["widgets_values"] = new JsonArray(widgets.Select(s => (JsonNode?)JsonValue.Create(s)).ToArray())
    };
    private static JsonObject Preview()
    {
        var node = Node(30, "PreviewAny", ["source"]); node["outputs"] = new JsonArray(); return node;
    }
    private static WorkflowDocument Document(params JsonObject[] nodes) => WorkflowDocument.Parse(new JsonObject
        { ["version"] = 1, ["nodes"] = new JsonArray(nodes.Cast<JsonNode?>().ToArray()), ["links"] = new JsonArray() }.ToJsonString());
    private static JsonObject Compile(WorkflowDocument document)
    {
        var result = PromptCompiler.Compile(document);
        Assert.True(result.Success, string.Join("; ", result.Diagnostics)); return result.Prompt!;
    }
    private static JsonObject Body(JsonObject prompt) => new() { ["prompt"] = prompt.DeepClone(), ["partial_execution_targets"] = new JsonArray("30") };
    private static async Task<JsonObject> Execute(WorkflowDocument document, string expectedStatus)
    {
        string before = document.ToJson(); var prompt = Compile(document); string promptBefore = prompt.ToJsonString();
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var response = await client.PostAsJsonAsync("/prompt", Body(prompt), timeout.Token); response.EnsureSuccessStatusCode();
        string id = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
        while (true)
        {
            var job = await client.GetFromJsonAsync<JsonObject>($"/api/jobs/{id}", timeout.Token);
            string status = job!["status"]!.GetValue<string>();
            if (status is "completed" or "failed" or "cancelled") { Assert.Equal(expectedStatus, status); break; }
            await Task.Delay(10, timeout.Token);
        }
        var history = await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token);
        Assert.Equal(before, document.ToJson()); Assert.Equal(promptBefore, prompt.ToJsonString());
        return history![id]!.AsObject();
    }
}
