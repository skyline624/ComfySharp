using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Nodes;
using ComfySharp.Host;
using ComfySharp.Workflow;
using Microsoft.AspNetCore.Mvc.Testing;

namespace ComfySharp.Host.Tests;

/// <summary>Real compiler/HTTP integration properties, distinct from the prospective source corpus.</summary>
public sealed class TextComparisonHostTests
{
    [Theory]
    [InlineData("StringContains")]
    [InlineData("StringCompare")]
    public async Task ObjectInfoRetainsV3NamesOrderChoicesAndRequiredAdvancedDefault(string type)
    {
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        var info = (await client.GetFromJsonAsync<JsonObject>("/object_info/" + type))![type]!;
        bool contains = type == "StringContains";
        Assert.Equal(type, info["name"]!.GetValue<string>());
        Assert.Equal(contains ? "Contains Text" : "Compare Text", info["display_name"]!.GetValue<string>());
        Assert.Equal("text", info["category"]!.GetValue<string>());
        Assert.Equal("comfy_extras.nodes_string", info["python_module"]!.GetValue<string>());
        Assert.Equal(contains ? new[] { "contains", "text includes", "string includes" }
            : new[] { "compare", "text match", "string equals", "starts with", "ends with" }, Strings(info["search_aliases"]));
        var names = contains ? new[] { "string", "substring", "case_sensitive" }
            : new[] { "string_a", "string_b", "mode", "case_sensitive" };
        Assert.Equal(names, Strings(info["input_order"]!["required"]));
        Assert.Equal(names, info["input"]!["required"]!.AsObject().Select(p => p.Key));
        Assert.Null(info["input"]!["optional"]);
        foreach (string name in names.Take(2))
            Assert.True(JsonNode.DeepEquals(JsonNode.Parse("""["STRING",{"multiline":true}]"""), info["input"]!["required"]![name]));
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse("""["BOOLEAN",{"default":true,"advanced":true}]"""), info["input"]!["required"]!["case_sensitive"]));
        if (!contains) Assert.True(JsonNode.DeepEquals(JsonNode.Parse("""["COMBO",{"multiselect":false,"options":["Starts With","Ends With","Equal"]}]"""), info["input"]!["required"]!["mode"]));
        Assert.Equal(new[] { "BOOLEAN" }, Strings(info["output"]));
        Assert.Equal(new[] { contains ? "contains" : "BOOLEAN" }, Strings(info["output_name"]));
        Assert.False(info["output_is_list"]![0]!.GetValue<bool>());
        Assert.False(info["output_node"]!.GetValue<bool>()); Assert.False(info["is_input_list"]!.GetValue<bool>());
        Assert.False(info["api_node"]!.GetValue<bool>()); Assert.False(info["has_intermediate_output"]!.GetValue<bool>());
    }

    [Theory]
    [InlineData("Hello", "ELL", true, false)]
    [InlineData("İ", "i\u0307", false, true)]
    [InlineData("ΟΣ", "ος", false, true)]
    [InlineData("é", "e\u0301", false, false)]
    [InlineData("Straße", "STRASSE", false, false)]
    [InlineData("", "", true, true)]
    public async Task CompiledContainsRunsTheRealHostWithoutNormalizationOrCaseFolding(string a, string b, bool sensitive, bool expected)
    {
        var document = ComparisonDocument("StringContains", a, b, sensitive);
        Assert.Equal(new[] { expected ? "True" : "False" }, await Execute(document));
    }

    [Theory]
    [InlineData("Starts With", "İstanbul", "i\u0307", false, true)]
    [InlineData("Ends With", "ΟΣ", "Σ", false, false)]
    [InlineData("Equal", "ΟΣ", "ος", false, true)]
    [InlineData("Equal", "Straße", "STRASSE", false, false)]
    [InlineData("Equal", "é", "e\u0301", false, false)]
    [InlineData("Equal", "😀", "😀", true, true)]
    [InlineData("Starts With", "abc", "A", true, false)]
    public async Task AllCompareModesUseTheRealPreviewAndWholeStringLower(string mode, string a, string b, bool sensitive, bool expected)
    {
        var document = ComparisonDocument("StringCompare", a, b, sensitive, mode);
        Assert.Equal(new[] { expected ? "True" : "False" }, await Execute(document));
    }

    [Theory]
    [InlineData("StringContains")]
    [InlineData("StringCompare")]
    public async Task ConnectedBooleanOverridesTheSavedTrueWidget(string type)
    {
        var document = ComparisonDocument(type, "ABC", "abc", true, "Equal", connectedInsensitive: true);
        var compiled = PromptCompiler.Compile(document); Assert.True(compiled.Success);
        Assert.True(JsonNode.DeepEquals(new JsonArray("2", 0), compiled.Prompt!["10"]!["inputs"]!["case_sensitive"]));
        Assert.Equal(new[] { "True" }, await Execute(document));
    }

    [Fact]
    public async Task RealContainsMapsTwoCreateListsAndRepeatsTheShorterColumn()
    {
        var prompt = JsonNode.Parse("""
            {"a":{"class_type":"CreateList","inputs":{"inputs.input0":"ΟΣ","inputs.input1":"abc","inputs.input2":"ABC"}},
             "b":{"class_type":"CreateList","inputs":{"inputs.input0":"ος","inputs.input1":"bc"}},
             "10":{"class_type":"StringContains","inputs":{"string":["a",0],"substring":["b",0],"case_sensitive":false}},
             "20":{"class_type":"PreviewAny","inputs":{"source":["10",0]}}}
            """)!.AsObject();
        Assert.Equal(new[] { "True", "True", "True" }, await Execute(prompt));
    }

    [Fact]
    public async Task RealCompareMapsWildcardModesAndTextWithoutAStringToComboLink()
    {
        // CreateList's actual wildcard output is compatible with COMBO; PrimitiveString is not.
        var prompt = JsonNode.Parse("""
            {"modes":{"class_type":"CreateList","inputs":{"inputs.input0":"Starts With","inputs.input1":"Ends With","inputs.input2":"Equal"}},
             "needles":{"class_type":"CreateList","inputs":{"inputs.input0":"a","inputs.input1":"c"}},
             "10":{"class_type":"StringCompare","inputs":{"string_a":"abc","string_b":["needles",0],"mode":["modes",0],"case_sensitive":true}},
             "20":{"class_type":"PreviewAny","inputs":{"source":["10",0]}}}
            """)!.AsObject();
        Assert.Equal(new[] { "True", "True", "False" }, await Execute(prompt));
    }

    [Theory]
    [InlineData("contains-missing-bool", "required_input_missing")]
    [InlineData("compare-missing-bool", "required_input_missing")]
    [InlineData("invalid-mode", "invalid_input_type")]
    [InlineData("object-mode", "invalid_input_type")]
    [InlineData("string-to-bool", "type_mismatch")]
    public async Task InvalidInputsAreRejectedBeforeQueueWithoutInventingLiteralCoercionRules(string scenario, string expectedCode)
    {
        bool contains = scenario == "contains-missing-bool";
        var inputs = contains ? new JsonObject { ["string"] = "a", ["substring"] = "a", ["case_sensitive"] = true }
            : new JsonObject { ["string_a"] = "a", ["string_b"] = "a", ["mode"] = "Equal", ["case_sensitive"] = true };
        if (scenario.EndsWith("missing-bool", StringComparison.Ordinal)) inputs.Remove("case_sensitive");
        if (scenario == "invalid-mode") inputs["mode"] = "equal";
        if (scenario == "object-mode") inputs["mode"] = new JsonObject { ["mode"] = "Equal" };
        var prompt = new JsonObject
        {
            ["10"] = new JsonObject { ["class_type"] = contains ? "StringContains" : "StringCompare", ["inputs"] = inputs },
            ["20"] = new JsonObject { ["class_type"] = "PreviewAny", ["inputs"] = new JsonObject { ["source"] = new JsonArray("10", 0) } }
        };
        if (scenario == "string-to-bool")
        {
            prompt["2"] = new JsonObject { ["class_type"] = "PrimitiveString", ["inputs"] = new JsonObject { ["value"] = "false" } };
            inputs["case_sensitive"] = new JsonArray("2", 0);
        }
        string before = prompt.ToJsonString();
        await using var factory = new WebApplicationFactory<Program>(); using var client = factory.CreateClient();
        using var response = await client.PostAsJsonAsync("/prompt", Body(prompt));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains(expectedCode, (await response.Content.ReadFromJsonAsync<JsonObject>())!.ToJsonString());
        var queue = await client.GetFromJsonAsync<JsonObject>("/queue");
        Assert.Empty(queue!["queue_pending"]!.AsArray()); Assert.Empty(queue["queue_running"]!.AsArray());
        Assert.Equal(before, prompt.ToJsonString());
    }

    private static WorkflowDocument ComparisonDocument(string type, string a, string b, bool sensitive,
        string mode = "Equal", bool connectedInsensitive = false)
    {
        var widgets = type == "StringContains" ? new JsonArray(a, b, sensitive) : new JsonArray(a, b, mode, sensitive);
        var comparison = new JsonObject
        {
            ["id"] = 10, ["type"] = type, ["widgets_values"] = widgets,
            ["inputs"] = new JsonArray(new JsonObject { ["name"] = "case_sensitive", ["type"] = "BOOLEAN", ["link"] = null }),
            ["outputs"] = new JsonArray(new JsonObject { ["name"] = "BOOLEAN", ["type"] = "BOOLEAN", ["links"] = new JsonArray() })
        };
        var nodes = new JsonArray(comparison, new JsonObject
        {
            ["id"] = 20, ["type"] = "PreviewAny", ["widgets_values"] = new JsonArray(), ["outputs"] = new JsonArray(),
            ["inputs"] = new JsonArray(new JsonObject { ["name"] = "source", ["type"] = "*", ["link"] = null })
        });
        if (connectedInsensitive) nodes.Add(new JsonObject
        {
            ["id"] = 2, ["type"] = "PrimitiveBoolean", ["widgets_values"] = new JsonArray(false),
            ["outputs"] = new JsonArray(new JsonObject { ["name"] = "BOOLEAN", ["type"] = "BOOLEAN", ["links"] = new JsonArray() })
        });
        var document = WorkflowDocument.Parse(new JsonObject { ["version"] = 1, ["nodes"] = nodes, ["links"] = new JsonArray() }.ToJsonString());
        document.Connect(new("10"), 0, new("20"), 0);
        if (connectedInsensitive) document.Connect(new("2"), 0, new("10"), 0);
        return document;
    }
    private static async Task<string[]> Execute(WorkflowDocument document)
    {
        string before = document.ToJson(); var compilation = PromptCompiler.Compile(document);
        Assert.True(compilation.Success, string.Join("; ", compilation.Diagnostics));
        var result = await Execute(compilation.Prompt!); Assert.Equal(before, document.ToJson()); return result;
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
            string state = job!["status"]!.GetValue<string>();
            if (state == "completed") break;
            Assert.DoesNotContain(state, new[] { "failed", "cancelled" }); await Task.Delay(10, timeout.Token);
        }
        var history = (await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token))![id]!;
        Assert.Equal("20", Assert.Single(history["outputs"]!.AsObject()).Key);
        Assert.Empty(history["status"]!["messages"]!.AsArray()); Assert.Equal(before, prompt.ToJsonString());
        return Strings(history["outputs"]!["20"]!["text"]);
    }
    private static JsonObject Body(JsonObject prompt) => new() { ["prompt"] = prompt.DeepClone(), ["partial_execution_targets"] = new JsonArray("20") };
    private static string[] Strings(JsonNode? array) => array!.AsArray().Select(v => v!.GetValue<string>()).ToArray();
}
