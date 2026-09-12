using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class ApiPromptImportTests
{
    [Fact]
    public void Reconstructs_links_without_normalizing_string_ids_and_round_trips_unknown_metadata()
    {
        const string json = """
            {"01":{"class_type":"PrimitiveString","inputs":{"value":"one"},"_meta":{"title":"First","vendor":[1,null]},"extra":{"opaque":true}},
             "1":{"class_type":"PrimitiveString","inputs":{"value":"two"}},
             "nested:1":{"class_type":"PreviewAny","inputs":{"source":["01",0]},"_meta":null},
             "PREVIEW":{"class_type":"PreviewAny","inputs":{"source":["1",0]},"_meta":{"title":"Upper"}}}
            """;
        var document = ApiPromptImport.Parse(json);
        Assert.Equal(new[] { "01", "1", "nested:1", "PREVIEW" }, document.Nodes.Select(n => n.Id.Value));
        Assert.Equal(2, document.Links.Count);
        Equal(JsonNode.Parse(json), Compile(document));
        for (int cycle = 0; cycle < 2; cycle++) document = WorkflowDocument.Parse(document.ToJson());
        Equal(JsonNode.Parse(json), Compile(document));
        document.SetWidgets(new("01"), new JsonObject { ["value"] = "edited" });
        document.Rename(new("01"), "New title");
        var compiled = Compile(document);
        Assert.Equal("edited", compiled["01"]!["inputs"]!["value"]!.GetValue<string>());
        Assert.Equal("New title", compiled["01"]!["_meta"]!["title"]!.GetValue<string>());
        Equal(JsonNode.Parse("[1,null]"), compiled["01"]!["_meta"]!["vendor"]);
        document.Undo(); document.Undo(); Equal(JsonNode.Parse(json), Compile(document));
    }

    [Theory]
    [InlineData("null", "null")]
    [InlineData("true", "true")]
    [InlineData("12.25", "12.25")]
    [InlineData("\"text\"", "\"text\"")]
    [InlineData("[1,2]", "{\"__value__\":[1,2]}")]
    [InlineData("[]", "{\"__value__\":[]}")]
    [InlineData("[\"a\",\"b\"]", "{\"__value__\":[\"a\",\"b\"]}")]
    [InlineData("{\"__value__\":[\"missing\",0],\"vendor\":1}", "{\"__value__\":[\"missing\",0],\"vendor\":1}")]
    [InlineData("{\"__value__\":{\"__value__\":[1]},\"extra\":true}", "{\"__value__\":{\"__value__\":[1]},\"extra\":true}")]
    public void Literal_values_are_editable_named_widgets_and_arrays_never_become_connections(string value, string expected)
    {
        var document = ApiPromptImport.Parse("{\"preview\":{\"class_type\":\"PreviewAny\",\"inputs\":{\"source\":" + value + "}}}");
        Assert.Empty(document.Links);
        Equal(JsonNode.Parse(value), document.Nodes[0].Data["widgets_values"]!["source"]);
        Equal(JsonNode.Parse(expected), Compile(document)["preview"]!["inputs"]!["source"]);
        document.SetWidgets(new("preview"), new JsonObject { ["source"] = "changed" });
        Assert.Equal("changed", Compile(document)["preview"]!["inputs"]!["source"]!.GetValue<string>());
        document.Undo(); Equal(JsonNode.Parse(expected), Compile(document)["preview"]!["inputs"]!["source"]);
    }

    [Fact]
    public void Unknown_nodes_remain_editable_and_connected_but_cannot_execute_until_ported()
    {
        var document = ApiPromptImport.Parse("""
            {"vendor":{"class_type":"ExternalNode","inputs":{"custom":{"opaque":[1,null]}},"external":[1,2]},
             "preview":{"class_type":"PreviewAny","inputs":{"source":["vendor",2]}}}
            """);
        var failure = PromptCompiler.Compile(document);
        Assert.Null(failure.Prompt); Assert.Contains(failure.Diagnostics, d => d.Code == "unsupported_node");
        Assert.Equal(3, document.Nodes[0].Data["outputs"]!.AsArray().Count);
        string initial = document.ToJson(); document.Move(new("vendor"), 7, 8); document.Undo(); Assert.Equal(initial, document.ToJson());
        var definitions = new Dictionary<string, NodeDefinition>(PromptCompiler.BaseDefinitions) { ["ExternalNode"] = new("ExternalNode", []) };
        var compiled = PromptCompiler.Compile(WorkflowDocument.Parse(initial), definitions);
        Assert.True(compiled.Success); Equal(JsonNode.Parse("[1,2]"), compiled.Prompt!["vendor"]!["external"]);
        var unavailable = PromptCompiler.Compile(document, definitions, new HashSet<string> { "PreviewAny" });
        Assert.Contains(unavailable.Diagnostics, d => d.Code == "unavailable_node" && d.Node == new NodeId("vendor"));
    }

    [Fact]
    public void Template_types_are_cloned_and_do_not_inject_values_or_extend_known_outputs()
    {
        var template = JsonNode.Parse("""
            {"inputs":[],"outputs":[{"name":"STRING","type":"STRING","links":[99]}],"widgets_values":["DO NOT INJECT"]}
            """)!.AsObject(); string before = template.ToJsonString();
        var document = ApiPromptImport.Parse("""
            {"a":{"class_type":"PrimitiveString","inputs":{}},"p":{"class_type":"PreviewAny","inputs":{"source":["a",4]}}}
            """, _ => template);
        Assert.Equal(before, template.ToJsonString()); Assert.Empty(document.Nodes[0].Data["widgets_values"]!.AsObject());
        Assert.Single(document.Nodes[0].Data["outputs"]!.AsArray());
        var errors = PromptCompiler.Compile(document).Diagnostics;
        Assert.Contains(errors, d => d.Code == "missing_widgets"); Assert.Contains(errors, d => d.Code == "invalid_output");
    }

    [Fact]
    public void Dynamic_inputs_accept_literals_and_connections_and_connection_edits_are_authoritative()
    {
        var document = ApiPromptImport.Parse("""
            {"s":{"class_type":"PrimitiveString","inputs":{"value":"connected"}},
             "list":{"class_type":"CreateList","inputs":{"inputs.input0":[1,2],"inputs.input1":["s",0]}},
             "format":{"class_type":"StringFormat","inputs":{"f_string":"{a}","values.a":true}}}
            """);
        var first = Compile(document);
        Equal(new JsonArray("s", 0), first["list"]!["inputs"]!["inputs.input1"]);
        document.SetWidgets(new("list"), new JsonObject { ["inputs.input0"] = new JsonArray(1,2), ["inputs.input1"] = "fallback" });
        document.Disconnect(document.Links.Single().Id);
        Assert.Equal("fallback", Compile(document)["list"]!["inputs"]!["inputs.input1"]!.GetValue<string>());
        document.Connect(new("s"), 0, new("list"), 1);
        Equal(new JsonArray("s", 0), Compile(document)["list"]!["inputs"]!["inputs.input1"]);
    }

    [Theory]
    [InlineData("[\"absent\",0]", "invalid_link")]
    [InlineData("[\"p\",0]", "cycle")]
    [InlineData("[\"p\",-1]", "invalid_output")]
    public void Invalid_graphs_are_preserved_with_compilation_diagnostics(string link, string code)
    {
        var document = ApiPromptImport.Parse("{\"p\":{\"class_type\":\"PreviewAny\",\"inputs\":{\"source\":" + link + "}}}");
        Assert.Single(document.Links);
        var compiled = PromptCompiler.Compile(WorkflowDocument.Parse(document.ToJson()));
        Assert.Null(compiled.Prompt); Assert.Contains(compiled.Diagnostics, d => d.Code == code);
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("[]")]
    [InlineData("{\"p\":{\"class_type\":\"PreviewAny\"}}")]
    [InlineData("{\"p\":{\"class_type\":1,\"inputs\":{}}}")]
    [InlineData("{\"p\":{\"class_type\":\"PreviewAny\",\"inputs\":[]}}")]
    public void Detection_and_import_reject_non_prompt_documents(string json)
    {
        Assert.False(ApiPromptImport.IsPrompt(JsonNode.Parse(json)));
        Assert.Throws<FormatException>(() => ApiPromptImport.Parse(json));
    }

    [Theory]
    [InlineData("0.5")]
    [InlineData("2147483648")]
    [InlineData("4096")]
    public void Unrepresentable_links_or_excessive_inferred_slots_fail_explicitly(string slot) =>
        Assert.Throws<FormatException>(() => ApiPromptImport.Parse("{\"p\":{\"class_type\":\"PreviewAny\",\"inputs\":{\"source\":[\"p\"," + slot + "]}}}"));

    [Theory]
    [InlineData("0.0", 0)]
    [InlineData("1e0", 1)]
    [InlineData("true", 1)]
    [InlineData("false", 0)]
    public void Numeric_and_python_boolean_link_slots_are_normalized(string slot, int expected)
    {
        var document = ApiPromptImport.Parse("{\"p\":{\"class_type\":\"PreviewAny\",\"inputs\":{\"source\":[\"absent\"," + slot + "]}}}");
        Assert.Equal(expected, Assert.Single(document.Links).SourceSlot);
    }

    [Fact]
    public void Compiled_extension_fields_are_detached_and_unknown_import_versions_do_not_execute()
    {
        var document = ApiPromptImport.Parse("""
            {"p":{"class_type":"PreviewAny","inputs":{"source":"hello"},"vendor":{"nested":[1]}}}
            """);
        string before = document.ToJson();
        Compile(document)["p"]!["vendor"]!["nested"]!.AsArray().Add(2);
        Assert.Equal(before, document.ToJson());
        var root = document.Snapshot(); root["nodes"]![0]!["properties"]!["comfysharp.api_import"]!["version"] = 2;
        var result = PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString()));
        Assert.Null(result.Prompt); Assert.Contains(result.Diagnostics, d => d.Code == "invalid_api_import");
    }

    [Fact]
    public void Named_values_without_a_binding_still_fail_and_dynamic_names_are_not_bypassed()
    {
        var document = ApiPromptImport.Parse("""
            {"p":{"class_type":"StringFormat","inputs":{"f_string":"{a}","values.aa":12}}}
            """);
        Assert.Contains(PromptCompiler.Compile(document).Diagnostics, d => d.Code == "invalid_input_name");
        document = ApiPromptImport.Parse("""{"p":{"class_type":"PreviewAny","inputs":{"source":"hello"}}}""");
        var root = document.Snapshot(); root["nodes"]![0]!["inputs"]![0]!.AsObject().Remove("widget");
        Assert.Contains(PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString())).Diagnostics, d => d.Code == "unknown_widget");
    }

    private static JsonObject Compile(WorkflowDocument document)
    {
        var result = PromptCompiler.Compile(document); Assert.True(result.Success, string.Join("; ", result.Diagnostics)); return result.Prompt!;
    }
    private static void Equal(JsonNode? expected, JsonNode? actual) => Assert.True(JsonNode.DeepEquals(expected, actual), $"Expected {expected}; actual {actual}");
}
