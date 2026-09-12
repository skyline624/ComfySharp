using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class WorkflowModeTests
{
    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void Bypass_is_a_projection_and_modes_survive_edit_undo_and_round_trips(double version)
    {
        var doc = Graph(["STRING"], ["STRING"], "STRING", 0, version); string initial = doc.ToJson();
        var result = PromptCompiler.Compile(doc, availableNodes: new HashSet<string> { "PrimitiveString", "PreviewAny" });
        Assert.True(result.Success); Assert.Empty(result.Warnings); Assert.False(result.Prompt!.ContainsKey("b"));
        Assert.Equal("s0", result.Prompt["p"]!["inputs"]!["source"]![0]!.GetValue<string>());
        Assert.Equal(initial, doc.ToJson()); Assert.Equal(2, doc.Links.Count);
        doc.SetExecutionMode(new("b"), 2); Assert.Equal("fallback", Compile(doc)["p"]!["inputs"]!["source"]!.GetValue<string>());
        doc.Undo(); Assert.Equal(initial, doc.ToJson()); doc.Redo(); Assert.Equal(2, doc.Nodes.Single(n => n.Id.Value == "b").Data["mode"]!.GetValue<int>());
        var reopened = WorkflowDocument.Parse(WorkflowDocument.Parse(doc.ToJson()).ToJson());
        Assert.True(JsonNode.DeepEquals(doc.Snapshot(), reopened.Snapshot()));
        Assert.True(JsonNode.DeepEquals(Compile(doc), Compile(reopened)));
    }

    [Theory]
    [InlineData("[\"STRING\",\"*\"]", "[\"STRING\",\"STRING\"]", "STRING", 1, "s1")]
    [InlineData("[\"STRING\",\"INT\"]", "[\"INT\"]", "INT", 0, "s1")]
    [InlineData("[\"INT\",\"STRING\"]", "[\"IMAGE\"]", "STRING", 0, "s1")]
    [InlineData("[\"FLOAT\",\"image,LATENT\"]", "[\"IMAGE\"]", "image", 0, "s1")]
    [InlineData("[\"FLOAT\",\"IMAGE\"]", "[\"IMAGE\"]", "LATENT,IMAGE", 0, "s1")]
    [InlineData("[\"IMAGE\",\"FLOAT\"]", "[\"STRING\",\"STRING\"]", "*", 1, "s1")]
    [InlineData("[\"IMAGE\"]", "[\"STRING\",\"STRING\",\"STRING\"]", "*", 2, "s0")]
    [InlineData("[\"IMAGE\"]", "[\"STRING\",\"STRING\"]", "", 1, "s0")]
    [InlineData("[0]", "[\"STRING\"]", "IMAGE", 0, "s0")]
    [InlineData("[-1]", "[-1]", "-1", 0, "s0")]
    [InlineData("[\"FLOAT\",\"IMAGE,\"]", "[\"STRING\"]", "BOOLEAN", 0, "s1")]
    public void Slot_selection_follows_opposite_exact_compatible_and_wildcard_priority(string inputTypes, string outputTypes, string targetType, int outputSlot, string expected)
    {
        var doc = Graph(JsonNode.Parse(inputTypes)!.AsArray(), JsonNode.Parse(outputTypes)!.AsArray(), targetType, outputSlot);
        Assert.Equal(expected, Compile(doc)["p"]!["inputs"]!["source"]![0]!.GetValue<string>());
    }

    [Fact]
    public void Unconnected_selected_slot_does_not_search_other_inputs_or_use_bypass_widgets()
    {
        var doc = Graph(["STRING", "STRING"], ["STRING"], "STRING", 0);
        doc.Disconnect(doc.Links.Single(l => l.Target.Value == "b" && l.TargetSlot == 0).Id);
        doc.SetWidgets(new("b"), new JsonObject { ["input0"] = "do not use" });
        Assert.Equal("fallback", Compile(doc)["p"]!["inputs"]!["source"]!.GetValue<string>());
    }

    [Fact]
    public void No_matching_bypass_type_is_an_explicit_warning_not_a_fake_connection()
    {
        var doc = Graph(["INT"], ["INT"], "IMAGE", 0);
        var result = PromptCompiler.Compile(doc); Assert.True(result.Success);
        Assert.Equal("bypass_no_match", Assert.Single(result.Warnings).Code);
        Assert.Equal("fallback", result.Prompt!["p"]!["inputs"]!["source"]!.GetValue<string>());
    }

    [Fact]
    public void Each_hop_uses_the_selected_input_type_and_unknown_bypass_nodes_need_no_runtime_definition()
    {
        var doc = Graph(["IMAGE", "FLOAT"], ["IMAGE", "IMAGE"], "IMAGE", 1);
        var outer = doc.AddNode("AnotherUnported", template: new JsonObject
        {
            ["mode"] = 4, ["inputs"] = new JsonArray(new JsonObject { ["name"] = "any", ["type"] = "*", ["link"] = null }),
            ["outputs"] = new JsonArray(new JsonObject { ["name"] = "out", ["type"] = "IMAGE", ["links"] = new JsonArray() })
        });
        doc.Connect(new("b"), 1, outer, 0); doc.Connect(outer, 0, new("p"), 0);
        // The outer wildcard input changes the next lookup from IMAGE to '*', selecting b input1.
        Assert.Equal("s1", Compile(doc)["p"]!["inputs"]!["source"]![0]!.GetValue<string>());
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void Dormant_cycles_do_not_invalidate_independent_outputs(int mode)
    {
        var doc = Graph(["STRING"], ["STRING"], "STRING", 0);
        doc.Disconnect(doc.Links.Single(l => l.Target.Value == "p").Id);
        doc.Connect(new("b"), 0, new("b"), 0); doc.SetExecutionMode(new("b"), mode);
        Assert.Equal("fallback", Compile(doc)["p"]!["inputs"]!["source"]!.GetValue<string>());
    }

    [Fact]
    public void Reachable_bypass_cycle_fails_and_muting_it_restores_literal_fallback()
    {
        var doc = Graph(["STRING"], ["STRING"], "STRING", 0); doc.Connect(new("b"), 0, new("b"), 0);
        var result = PromptCompiler.Compile(doc); Assert.Null(result.Prompt); Assert.Contains(result.Diagnostics, d => d.Code == "cycle");
        doc.SetExecutionMode(new("b"), 2); Assert.Equal("fallback", Compile(doc)["p"]!["inputs"]!["source"]!.GetValue<string>());
    }

    [Fact]
    public void Missing_bypass_output_and_missing_selected_input_are_diagnosed()
    {
        var doc = Graph(["STRING"], ["STRING"], "STRING", 0);
        var root = doc.Snapshot(); root["nodes"]!.AsArray().OfType<JsonObject>().Single(n => n["id"]!.GetValue<string>() == "b")["outputs"] = new JsonArray();
        Assert.Contains(PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString())).Diagnostics, d => d.Code == "invalid_output");
        doc = Graph([], ["STRING"], "*", 0);
        Assert.Contains(PromptCompiler.Compile(doc).Diagnostics, d => d.Code == "invalid_input");
    }

    [Fact]
    public void Muted_source_omits_input_without_a_persisted_widget_and_keeps_null_widget_when_present()
    {
        var doc = Graph(["STRING"], ["STRING"], "STRING", 0); doc.SetExecutionMode(new("s0"), 2);
        doc.SetWidgets(new("p"), new JsonObject()); Assert.Empty(Compile(doc)["p"]!["inputs"]!.AsObject());
        doc.SetWidgets(new("p"), new JsonObject { ["source"] = null });
        var inputs = Compile(doc)["p"]!["inputs"]!.AsObject(); Assert.True(inputs.ContainsKey("source")); Assert.Null(inputs["source"]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    public void Non_skipped_event_modes_serialize_as_in_frontend(int mode)
    {
        var doc = ApiPromptImport.Parse("""{"p":{"class_type":"PreviewAny","inputs":{"source":"value"}}}""");
        doc.SetExecutionMode(new("p"), mode); Assert.Equal("value", Compile(doc)["p"]!["inputs"]!["source"]!.GetValue<string>());
        string before = doc.ToJson(); Assert.Throws<ArgumentOutOfRangeException>(() => doc.SetExecutionMode(new("p"), 5)); Assert.Equal(before, doc.ToJson());
    }

    private static JsonObject Compile(WorkflowDocument doc)
    {
        var result = PromptCompiler.Compile(doc); Assert.True(result.Success, string.Join("; ", result.Diagnostics)); return result.Prompt!;
    }

    private static WorkflowDocument Graph(JsonArray inputs, JsonArray outputs, string targetType, int sourceSlot, double version = 0.4)
    {
        var nodes = new JsonArray(); var links = new JsonArray(); var bypassInputs = new JsonArray();
        for (int i = 0; i < inputs.Count; i++)
        {
            nodes.Add(new JsonObject { ["id"] = "s" + i, ["type"] = "PrimitiveString", ["widgets_values"] = new JsonArray("value" + i),
                ["outputs"] = new JsonArray(new JsonObject { ["name"] = "out", ["type"] = inputs[i]?.DeepClone(), ["links"] = new JsonArray(i + 1) }) });
            bypassInputs.Add(new JsonObject { ["name"] = "input" + i, ["type"] = inputs[i]?.DeepClone(), ["link"] = i + 1 });
            links.Add(new JsonArray(i + 1, "s" + i, 0, "b", i, inputs[i]?.DeepClone()));
        }
        var bypassOutputs = new JsonArray();
        for (int i = 0; i < outputs.Count; i++) bypassOutputs.Add(new JsonObject { ["name"] = "out" + i, ["type"] = outputs[i]?.DeepClone(), ["links"] = i == sourceSlot ? new JsonArray(inputs.Count + 1) : new JsonArray() });
        nodes.Add(new JsonObject { ["id"] = "b", ["type"] = "UnportedBypass", ["mode"] = 4, ["inputs"] = bypassInputs, ["outputs"] = bypassOutputs, ["vendor"] = new JsonObject { ["keep"] = true } });
        nodes.Add(new JsonObject { ["id"] = "p", ["type"] = "PreviewAny", ["inputs"] = new JsonArray(new JsonObject { ["name"] = "source", ["type"] = targetType, ["link"] = inputs.Count + 1, ["widget"] = new JsonObject { ["name"] = "source" } }), ["widgets_values"] = new JsonObject { ["source"] = "fallback" } });
        links.Add(new JsonArray(inputs.Count + 1, "b", sourceSlot, "p", 0, targetType));
        if (version == 1) links = new JsonArray(links.Select(link => (JsonNode)new JsonObject { ["id"] = link![0]!.DeepClone(), ["origin_id"] = link[1]!.DeepClone(), ["origin_slot"] = link[2]!.DeepClone(), ["target_id"] = link[3]!.DeepClone(), ["target_slot"] = link[4]!.DeepClone(), ["type"] = link[5]?.DeepClone() }).ToArray());
        return WorkflowDocument.Parse(new JsonObject { ["version"] = version, ["nodes"] = nodes, ["links"] = links }.ToJsonString());
    }
}
