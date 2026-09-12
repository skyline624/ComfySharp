using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class WorkflowDuplicationTests
{
    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void Duplicate_preserves_unknown_data_and_remaps_internal_links_in_one_atomic_undo(double version)
    {
        var doc = Document(version); string before = doc.ToJson(); int changed = 0; doc.Changed += (_, _) => changed++;
        var map = doc.DuplicateNodes([new("01"), new("vendor:2"), new("01")], 13, -17);
        Assert.Equal(2, map.Count); Assert.Equal(1, changed); Assert.Equal(5, doc.Nodes.Count); Assert.Equal(3, doc.Links.Count);
        Assert.Equal("101", map[new("01")].Value); Assert.Equal("102", map[new("vendor:2")].Value);
        var source = doc.Nodes.Single(n => n.Id == map[new("01")]); var target = doc.Nodes.Single(n => n.Id == map[new("vendor:2")]);
        Assert.Equal(23, source.X); Assert.Equal(3, source.Y); Assert.Equal(43, target.X); Assert.Equal(23, target.Y);
        Assert.Equal("tail", source.Data["pos"]![2]!.GetValue<string>());
        Assert.Equal("keep", target.Data["pos"]!["vendor"]!.GetValue<string>());
        Assert.Equal(4, target.Data["mode"]!.GetValue<int>());
        Equal(JsonNode.Parse("{\"nested\":[null,1,\"vendor:2\"]}"), target.Data["opaque"]);
        var link = doc.Links.Single(l => l.Target == target.Id);
        Assert.Equal(51, link.Id); Assert.Equal(source.Id, link.Source);
        Assert.Equal(51, target.Data["inputs"]![0]!["link"]!.GetValue<long>());
        Assert.Equal(51, Assert.Single(source.Data["outputs"]![0]!["links"]!.AsArray())!.GetValue<long>());
        Assert.Empty(target.Data["outputs"]![0]!["links"]!.AsArray()); // No connection to the unselected destination.
        var raw = doc.Snapshot()["links"]!.AsArray().Last()!;
        Equal(JsonNode.Parse("{\"extra\":true}"), version == 1 ? raw["vendor"] : raw[6]);
        Assert.Equal(102, (version == 1 ? doc.Snapshot()["state"]!["lastNodeId"] : doc.Snapshot()["last_node_id"])!.GetValue<long>());
        string after = doc.ToJson(); doc.Undo(); Assert.Equal(before, doc.ToJson()); doc.Redo(); Assert.Equal(after, doc.ToJson());
        Equal(doc.Snapshot(), WorkflowDocument.Parse(WorkflowDocument.Parse(after).ToJson()).Snapshot());
    }

    [Theory]
    [InlineData(0.4, false)]
    [InlineData(0.4, true)]
    [InlineData(1.0, false)]
    [InlineData(1.0, true)]
    public void External_input_connections_are_opt_in_and_existing_outgoing_connections_are_never_retargeted(double version, bool connectInputs)
    {
        var doc = Document(version); var originalSource = doc.Nodes[0].Data.DeepClone();
        var map = doc.DuplicateNodes([new("vendor:2")], connectInputs: connectInputs); var copy = doc.Nodes.Single(n => n.Id == map[new("vendor:2")]);
        Assert.Equal(connectInputs ? 3 : 2, doc.Links.Count);
        if (connectInputs)
        {
            var link = doc.Links.Single(l => l.Target == copy.Id); Assert.Equal(new NodeId("01"), link.Source);
            Assert.Equal(2, doc.Nodes[0].Data["outputs"]![0]!["links"]!.AsArray().Count);
        }
        else { Assert.Null(copy.Data["inputs"]![0]!["link"]); Equal(originalSource, doc.Nodes[0].Data); }
        Assert.Equal(new NodeId("vendor:2"), doc.Links.Single(l => l.Target.Value == "external").Source);
        Assert.Empty(copy.Data["outputs"]![0]!["links"]!.AsArray());
    }

    [Fact]
    public void Copies_and_returned_snapshots_have_no_shared_json_storage()
    {
        var doc = Document(1); var map = doc.DuplicateNodes([new("01"), new("vendor:2")]);
        doc.SetWidgets(map[new("01")], new JsonObject { ["value"] = "changed" });
        Assert.Equal("original", doc.Nodes.Single(n => n.Id.Value == "01").Data["widgets_values"]!["value"]!.GetValue<string>());
        var snapshot = doc.Snapshot(); snapshot["nodes"]!.AsArray().Last()!["opaque"]!["nested"]!.AsArray().Add(99);
        Assert.Equal(3, doc.Nodes.Last().Data["opaque"]!["nested"]!.AsArray().Count);
    }

    [Fact]
    public void Non_clonable_nodes_are_skipped_and_an_empty_selection_has_no_undo_entry()
    {
        var root = Document(0.4).Snapshot(); root["nodes"]![0]!["clonable"] = false;
        var doc = WorkflowDocument.Parse(root.ToJsonString()); string before = doc.ToJson();
        Assert.Empty(doc.DuplicateNodes([])); Assert.Empty(doc.DuplicateNodes([new("01")]));
        Assert.Equal(before, doc.ToJson()); Assert.False(doc.CanUndo);
        var map = doc.DuplicateNodes([new("01"), new("vendor:2")]); Assert.Single(map);
        Assert.Null(doc.Nodes.Last().Data["inputs"]![0]!["link"]);
    }

    [Theory]
    [InlineData("input")]
    [InlineData("output")]
    [InlineData("node-budget")]
    [InlineData("link-budget")]
    [InlineData("reroute")]
    [InlineData("subgraph")]
    public void Invalid_or_unported_structures_roll_back_every_copy_and_counter(string corruption)
    {
        var root = Document(1).Snapshot();
        switch (corruption)
        {
            case "input": root["nodes"]![1]!["inputs"]![0]!["link"] = 999; break;
            case "output": root["nodes"]![0]!["outputs"]![0]!["links"] = new JsonArray(); break;
            case "node-budget": root["state"]!["lastNodeId"] = 9007199254740990L; break;
            case "link-budget": root["state"]!["lastLinkId"] = 9007199254740991L; break;
            case "reroute": root["links"]![0]!["parentId"] = 1; break;
            case "subgraph": root["definitions"] = JsonNode.Parse("""{"subgraphs":[{"id":"UnportedNode"}]}"""); break;
        }
        var doc = WorkflowDocument.Parse(root.ToJsonString()); string before = doc.ToJson(); int changes = 0; doc.Changed += (_, _) => changes++;
        var error = Record.Exception(() => doc.DuplicateNodes([new("01"), new("vendor:2")])); Assert.NotNull(error);
        Assert.True(error is FormatException or InvalidOperationException or NotSupportedException);
        Assert.Equal(before, doc.ToJson()); Assert.False(doc.CanUndo); Assert.Equal(0, changes);
    }

    [Fact]
    public void Bad_selection_and_nonfinite_offsets_never_change_the_document()
    {
        var doc = Document(0.4); string before = doc.ToJson();
        Assert.Throws<ArgumentException>(() => doc.DuplicateNodes([new("absent")]));
        Assert.Throws<ArgumentOutOfRangeException>(() => doc.DuplicateNodes([new("01")], double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => doc.DuplicateNodes([new("01")], offsetY: double.PositiveInfinity));
        Assert.Equal(before, doc.ToJson()); Assert.False(doc.CanUndo);
    }

    [Fact]
    public void Copied_executable_prompt_uses_new_ids_and_independent_edited_widgets()
    {
        var doc = ApiPromptImport.Parse("""
            {"1":{"class_type":"PrimitiveString","inputs":{"value":"old"}},"01":{"class_type":"PreviewAny","inputs":{"source":["1",0]}}}
            """);
        var map = doc.DuplicateNodes([new("1"), new("01")]);
        doc.SetWidgets(map[new("1")], new JsonObject { ["value"] = "new" });
        var compiled = PromptCompiler.Compile(doc); Assert.True(compiled.Success);
        Assert.Equal("old", compiled.Prompt!["1"]!["inputs"]!["value"]!.GetValue<string>());
        Assert.Equal("new", compiled.Prompt[map[new("1")].Value]!["inputs"]!["value"]!.GetValue<string>());
        Assert.Equal(map[new("1")].Value, compiled.Prompt[map[new("01")].Value]!["inputs"]!["source"]![0]!.GetValue<string>());
        Assert.Equal("1", compiled.Prompt["01"]!["inputs"]!["source"]![0]!.GetValue<string>());
    }

    private static WorkflowDocument Document(double version)
    {
        var root = JsonNode.Parse("""
            {"version":0.4,"last_node_id":100,"last_link_id":50,"nodes":[
             {"id":"01","type":"PrimitiveString","pos":[10,20,"tail"],"widgets_values":{"value":"original"},"outputs":[{"name":"out","type":"STRING","links":[7],"vendor":"port"}]},
             {"id":"vendor:2","type":"UnportedNode","mode":4,"pos":{"0":30,"1":40,"vendor":"keep"},"opaque":{"nested":[null,1,"vendor:2"]},"inputs":[{"name":"in","type":"STRING","link":7}],"outputs":[{"name":"out","type":"STRING","links":[8]}]},
             {"id":"external","type":"PreviewAny","pos":[70,80],"inputs":[{"name":"source","type":"STRING","link":8}]}],
             "links":[[7,"01",0,"vendor:2",0,"STRING",{"extra":true}],[8,"vendor:2",0,"external",0,"STRING"]],"extra":{"preserve":true}}
            """)!.AsObject();
        if (version == 1)
        {
            root["version"] = 1; root["state"] = new JsonObject { ["lastNodeId"] = 100, ["lastLinkId"] = 50, ["vendor"] = "state" };
            root.Remove("last_node_id"); root.Remove("last_link_id");
            root["links"] = new JsonArray(root["links"]!.AsArray().Select(l => (JsonNode)new JsonObject { ["id"] = l![0]!.DeepClone(), ["origin_id"] = l[1]!.DeepClone(), ["origin_slot"] = l[2]!.DeepClone(), ["target_id"] = l[3]!.DeepClone(), ["target_slot"] = l[4]!.DeepClone(), ["type"] = l[5]!.DeepClone(), ["vendor"] = l.AsArray().Count > 6 ? l[6]!.DeepClone() : null }).ToArray());
        }
        return WorkflowDocument.Parse(root.ToJsonString());
    }
    private static void Equal(JsonNode? expected, JsonNode? actual) => Assert.True(JsonNode.DeepEquals(expected, actual));
}
