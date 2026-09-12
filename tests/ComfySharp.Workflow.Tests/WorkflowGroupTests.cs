using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class WorkflowGroupTests
{
    [Fact]
    public void Native_json_integer_template_dimensions_can_be_grouped_before_serialization()
    {
        var doc = WorkflowDocument.Create(); var node = doc.AddNode("PrimitiveString");
        int index = doc.GroupNodes([node]); Assert.True(doc.Groups[index].Bounds.Width >= 220);
        doc.MoveGroup(index, 10, 20); Assert.Equal(110, doc.Nodes[0].X); Assert.Equal(120, doc.Nodes[0].Y);
    }
    [Theory]
    [InlineData(0.4)]
    [InlineData(1)]
    public void Imported_group_data_roundtrips_without_assigning_missing_ids_and_edits_preserve_unknown_fields(double version)
    {
        var doc = Scene(version); var before = doc.Snapshot(); Assert.Null(doc.Groups[0].Data["id"]);
        Assert.True(JsonNode.DeepEquals(before, WorkflowDocument.Parse(WorkflowDocument.Parse(doc.ToJson()).ToJson()).Snapshot()));
        doc.UpdateGroup(0, "Renamed", "#123456", 500, 350);
        Assert.Equal("keep", doc.Groups[0].Data["vendor"]!["nested"]!.GetValue<string>()); Assert.Equal(42, doc.Groups[0].Data["font_size"]!.GetValue<int>());
        Assert.Equal("tail", doc.Groups[0].Data["bounding"]![4]!.GetValue<string>()); Assert.Equal(500, doc.Groups[0].Bounds.Width);
        doc.Undo(); Assert.True(JsonNode.DeepEquals(before, doc.Snapshot()));
    }

    [Theory]
    [InlineData(0.4, 8)]
    [InlineData(1, 101)]
    public void New_group_fits_selected_measured_bounds_and_allocates_its_own_id(double version, long expectedId)
    {
        var doc = Scene(version); string before = doc.ToJson();
        var measured = new Dictionary<NodeId, GraphRect> { [new("a")] = new(100, 90, 120, 60), [new("b")] = new(300, 200, 90, 70) };
        int index = doc.GroupNodes([new("a"), new("b"), new("a")], "Selection", measured);
        Assert.Equal(new GraphRect(90, 50, 310, 230), doc.Groups[index].Bounds); Assert.Equal(expectedId, doc.Groups[index].Data["id"]!.GetValue<long>());
        doc.Undo(); Assert.Equal(before, doc.ToJson());
        index = doc.AddGroup("Small", new(0, 0, 2, 3)); Assert.Equal(140, doc.Groups[index].Bounds.Width); Assert.Equal(80, doc.Groups[index].Bounds.Height);
    }

    [Theory]
    [InlineData(0.4, false)]
    [InlineData(0.4, true)]
    [InlineData(1, false)]
    [InlineData(1, true)]
    public void Group_move_uses_node_centers_moves_nested_groups_once_and_keeps_pins_and_external_nodes(double version, bool frameOnly)
    {
        var doc = Scene(version); string before = doc.ToJson(); int changes = 0; doc.Changed += (_, _) => changes++;
        doc.MoveGroup(0, 25, -10, frameOnly);
        Assert.Equal(1, changes); Assert.Equal(25, doc.Groups[0].Bounds.X); Assert.Equal(-10, doc.Groups[0].Bounds.Y);
        Assert.Equal(frameOnly ? 100 : 125, doc.Groups[1].Bounds.X);
        Assert.Equal(frameOnly ? 100 : 125, doc.Nodes.Single(n => n.Id.Value == "a").X);
        Assert.Equal(390, doc.Nodes.Single(n => n.Id.Value == "b").X); // Partially overlaps; center is outside.
        Assert.Equal(100, doc.Nodes.Single(n => n.Id.Value == "pinned").X);
        var reroutes = version == 1 ? doc.Snapshot()["reroutes"] : doc.Snapshot()["extra"]!["reroutes"];
        Assert.Equal(frameOnly ? 120 : 145, reroutes![0]!["pos"]![0]!.GetValue<double>());
        Assert.Equal("metadata", reroutes[0]!["vendor"]!.GetValue<string>());
        string after = doc.ToJson(); doc.Undo(); Assert.Equal(before, doc.ToJson()); doc.Redo(); Assert.Equal(after, doc.ToJson());
    }

    [Fact]
    public void Pinned_group_keeps_bounds_but_can_be_renamed_unpinned_and_removed_without_deleting_nodes()
    {
        var doc = Scene(1); doc.SetGroupPinned(0, true); var bounds = doc.Groups[0].Bounds;
        doc.MoveGroup(0, 200, 200); Assert.Equal(bounds, doc.Groups[0].Bounds);
        doc.UpdateGroup(0, "Pinned title", "#456", 600, 600); Assert.Equal(bounds, doc.Groups[0].Bounds); Assert.Equal("Pinned title", doc.Groups[0].Title);
        doc.SetGroupPinned(0, false); Assert.Null(doc.Groups[0].Data["flags"]!["pinned"]);
        var nodes = doc.Snapshot()["nodes"]!.DeepClone(); string before = doc.ToJson(); doc.RemoveGroup(0);
        Assert.True(JsonNode.DeepEquals(nodes, doc.Snapshot()["nodes"])); Assert.Single(doc.Groups); doc.Undo(); Assert.Equal(before, doc.ToJson());
    }

    [Fact]
    public void Invalid_move_rolls_back_all_group_node_and_reroute_changes()
    {
        var root = Scene(1).Snapshot(); root["reroutes"] = new JsonObject { ["unsupported"] = true };
        var doc = WorkflowDocument.Parse(root.ToJsonString()); string before = doc.ToJson();
        Assert.Throws<FormatException>(() => doc.MoveGroup(0, 5, 7)); Assert.Equal(before, doc.ToJson()); Assert.False(doc.CanUndo);
        Assert.Throws<ArgumentOutOfRangeException>(() => doc.MoveGroup(0, double.PositiveInfinity, 0)); Assert.Equal(before, doc.ToJson());
        Assert.Throws<FormatException>(() => doc.AddGroup("Invalid", new(0, 0, -1, 80))); Assert.Equal(before, doc.ToJson());
    }

    [Theory]
    [InlineData("null")]
    [InlineData("[1,2,3]")]
    [InlineData("\"unknown\"")]
    [InlineData("{\"0\":1}")]
    public void Malformed_group_is_preserved_but_not_projected_as_an_editable_rectangle(string bounding)
    {
        var root = Scene(0.4).Snapshot(); root["groups"]![0]!["bounding"] = JsonNode.Parse(bounding);
        var doc = WorkflowDocument.Parse(root.ToJsonString()); string before = doc.ToJson();
        Assert.Throws<FormatException>(() => doc.Groups); Assert.Equal(before, doc.ToJson()); Assert.False(doc.CanUndo);
    }

    [Fact]
    public void Object_backed_positions_keep_extra_fields_and_collapsed_nodes_require_measured_bounds()
    {
        var root = Scene(1).Snapshot(); root["groups"]![0]!["bounding"] = JsonNode.Parse("{\"0\":0,\"1\":0,\"2\":400,\"3\":300,\"vendor\":true}");
        root["nodes"]![0]!["pos"] = JsonNode.Parse("{\"0\":100,\"1\":100,\"extra\":\"keep\"}"); root["nodes"]![0]!["flags"] = JsonNode.Parse("{\"collapsed\":true}");
        var doc = WorkflowDocument.Parse(root.ToJsonString()); Assert.Throws<NotSupportedException>(() => doc.MoveGroup(0, 10, 10)); Assert.False(doc.CanUndo);
        doc.MoveGroup(0, 10, 10, measuredBounds: new Dictionary<NodeId, GraphRect> { [new("a")] = new(100, 80, 100, 30) });
        Assert.Equal("keep", doc.Nodes[0].Data["pos"]!["extra"]!.GetValue<string>()); Assert.True(doc.Groups[0].Data["bounding"]!["vendor"]!.GetValue<bool>());
    }

    [Fact]
    public void Grouping_does_not_change_executable_prompt_or_existing_links()
    {
        var doc = ApiPromptImport.Parse("""{"a":{"class_type":"PrimitiveString","inputs":{"value":"grouped"}},"b":{"class_type":"PreviewAny","inputs":{"source":["a",0]}}}""");
        var before = PromptCompiler.Compile(doc).Prompt; var links = doc.Snapshot()["links"]!.DeepClone();
        int group = doc.GroupNodes(doc.Nodes.Select(n => n.Id)); doc.MoveGroup(group, 30, 40);
        Assert.True(JsonNode.DeepEquals(before, PromptCompiler.Compile(doc).Prompt)); Assert.True(JsonNode.DeepEquals(links, doc.Snapshot()["links"]));
    }

    private static WorkflowDocument Scene(double version)
    {
        var root = JsonNode.Parse("""
            {"version":0.4,"nodes":[
              {"id":"a","type":"Unknown","pos":[100,100],"size":[100,50]},
              {"id":"b","type":"Unknown","pos":[390,100],"size":[100,50]},
              {"id":"pinned","type":"Unknown","pos":[100,200],"size":[100,50],"flags":{"pinned":true}}],"links":[],
             "groups":[{"title":"Outer","bounding":[0,0,400,300,"tail"],"font_size":42,"vendor":{"nested":"keep"}},
                       {"id":7,"title":"Inner","bounding":[100,80,200,150]}],
             "extra":{"reroutes":[{"id":1,"pos":[120,120],"vendor":"metadata"}]}}
            """)!;
        if (version == 1) { root["version"] = 1; root["state"] = new JsonObject { ["lastGroupId"] = 100 }; root["reroutes"] = root["extra"]!["reroutes"]!.DeepClone(); root["extra"]!.AsObject().Remove("reroutes"); }
        return WorkflowDocument.Parse(root.ToJsonString());
    }
}
