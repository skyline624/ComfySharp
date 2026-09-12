using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class WorkflowDeletionTests
{
    [Theory]
    [InlineData(0.4, false)]
    [InlineData(0.4, true)]
    [InlineData(1, false)]
    [InlineData(1, true)]
    public void Selected_chain_is_reconnected_in_either_order_as_one_undoable_edit(double version, bool reverse)
    {
        var doc = Chain(version, 4); string before = doc.ToJson(); int events = 0; doc.Changed += (_, _) => events++;
        NodeId[] selected = reverse ? [new("3"), new("2"), new("3")] : [new("2"), new("3")];
        var removed = doc.DeleteNodes(selected); Assert.Equal(2, removed.Count); Assert.Equal(1, events);
        Assert.Equal(new[] { "1", "4" }, doc.Nodes.Select(n => n.Id.Value));
        var link = Assert.Single(doc.Links); Assert.Equal(new NodeId("1"), link.Source); Assert.Equal(new NodeId("4"), link.Target);
        Assert.True(link.Id > 100); Assert.Equal(link.Id, doc.Nodes[1].Data["inputs"]![0]!["link"]!.GetValue<long>());
        Assert.Equal(link.Id, Assert.Single(doc.Nodes[0].Data["outputs"]![0]!["links"]!.AsArray())!.GetValue<long>());
        string after = doc.ToJson(); doc.Undo(); Assert.Equal(before, doc.ToJson()); doc.Redo(); Assert.Equal(after, doc.ToJson());
        Assert.True(JsonNode.DeepEquals(doc.Snapshot(), WorkflowDocument.Parse(after).Snapshot()));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Mismatched_slot_indexes_reconnect_only_when_keepAllLinksOnBypass_is_enabled(bool enabled)
    {
        var root = Chain(0.4, 3).Snapshot(); var middle = root["nodes"]![1]!;
        middle["outputs"]!.AsArray().Insert(0, JsonNode.Parse("{\"type\":\"IMAGE\",\"links\":[]}"));
        root["links"]![1]![2] = 1; middle["flags"] = new JsonObject { ["keepAllLinksOnBypass"] = enabled };
        var doc = WorkflowDocument.Parse(root.ToJsonString()); doc.DeleteNodes([new("2")]);
        Assert.Equal(enabled ? 1 : 0, doc.Links.Count);
        if (enabled) Assert.Equal(new NodeId("1"), doc.Links[0].Source); else Assert.Null(doc.Nodes[1].Data["inputs"]![0]!["link"]);
    }

    [Theory]
    [InlineData("block_delete", true)]
    [InlineData("ignore_remove", true)]
    [InlineData("removable", false)]
    public void Protected_nodes_are_left_intact_and_are_not_bypassed(string key, bool value)
    {
        var root = Chain(1, 3).Snapshot(); root["nodes"]![1]![key] = value;
        var doc = WorkflowDocument.Parse(root.ToJsonString()); string before = doc.ToJson();
        Assert.Empty(doc.DeleteNodes([new("2")])); Assert.Equal(before, doc.ToJson()); Assert.False(doc.CanUndo);
        Assert.Throws<InvalidOperationException>(() => doc.PrepareCut([new("2")]));
    }

    [Theory]
    [InlineData("input")]
    [InlineData("output")]
    [InlineData("dangling")]
    [InlineData("budget")]
    [InlineData("subgraph")]
    [InlineData("routed")]
    [InlineData("floating")]
    public void Invalid_or_unported_deletions_leave_document_and_undo_unchanged(string kind)
    {
        var root = Chain(1, 4).Snapshot();
        switch (kind)
        {
            case "input": root["nodes"]![3]!["inputs"]![0]!["link"] = 999; break;
            case "output": root["nodes"]![2]!["outputs"]![0]!["links"] = new JsonArray(); break;
            case "dangling": root["nodes"]![2]!["outputs"]![0]!["links"]!.AsArray().Add(999); break;
            case "budget": root["state"]!["lastLinkId"] = 9007199254740990L; break;
            case "subgraph": root["definitions"] = JsonNode.Parse("{\"subgraphs\":[{\"id\":\"Extension\"}]}"); break;
            case "routed": root["links"]![1]!["parentId"] = 2; break;
            case "floating": root["floatingLinks"] = JsonNode.Parse("[{\"origin_id\":\"3\",\"target_id\":null}]"); break;
        }
        var doc = WorkflowDocument.Parse(root.ToJsonString()); string before = doc.ToJson(); int changes = 0; doc.Changed += (_, _) => changes++;
        Assert.NotNull(Record.Exception(() => doc.DeleteNodes([new("2"), new("3")])));
        Assert.Equal(before, doc.ToJson()); Assert.False(doc.CanUndo); Assert.Equal(0, changes);
        Assert.NotNull(Record.Exception(() => doc.PrepareCut([new("2"), new("3")]))); Assert.Equal(before, doc.ToJson());
    }

    [Fact]
    public void Cut_plan_copies_only_removable_nodes_and_can_be_committed_once_without_modifying_clipboard_source_data()
    {
        var root = Chain(1, 3).Snapshot(); root["nodes"]![0]!["block_delete"] = true;
        var doc = WorkflowDocument.Parse(root.ToJsonString()); string before = doc.ToJson();
        var cut = doc.PrepareCut([new("1"), new("2")]); Assert.Equal(before, doc.ToJson()); Assert.False(doc.CanUndo);
        var payload = JsonNode.Parse(cut.ClipboardText)!; Assert.Single(payload["workflow"]!["nodes"]!.AsArray());
        doc.CommitCut(cut); Assert.Equal(2, doc.Nodes.Count); Assert.Equal(new NodeId("1"), Assert.Single(doc.Links).Source);
        Assert.Throws<InvalidOperationException>(() => doc.CommitCut(cut)); doc.Undo(); Assert.Equal(before, doc.ToJson());
        var destination = WorkflowDocument.Create(); destination.PasteNodes(cut.ClipboardText); Assert.Single(destination.Nodes);
    }

    [Fact]
    public void Cut_plan_rejects_other_documents_changed_source_and_nonclonable_selection()
    {
        var doc = Chain(0.4, 3); var plan = doc.PrepareCut([new("2")]); var other = WorkflowDocument.Parse(doc.ToJson());
        Assert.Throws<InvalidOperationException>(() => other.CommitCut(plan)); Assert.False(other.CanUndo);
        doc.Rename(new("1"), "edited"); string after = doc.ToJson(); Assert.Throws<InvalidOperationException>(() => doc.CommitCut(plan)); Assert.Equal(after, doc.ToJson());
        var root = doc.Snapshot(); root["nodes"]![1]!["clonable"] = false; var restricted = WorkflowDocument.Parse(root.ToJsonString());
        Assert.Throws<InvalidOperationException>(() => restricted.PrepareCut([new("2")])); Assert.False(restricted.CanUndo);
    }

    [Fact]
    public void No_selection_is_noop_missing_selection_fails_and_self_connection_is_not_created()
    {
        var doc = Chain(0.4, 2); string before = doc.ToJson(); Assert.Empty(doc.DeleteNodes([])); Assert.False(doc.CanUndo);
        Assert.Throws<ArgumentException>(() => doc.DeleteNodes([new("missing")])); Assert.Equal(before, doc.ToJson());
        doc.Connect(new("2"), 0, new("1"), 0); doc.DeleteNodes([new("2")]); Assert.Empty(doc.Links); Assert.Single(doc.Nodes);
    }

    [Fact]
    public void Reconnection_uses_source_to_destination_type_compatibility_and_keeps_all_consumers()
    {
        var root = Chain(1, 3).Snapshot(); root["nodes"]![0]!["outputs"]![0]!["type"] = "string,INT";
        root["nodes"]![2]!["inputs"]![0]!["type"] = "STRING";
        var doc = WorkflowDocument.Parse(root.ToJsonString()); var extra = doc.AddNode("Extension", template: new JsonObject {
            ["inputs"] = JsonNode.Parse("[{\"name\":\"in\",\"type\":\"STRING\",\"link\":null}]") });
        doc.Connect(new("2"), 0, extra, 0); doc.DeleteNodes([new("2")]);
        Assert.Equal(2, doc.Links.Count); Assert.All(doc.Links, l => Assert.Equal(new NodeId("1"), l.Source));
    }

    private static WorkflowDocument Chain(double version, int count)
    {
        var doc = WorkflowDocument.Parse(new JsonObject { ["version"] = version, ["nodes"] = new JsonArray(), ["links"] = new JsonArray() }.ToJsonString());
        for (int i = 0; i < count; i++) doc.AddNode("Extension", template: new JsonObject {
            ["inputs"] = JsonNode.Parse("[{\"name\":\"in\",\"type\":\"STRING\",\"link\":null}]"),
            ["outputs"] = JsonNode.Parse("[{\"name\":\"out\",\"type\":\"STRING\",\"links\":[]}]"), ["vendor"] = new JsonArray("keep", i) });
        for (int i = 1; i < count; i++) doc.Connect(new(i.ToString()), 0, new((i + 1).ToString()), 0);
        var root = doc.Snapshot(); if (version == 1) root["state"]!["lastLinkId"] = 100; else root["last_link_id"] = 100;
        return WorkflowDocument.Parse(root.ToJsonString());
    }
}
