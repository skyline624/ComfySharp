using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class WorkflowClipboardTests
{
    [Theory]
    [InlineData(0.4, 0.4)]
    [InlineData(0.4, 1)]
    [InlineData(1, 0.4)]
    [InlineData(1, 1)]
    public void Selection_moves_between_documents_with_fresh_ids_ports_position_and_atomic_undo(double sourceVersion, double targetVersion)
    {
        var source = Graph(sourceVersion); source.MarkSaved(); string original = source.ToJson();
        var text = source.CopyNodes([new("source:01"), new("sink")]);
        Assert.Equal(original, source.ToJson()); Assert.False(source.IsDirty); Assert.False(source.CanUndo);
        var destination = Graph(targetVersion); string before = destination.ToJson(); int changes = 0;
        destination.Changed += (_, _) => changes++;
        var pasted = destination.PasteNodes(text, 700, 800); Assert.Equal(2, pasted.Count); Assert.Equal(1, changes);
        var copies = destination.Nodes.Where(n => pasted.Values.Contains(n.Id)).ToArray();
        Assert.Equal(700, copies.Min(n => n.X)); Assert.Equal(800, copies.Min(n => n.Y));
        Assert.Equal(50, copies[1].X - copies[0].X); Assert.Equal(30, copies[1].Y - copies[0].Y);
        Assert.Equal("101", copies[0].Id.Value); Assert.Equal("102", copies[1].Id.Value);
        Assert.Equal("extension payload", copies[1].Data["properties"]!["vendor"]!.GetValue<string>());
        Assert.Equal(2, destination.Links.Count); var link = destination.Links.Single(l => l.Target == copies[1].Id);
        Assert.Equal(201, link.Id); Assert.Equal(copies[0].Id, link.Source);
        Assert.Equal(201, copies[1].Data["inputs"]![0]!["link"]!.GetValue<long>());
        Assert.Equal(201, Assert.Single(copies[0].Data["outputs"]![0]!["links"]!.AsArray())!.GetValue<long>());
        var raw = destination.Snapshot()["links"]!.AsArray().Last(); Assert.True(targetVersion == 1 ? raw is JsonObject : raw is JsonArray);
        string after = destination.ToJson(); destination.Undo(); Assert.Equal(before, destination.ToJson()); destination.Redo(); Assert.Equal(after, destination.ToJson());
        Assert.True(JsonNode.DeepEquals(destination.Snapshot(), WorkflowDocument.Parse(after).Snapshot()));
        destination.SetWidgets(copies[0].Id, new JsonObject { ["value"] = "changed" });
        Assert.Equal(original, source.ToJson());
        Assert.Equal("initial", destination.Nodes[0].Data["widgets_values"]!["value"]!.GetValue<string>());
    }

    [Fact]
    public void Clipboard_contains_only_selected_node_data_and_cannot_connect_to_coincidental_destination_ids()
    {
        var source = Graph(0.4); string before = source.ToJson();
        var text = source.CopyNodes([new("sink")]); Assert.DoesNotContain("private document setting", text);
        Assert.DoesNotContain("initial", text); Assert.DoesNotContain("source:01", text);
        var destination = Graph(0.4); var pasted = destination.PasteNodes(text);
        var copy = destination.Nodes.Single(n => n.Id == pasted.Values.Single());
        Assert.Null(copy.Data["inputs"]![0]!["link"]); Assert.Single(destination.Links);
        Assert.Equal(before, source.ToJson());
    }

    [Theory]
    [InlineData(0.4)]
    [InlineData(1)]
    public void Unknown_link_fields_survive_same_version_transfer_and_cross_version_refuses_loss(double version)
    {
        var root = Graph(version).Snapshot(); var raw = root["links"]![0]!;
        if (raw is JsonArray array) array.Add(JsonNode.Parse("{\"vendor\":[1,null,3]}"));
        else raw["vendor"] = JsonNode.Parse("[1,null,3]");
        var source = WorkflowDocument.Parse(root.ToJsonString()); string text = source.CopyNodes(source.Nodes.Select(n => n.Id));
        var same = Graph(version); same.PasteNodes(text);
        var copiedRaw = same.Snapshot()["links"]!.AsArray().Last()!;
        Assert.True(JsonNode.DeepEquals(version == 1 ? raw["vendor"] : raw[6], version == 1 ? copiedRaw["vendor"] : copiedRaw[6]));
        var different = Graph(version == 1 ? 0.4 : 1); string before = different.ToJson();
        Assert.Throws<NotSupportedException>(() => different.PasteNodes(text)); Assert.Equal(before, different.ToJson()); Assert.False(different.CanUndo);
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("not json")]
    [InlineData("{\"format\":\"ComfySharp.nodes\",\"version\":2}")]
    [InlineData("{\"format\":\"ComfySharp.nodes\",\"version\":1,\"workflow\":{\"version\":0.4,\"nodes\":[],\"links\":[]}}")]
    public void Unsupported_clipboard_never_changes_destination(string text)
    {
        var doc = Graph(1); string before = doc.ToJson(); var error = Record.Exception(() => doc.PasteNodes(text));
        Assert.True(error is FormatException or JsonException); Assert.Equal(before, doc.ToJson()); Assert.False(doc.CanUndo);
    }

    [Fact]
    public void Paste_failure_preserves_existing_undo_redo_and_emits_no_change()
    {
        var source = Graph(1); var text = JsonNode.Parse(source.CopyNodes(source.Nodes.Select(n => n.Id)))!;
        text["workflow"]!["nodes"]![1]!["inputs"]![0]!["link"] = 666;
        var doc = Graph(1); doc.Rename(new("sink"), "renamed"); doc.Undo(); string before = doc.ToJson(); int changed = 0; doc.Changed += (_, _) => changed++;
        Assert.Throws<FormatException>(() => doc.PasteNodes(text.ToJsonString())); Assert.Equal(before, doc.ToJson());
        Assert.True(doc.CanRedo); Assert.False(doc.CanUndo); Assert.Equal(0, changed);
        doc.Redo(); Assert.Equal("renamed", doc.Nodes[1].Title);
    }

    [Fact]
    public void Repeated_paste_is_independent_and_never_reuses_destination_ids()
    {
        var source = Graph(0.4); var text = source.CopyNodes(source.Nodes.Select(n => n.Id)); var destination = WorkflowDocument.Create();
        var first = destination.PasteNodes(text); var second = destination.PasteNodes(text);
        Assert.Empty(first.Values.Intersect(second.Values)); Assert.Equal(2, destination.Links.Count);
        destination.Undo(); Assert.Equal(2, destination.Nodes.Count); destination.Undo(); Assert.Empty(destination.Nodes);
    }

    [Fact]
    public void Empty_selection_oversize_text_and_nonfinite_coordinates_are_rejected_before_editing()
    {
        var source = Graph(0.4); string before = source.ToJson();
        Assert.Throws<InvalidOperationException>(() => source.CopyNodes([]));
        Assert.Throws<FormatException>(() => source.PasteNodes(new string(' ', 16 * 1024 * 1024 + 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.PasteNodes("{}", double.PositiveInfinity));
        Assert.Equal(before, source.ToJson()); Assert.False(source.CanUndo);
    }

    private static WorkflowDocument Graph(double version)
    {
        var root = JsonNode.Parse("""
            {"version":0.4,"last_node_id":100,"last_link_id":200,"extra":{"user":"private document setting"},"nodes":[
             {"id":"source:01","type":"PrimitiveString","pos":[30,40],"widgets_values":{"value":"initial"},"outputs":[{"name":"out","type":"STRING","links":[5]}]},
             {"id":"sink","type":"UnknownExtension","pos":[80,70],"properties":{"vendor":"extension payload"},"inputs":[{"name":"source","type":"STRING","link":5}]}],
             "links":[[5,"source:01",0,"sink",0,"STRING"]]}
            """)!;
        if (version == 1)
        {
            root["version"] = 1; root["state"] = JsonNode.Parse("{\"lastNodeId\":100,\"lastLinkId\":200}");
            root["links"] = JsonNode.Parse("""[{"id":5,"origin_id":"source:01","origin_slot":0,"target_id":"sink","target_slot":0,"type":"STRING"}]""");
        }
        return WorkflowDocument.Parse(root.ToJsonString());
    }
}
