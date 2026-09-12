using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class CreateListWorkflowTests
{
    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void ImportedSparsePortsKeepTheirIndicesOrderMetadataAndFlatPrompt(double version)
    {
        var document = Sparse(version);
        string before = document.ToJson();
        var reopened = WorkflowDocument.Parse(WorkflowDocument.Parse(before).ToJson());
        var result = PromptCompiler.Compile(reopened);
        Assert.True(result.Success, string.Join("; ", result.Diagnostics));
        var inputs = result.Prompt!["3"]!["inputs"]!.AsObject();
        Assert.Equal(new[] { "inputs.input2", "inputs.input0" }, inputs.Select(p => p.Key));
        Assert.True(JsonNode.DeepEquals(new JsonArray("2", 0), inputs["inputs.input2"]));
        Assert.True(JsonNode.DeepEquals(new JsonArray("1", 0), inputs["inputs.input0"]));
        Assert.False(inputs.ContainsKey("inputs"));
        Assert.False(inputs.ContainsKey("inputs.input1"));
        Assert.Empty(PromptCompiler.BaseDefinitions["CreateList"].Widgets);
        Assert.Equal(before, reopened.ToJson());
        var imported = reopened.Nodes.Single(n => n.Id == new NodeId("3"));
        Assert.Equal(3, imported.Data["inputs"]!.AsArray().Count);
        Assert.Equal("kept", imported.Data["vendor"]!["note"]!.GetValue<string>());
        Assert.Null(imported.Data["widgets_values"]); // No manufactured widget field on import.
        Assert.Equal(new[] { 1, 0 }, reopened.Links.Select(l => l.TargetSlot));
        reopened.Move(new("3"), 10, 20);
        reopened.Undo();
        Assert.Equal(before, reopened.ToJson());
        reopened.Redo();
        Assert.Equal(3, reopened.Nodes.Single(n => n.Id == new NodeId("3")).Data["inputs"]!.AsArray().Count);
    }

    [Theory]
    [InlineData("inputs.input10")]
    [InlineData("inputs.input01")]
    [InlineData("inputs")]
    [InlineData("inputs.Input2")]
    [InlineData("other.input2")]
    public void InvalidDynamicNamesProduceDiagnosticsWithoutDiscardingConnections(string name)
    {
        var root = Sparse(1).Snapshot();
        root["nodes"]![2]!["inputs"]![0]!["name"] = name;
        var document = WorkflowDocument.Parse(root.ToJsonString());
        string before = document.ToJson();
        var result = PromptCompiler.Compile(document);
        Assert.False(result.Success);
        Assert.Null(result.Prompt);
        Assert.Contains(result.Diagnostics, d => d.Code == "invalid_input_name" && d.Node == new NodeId("3"));
        Assert.Equal(before, document.ToJson());
        Assert.Equal(2, document.Links.Count);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MissingOrNonStringPortNamesAreDiagnosedInsteadOfThrowing(bool numeric)
    {
        var root = Sparse(1).Snapshot();
        var port = root["nodes"]![2]!["inputs"]![0]!.AsObject();
        if (numeric) port["name"] = 2; else port.Remove("name");
        var result = PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString()));
        Assert.Null(result.Prompt);
        Assert.Contains(result.Diagnostics, d => d.Code == "invalid_input_name");
    }

    [Fact]
    public void DuplicateNamesAreRejectedEvenWhenTheDuplicateSlotIsUnconnected()
    {
        var root = Sparse(1).Snapshot();
        root["nodes"]![2]!["inputs"]![2]!["name"] = "inputs.input0";
        var result = PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString()));
        Assert.Null(result.Prompt);
        Assert.Contains(result.Diagnostics, d => d.Code == "duplicate_input_name" && d.Node == new NodeId("3"));
    }

    [Fact]
    public void OptionalConnectionCannotReplaceRequiredInputZero()
    {
        var document = Sparse(1);
        document.Disconnect(1);
        var result = PromptCompiler.Compile(document);
        Assert.Null(result.Prompt);
        Assert.Contains(result.Diagnostics, d => d.Code == "missing_input" && d.Message.Contains("inputs.input0"));
        Assert.Single(document.Links);
        Assert.Equal(0, document.Links[0].TargetSlot);
    }

    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void WrongTargetSlotIsNotRepairedOrMatchedByPortName(double version)
    {
        var root = Sparse(version).Snapshot();
        if (version == 1) root["links"]![0]!["target_slot"] = 2;
        else root["links"]![0]![4] = 2;
        var document = WorkflowDocument.Parse(root.ToJsonString());
        string before = document.ToJson();
        var result = PromptCompiler.Compile(document);
        Assert.Null(result.Prompt);
        Assert.Contains(result.Diagnostics, d => d.Code == "invalid_link" && d.Node == new NodeId("3"));
        Assert.Contains(result.Diagnostics, d => d.Code == "inconsistent_link");
        Assert.Equal(before, document.ToJson());
    }

    [Fact]
    public void CreateListHasNoPositionalOrNamedWidgetFallback()
    {
        foreach (JsonNode widgets in new JsonNode[] { new JsonArray("not an input"), new JsonObject { ["inputs.input0"] = "not a link" } })
        {
            var root = Sparse(1).Snapshot();
            root["nodes"]![2]!["widgets_values"] = widgets;
            var result = PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString()));
            Assert.Null(result.Prompt);
            Assert.Contains(result.Diagnostics, d => d.Code is "widget_layout" or "unknown_widget");
        }
    }

    private static WorkflowDocument Sparse(double version)
    {
        var root = JsonNode.Parse("""
            {"version":1,"nodes":[
              {"id":1,"type":"PrimitiveString","widgets_values":["alpha"],"outputs":[{"name":"STRING","type":"STRING","links":[1]}]},
              {"id":2,"type":"PrimitiveString","widgets_values":["🌍"],"outputs":[{"name":"STRING","type":"STRING","links":[2]}]},
              {"id":3,"type":"CreateList","vendor":{"note":"kept"},"inputs":[
                {"name":"inputs.input2","type":"*","link":2,"vendorPort":9},
                {"name":"inputs.input0","type":"*","link":1},
                {"name":"inputs.input9","type":"*","link":null}],
                "outputs":[{"name":"list","type":"*","links":[],"is_list":true}]}],
             "links":[
               {"id":1,"origin_id":1,"origin_slot":0,"target_id":3,"target_slot":1,"type":"STRING","vendorLink":true},
               {"id":2,"origin_id":2,"origin_slot":0,"target_id":3,"target_slot":0,"type":"STRING"}]}
            """)!.AsObject();
        root["version"] = version;
        if (version == 0.4) root["links"] = new JsonArray(new JsonArray(1, 1, 0, 3, 1, "STRING"), new JsonArray(2, 2, 0, 3, 0, "STRING"));
        return WorkflowDocument.Parse(root.ToJsonString());
    }
}
