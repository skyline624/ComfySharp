using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class StringFormatWorkflowTests
{
    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void SparseImportedNamesAndUnknownFieldsSurviveCompilationAndTwoRoundTrips(double version)
    {
        var document = Sparse(version);
        string before = document.ToJson();
        var reopened = WorkflowDocument.Parse(WorkflowDocument.Parse(before).ToJson());
        var result = PromptCompiler.Compile(reopened);
        Assert.True(result.Success, string.Join("; ", result.Diagnostics));
        var inputs = result.Prompt!["3"]!["inputs"]!.AsObject();
        Assert.Equal(new[] { "f_string", "values.z" }, inputs.Select(p => p.Key));
        Assert.Equal("[{z}]", inputs["f_string"]!.GetValue<string>());
        Assert.True(JsonNode.DeepEquals(new JsonArray("1", 0), inputs["values.z"]));
        Assert.False(inputs.ContainsKey("values.a"));
        Assert.False(inputs.ContainsKey("values"));
        Assert.Equal("f_string", Assert.Single(PromptCompiler.BaseDefinitions["StringFormat"].Widgets).Name);
        Assert.Equal(before, reopened.ToJson());
        var node = reopened.Nodes.Single(n => n.Id == new NodeId("3"));
        Assert.Equal(new[] { "values.z", "f_string", "values.a" },
            node.Data["inputs"]!.AsArray().Select(p => p!["name"]!.GetValue<string>()));
        Assert.Equal("kept", node.Data["vendor"]!["note"]!.GetValue<string>());
        reopened.Move(new("3"), 20, 30); reopened.Undo();
        Assert.Equal(before, reopened.ToJson());
    }

    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void ConnectedFormatOverridesItsWidgetAndUndoRestoresTheConnection(double version)
    {
        var document = Sparse(version, formatLink: true);
        var connected = PromptCompiler.Compile(document);
        Assert.True(connected.Success);
        Assert.True(JsonNode.DeepEquals(new JsonArray("2", 0), connected.Prompt!["3"]!["inputs"]!["f_string"]));
        document.Disconnect(2);
        var disconnected = PromptCompiler.Compile(document);
        Assert.True(disconnected.Success);
        Assert.Equal("[{z}]", disconnected.Prompt!["3"]!["inputs"]!["f_string"]!.GetValue<string>());
        document.Undo();
        Assert.True(JsonNode.DeepEquals(connected.Prompt, PromptCompiler.Compile(document).Prompt));
    }

    [Theory]
    [InlineData("values")]
    [InlineData("values.A")]
    [InlineData("values.aa")]
    [InlineData("values.a.b")]
    [InlineData("F_string")]
    public void UnavailableInputNamesProduceDiagnosticsAndPreserveTheDocument(string name)
    {
        var root = Sparse(1).Snapshot();
        root["nodes"]![2]!["inputs"]![0]!["name"] = name;
        var document = WorkflowDocument.Parse(root.ToJsonString());
        string before = document.ToJson();
        var compiled = PromptCompiler.Compile(document);
        Assert.False(compiled.Success); Assert.Null(compiled.Prompt);
        Assert.Contains(compiled.Diagnostics, d => d.Code == "invalid_input_name" && d.Node == new NodeId("3"));
        Assert.Equal(before, document.ToJson()); Assert.Single(document.Links);
    }

    [Theory]
    [InlineData("values.z")]
    [InlineData("f_string")]
    public void DuplicateNamesAreRejectedEvenWhenUnconnected(string name)
    {
        var root = Sparse(1).Snapshot();
        root["nodes"]![2]!["inputs"]![2]!["name"] = name;
        var compiled = PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString()));
        Assert.False(compiled.Success);
        Assert.Contains(compiled.Diagnostics, d => d.Code == "duplicate_input_name");
    }

    [Fact]
    public void ConstantNeedsNoValueSlotsButRequiresItsWidgetOrAFormatConnection()
    {
        var document = WorkflowDocument.Parse("""
            {"version":1,"nodes":[{"id":1,"type":"StringFormat","widgets_values":["constant"]}],"links":[]}
            """);
        var compiled = PromptCompiler.Compile(document);
        Assert.True(compiled.Success); Assert.Empty(compiled.Diagnostics);
        var inputs = compiled.Prompt!["1"]!["inputs"]!.AsObject();
        Assert.Single(inputs); Assert.Equal("constant", inputs["f_string"]!.GetValue<string>());
        var missing = document.Snapshot(); missing["nodes"]![0]!.AsObject().Remove("widgets_values");
        Assert.Contains(PromptCompiler.Compile(WorkflowDocument.Parse(missing.ToJsonString())).Diagnostics,
            d => d.Code == "missing_widgets");
    }

    [Theory]
    [InlineData("bad")]
    [InlineData(1.5)]
    public void InvalidLinkIdsAreDiagnosedWithoutThrowing(object invalid)
    {
        var root = Sparse(1).Snapshot();
        root["nodes"]![2]!["inputs"]![0]!["link"] = invalid is string s ? JsonValue.Create(s) : JsonValue.Create((double)invalid);
        var compiled = PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString()));
        Assert.False(compiled.Success);
        Assert.Contains(compiled.Diagnostics, d => d.Code == "invalid_link");
        Assert.Contains(compiled.Diagnostics, d => d.Code == "inconsistent_link");
    }

    private static WorkflowDocument Sparse(double version, bool formatLink = false)
    {
        var root = JsonNode.Parse("""
            {"version":1,"nodes":[
            {"id":1,"type":"PrimitiveString","widgets_values":["value"],"outputs":[{"name":"STRING","type":"STRING","links":[1]}]},
            {"id":2,"type":"PrimitiveString","widgets_values":["{z:>8}"],"outputs":[{"name":"STRING","type":"STRING","links":[]}]},
            {"id":3,"type":"StringFormat","vendor":{"note":"kept"},"widgets_values":["[{z}]"],
             "inputs":[{"name":"values.z","type":"*","link":1},{"name":"f_string","type":"STRING","link":null},{"name":"values.a","type":"*","link":null}],
             "outputs":[{"name":"STRING","type":"STRING","links":[]}]}],"links":[]}
            """)!.AsObject();
        root["version"] = version;
        var links = root["links"]!.AsArray();
        void Link(int id, int origin, int slot) => links.Add(version == 1
            ? new JsonObject { ["id"] = id, ["origin_id"] = origin, ["origin_slot"] = 0, ["target_id"] = 3, ["target_slot"] = slot, ["type"] = "STRING" }
            : new JsonArray(id, origin, 0, 3, slot, "STRING"));
        Link(1, 1, 0);
        if (formatLink)
        {
            root["nodes"]![2]!["inputs"]![1]!["link"] = 2;
            root["nodes"]![1]!["outputs"]![0]!["links"]!.AsArray().Add(2);
            Link(2, 2, 1);
        }
        return WorkflowDocument.Parse(root.ToJsonString());
    }
}
