using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class TextComparisonWorkflowTests
{
    [Theory]
    [InlineData("StringContains", 0.4)]
    [InlineData("StringContains", 1.0)]
    [InlineData("StringCompare", 0.4)]
    [InlineData("StringCompare", 1.0)]
    public void WidgetsAndConnectedSensitivitySurviveRoundTripsAndUndo(string type, double version)
    {
        bool contains = type == "StringContains";
        string first = contains ? "string" : "string_a", second = contains ? "substring" : "string_b";
        var root = new JsonObject
        {
            ["version"] = version,
            ["nodes"] = new JsonArray(
                new JsonObject { ["id"] = 1, ["type"] = "PrimitiveBoolean", ["widgets_values"] = new JsonArray(false),
                    ["outputs"] = new JsonArray(new JsonObject { ["name"] = "BOOLEAN", ["type"] = "BOOLEAN", ["links"] = new JsonArray(1) }) },
                new JsonObject { ["id"] = 2, ["type"] = type, ["vendor"] = new JsonObject { ["keep"] = "yes" },
                    ["widgets_values"] = contains ? new JsonArray("ΟΣ", "ος", true) : new JsonArray("ΟΣ", "ος", "Equal", true),
                    ["inputs"] = new JsonArray(new JsonObject { ["name"] = "case_sensitive", ["type"] = "BOOLEAN", ["link"] = 1 }),
                    ["outputs"] = new JsonArray() }),
            ["links"] = new JsonArray(new JsonArray(1, 1, 0, 2, 0, "BOOLEAN"))
        };
        var document = WorkflowDocument.Parse(root.ToJsonString());
        string before = document.ToJson();
        var reopened = WorkflowDocument.Parse(WorkflowDocument.Parse(before).ToJson());
        Assert.Equal(before, reopened.ToJson());
        var compiled = PromptCompiler.Compile(reopened);
        Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
        var inputs = compiled.Prompt!["2"]!["inputs"]!;
        Assert.Equal("ΟΣ", inputs[first]!.GetValue<string>());
        Assert.Equal("ος", inputs[second]!.GetValue<string>());
        if (!contains) Assert.Equal("Equal", inputs["mode"]!.GetValue<string>());
        Assert.True(JsonNode.DeepEquals(new JsonArray("1", 0), inputs["case_sensitive"]));
        reopened.Disconnect(1);
        Assert.True(PromptCompiler.Compile(reopened).Prompt!["2"]!["inputs"]!["case_sensitive"]!.GetValue<bool>());
        reopened.Undo();
        Assert.Equal(before, reopened.ToJson());
        Assert.True(JsonNode.DeepEquals(compiled.Prompt, PromptCompiler.Compile(reopened).Prompt));
    }

    [Theory]
    [InlineData("StringContains")]
    [InlineData("StringCompare")]
    public void ObjectWidgetsPreserveNamesAndUnavailableNodesStayInTheDocument(string type)
    {
        bool contains = type == "StringContains";
        var widgets = contains ? new JsonObject { ["string"] = "", ["substring"] = "", ["case_sensitive"] = false }
            : new JsonObject { ["string_a"] = "", ["string_b"] = "", ["mode"] = "Ends With", ["case_sensitive"] = false };
        var document = WorkflowDocument.Parse(new JsonObject
        { ["version"] = 1, ["nodes"] = new JsonArray(new JsonObject { ["id"] = "test", ["type"] = type, ["widgets_values"] = widgets }), ["links"] = new JsonArray() }.ToJsonString());
        string before = document.ToJson();
        var supported = PromptCompiler.Compile(document);
        Assert.True(supported.Success);
        Assert.True(JsonNode.DeepEquals(widgets, supported.Prompt!["test"]!["inputs"]));
        var unavailable = PromptCompiler.Compile(document, availableNodes: new HashSet<string>());
        Assert.False(unavailable.Success);
        Assert.Contains(unavailable.Diagnostics, d => d.Code == "unavailable_node");
        Assert.Equal(before, document.ToJson());
    }
}
