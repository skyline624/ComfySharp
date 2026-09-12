using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class CaseConverterWorkflowTests
{
    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void ConnectedTextAndSavedModeSurviveRoundTripsAndUndo(double version)
    {
        var root = new JsonObject
        {
            ["version"] = version,
            ["nodes"] = new JsonArray(
                new JsonObject { ["id"] = 1, ["type"] = "PrimitiveString", ["widgets_values"] = new JsonArray("ǳABC"),
                    ["outputs"] = new JsonArray(new JsonObject { ["name"] = "STRING", ["type"] = "STRING", ["links"] = new JsonArray(1) }) },
                new JsonObject { ["id"] = 2, ["type"] = "CaseConverter", ["vendor"] = new JsonObject { ["keep"] = "yes" },
                    ["widgets_values"] = new JsonArray("saved", "Capitalize"),
                    ["inputs"] = new JsonArray(new JsonObject { ["name"] = "string", ["type"] = "STRING", ["link"] = 1 }),
                    ["outputs"] = new JsonArray() }),
            ["links"] = new JsonArray(new JsonArray(1, 1, 0, 2, 0, "STRING"))
        };
        var document = WorkflowDocument.Parse(root.ToJsonString());
        string before = document.ToJson();
        var reopened = WorkflowDocument.Parse(WorkflowDocument.Parse(before).ToJson());
        Assert.Equal(before, reopened.ToJson());
        var compiled = PromptCompiler.Compile(reopened);
        Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
        var inputs = compiled.Prompt!["2"]!["inputs"]!;
        Assert.Equal("Capitalize", inputs["mode"]!.GetValue<string>());
        Assert.True(JsonNode.DeepEquals(new JsonArray("1", 0), inputs["string"]));
        reopened.Disconnect(1);
        Assert.Equal("saved", PromptCompiler.Compile(reopened).Prompt!["2"]!["inputs"]!["string"]!.GetValue<string>());
        reopened.Undo();
        Assert.Equal(before, reopened.ToJson());
        Assert.True(JsonNode.DeepEquals(compiled.Prompt, PromptCompiler.Compile(reopened).Prompt));
    }

    [Fact]
    public void ObjectWidgetsRemainIntactWhenTheHostDoesNotProvideTheNode()
    {
        var widgets = new JsonObject { ["string"] = "AΣ", ["mode"] = "Title Case" };
        var document = WorkflowDocument.Parse(new JsonObject
        { ["version"] = 1, ["nodes"] = new JsonArray(new JsonObject { ["id"] = "case", ["type"] = "CaseConverter", ["widgets_values"] = widgets }), ["links"] = new JsonArray() }.ToJsonString());
        string before = document.ToJson();
        var supported = PromptCompiler.Compile(document);
        Assert.True(supported.Success);
        Assert.True(JsonNode.DeepEquals(widgets, supported.Prompt!["case"]!["inputs"]));
        var unavailable = PromptCompiler.Compile(document, availableNodes: new HashSet<string>());
        Assert.False(unavailable.Success);
        Assert.Contains(unavailable.Diagnostics, d => d.Code == "unavailable_node");
        Assert.Equal(before, document.ToJson());
    }
}
