using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class ImageFileWorkflowTests
{
    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void Save_output_connects_to_preview_and_preserves_filename_widget_and_unknown_fields(double version)
    {
        var document = WorkflowDocument.Parse(new JsonObject
        {
            ["version"] = version, ["links"] = new JsonArray(),
            ["nodes"] = new JsonArray(
                Node(1, "EmptyImage", new(2, 3, 2, 0x336699), false),
                Node(2, "SaveImage", new("folder/image_%batch_num%"), true),
                Node(3, "PreviewImage", new(), true)),
            ["extension"] = new JsonObject { ["retain"] = new JsonArray(1, "é", null) }
        }.ToJsonString());
        document.Connect(new("1"), 0, new("2"), 0); document.Connect(new("2"), 0, new("3"), 0);
        string saved = document.ToJson();
        var reopened = WorkflowDocument.Parse(WorkflowDocument.Parse(saved).ToJson()); Assert.Equal(saved, reopened.ToJson());
        var compiled = PromptCompiler.Compile(reopened); Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
        Assert.Equal("folder/image_%batch_num%", compiled.Prompt!["2"]!["inputs"]!["filename_prefix"]!.GetValue<string>());
        Assert.True(JsonNode.DeepEquals(new JsonArray("1", 0), compiled.Prompt["2"]!["inputs"]!["images"]));
        Assert.True(JsonNode.DeepEquals(new JsonObject { ["images"] = new JsonArray("2", 0) }, compiled.Prompt["3"]!["inputs"]));
        Assert.False(compiled.Prompt["2"]!["inputs"]!.AsObject().ContainsKey("prompt"));
        document.Undo(); Assert.Single(document.Links); document.Redo(); Assert.Equal(saved, document.ToJson());
    }

    private static JsonObject Node(int id, string type, JsonArray widgets, bool imageInput) => new()
    {
        ["id"] = id, ["type"] = type, ["widgets_values"] = widgets,
        ["inputs"] = imageInput ? new JsonArray(new JsonObject { ["name"] = "images", ["type"] = "IMAGE", ["link"] = null }) : new JsonArray(),
        ["outputs"] = new JsonArray(new JsonObject { ["name"] = "images", ["type"] = "IMAGE", ["links"] = new JsonArray() })
    };
}
