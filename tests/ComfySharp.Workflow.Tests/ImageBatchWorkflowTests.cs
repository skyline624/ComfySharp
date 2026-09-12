using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class ImageBatchWorkflowTests
{
    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void TwoImagesCompileThroughBatchAndSurviveRoundTrip(double version)
    {
        var document = Create(version, ["image1", "image2"]);
        document.Connect(new("1"), 0, new("3"), 0);
        document.Connect(new("2"), 0, new("3"), 1);
        document.Connect(new("3"), 0, new("4"), 0);
        string saved = document.ToJson();
        var reopened = WorkflowDocument.Parse(WorkflowDocument.Parse(saved).ToJson());
        Assert.Equal(saved, reopened.ToJson());
        var result = PromptCompiler.Compile(reopened);
        Assert.True(result.Success, string.Join("; ", result.Diagnostics));
        Assert.Equal("ImageBatch", result.Prompt!["3"]!["class_type"]!.GetValue<string>());
        Assert.Equal(new[] { "image1", "image2" }, result.Prompt["3"]!["inputs"]!.AsObject().Select(p => p.Key));
        Assert.True(JsonNode.DeepEquals(new JsonArray("1", 0), result.Prompt["3"]!["inputs"]!["image1"]));
        Assert.True(JsonNode.DeepEquals(new JsonArray("2", 0), result.Prompt["3"]!["inputs"]!["image2"]));
        Assert.True(JsonNode.DeepEquals(new JsonArray("3", 0), result.Prompt["4"]!["inputs"]!["image"]));
        Assert.Equal(0xff0000, result.Prompt["1"]!["inputs"]!["color"]!.GetValue<int>());
        Assert.Equal(3, result.Prompt["2"]!["inputs"]!["height"]!.GetValue<int>());
        Assert.Empty(reopened.Nodes.Single(n => n.Id.Value == "3").Data["widgets_values"]!.AsArray());
        Assert.Equal("preserved", reopened.Nodes.Single(n => n.Id.Value == "3").Data["extension"]!.GetValue<string>());
    }

    [Fact]
    public void ImportedPortOrderAndConnectionReplacementRemainUndoable()
    {
        var document = Create(1.0, ["image2", "image1"]);
        document.Connect(new("1"), 0, new("3"), 0);
        document.Connect(new("2"), 0, new("3"), 1);
        string before = document.ToJson();
        var initial = PromptCompiler.Compile(document);
        Assert.True(initial.Success, string.Join("; ", initial.Diagnostics));
        Assert.True(JsonNode.DeepEquals(new JsonArray("1", 0), initial.Prompt!["3"]!["inputs"]!["image2"]));
        Assert.True(JsonNode.DeepEquals(new JsonArray("2", 0), initial.Prompt["3"]!["inputs"]!["image1"]));

        document.Connect(new("2"), 0, new("3"), 0);
        string replaced = document.ToJson();
        Assert.Equal(2, document.Links.Count);
        Assert.True(JsonNode.DeepEquals(new JsonArray("2", 0), PromptCompiler.Compile(document).Prompt!["3"]!["inputs"]!["image2"]));
        document.Undo();
        Assert.Equal(before, document.ToJson());
        document.Redo();
        Assert.Equal(replaced, WorkflowDocument.Parse(document.ToJson()).ToJson());
        Assert.Equal(new[] { "image2", "image1" }, document.Nodes.Single(n => n.Id.Value == "3").Data["inputs"]!.AsArray().Select(p => p!["name"]!.GetValue<string>()));
    }

    private static WorkflowDocument Create(double version, string[] batchInputs)
    {
        var batch = Node(3, "ImageBatch", new(), batchInputs);
        batch["extension"] = "preserved";
        return WorkflowDocument.Parse(new JsonObject
        {
            ["version"] = version,
            ["nodes"] = new JsonArray(
                Node(1, "EmptyImage", new(2, 1, 1, 0xff0000), []),
                Node(2, "EmptyImage", new(1, 3, 2, 0x0000ff), []),
                batch,
                Node(4, "ImageInvert", new(), ["image"])),
            ["links"] = new JsonArray()
        }.ToJsonString());
    }

    private static JsonObject Node(int id, string type, JsonArray widgets, string[] inputNames) => new()
    {
        ["id"] = id, ["type"] = type, ["widgets_values"] = widgets,
        ["inputs"] = new JsonArray(inputNames.Select(name => (JsonNode)new JsonObject
        { ["name"] = name, ["type"] = "IMAGE", ["link"] = null }).ToArray()),
        ["outputs"] = new JsonArray(new JsonObject { ["name"] = "IMAGE", ["type"] = "IMAGE", ["links"] = new JsonArray() })
    };
}
