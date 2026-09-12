using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class ImagePrimitiveWorkflowTests
{
    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void ImageChainCompilesAndRoundTripsWithConnectedIntegerOverride(double version)
    {
        var document = WorkflowDocument.Parse(new JsonObject
        {
            ["version"] = version,
            ["nodes"] = new JsonArray(
                Node(1, "EmptyImage", new(2, 3, 1, 0x017ffe), ["width"], "IMAGE"),
                Node(2, "RepeatImageBatch", new(3), ["image", "amount"], "IMAGE"),
                Node(3, "ImageFromBatch", new(-1, 1), ["image", "batch_index", "length"], "IMAGE"),
                Node(4, "ImageInvert", new(), ["image"], "IMAGE"),
                Node(5, "PrimitiveInt", new(1, "fixed"), [], "INT")),
            ["links"] = new JsonArray()
        }.ToJsonString());
        document.Connect(new("1"), 0, new("2"), 0);
        document.Connect(new("2"), 0, new("3"), 0);
        document.Connect(new("3"), 0, new("4"), 0);
        document.Connect(new("5"), 0, new("1"), 0);
        string saved = document.ToJson();
        var reopened = WorkflowDocument.Parse(WorkflowDocument.Parse(saved).ToJson());
        Assert.Equal(saved, reopened.ToJson());
        var compiled = PromptCompiler.Compile(reopened);
        Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
        Assert.True(JsonNode.DeepEquals(new JsonArray("5", 0), compiled.Prompt!["1"]!["inputs"]!["width"]));
        Assert.Equal(3, compiled.Prompt["1"]!["inputs"]!["height"]!.GetValue<int>());
        Assert.Equal(0x017ffe, compiled.Prompt["1"]!["inputs"]!["color"]!.GetValue<int>());
        Assert.Equal(-1, compiled.Prompt["3"]!["inputs"]!["batch_index"]!.GetValue<int>());
        Assert.Equal("image", Assert.Single(compiled.Prompt["4"]!["inputs"]!.AsObject()).Key);
        reopened.Disconnect(reopened.Links.Single(l => l.Source.Value == "5").Id);
        Assert.Equal(2, PromptCompiler.Compile(reopened).Prompt!["1"]!["inputs"]!["width"]!.GetValue<int>());
        reopened.Undo(); Assert.Equal(saved, reopened.ToJson());
    }

    [Fact]
    public void UnavailableImageNodesAndUnknownPropertiesRemainInTheDocument()
    {
        var node = Node(1, "ImageFromBatch", new(-3, 2), ["image"], "IMAGE");
        node["extension_payload"] = new JsonObject { ["retain"] = new JsonArray(1, "x") };
        var document = WorkflowDocument.Parse(new JsonObject { ["version"] = 1, ["nodes"] = new JsonArray(node), ["links"] = new JsonArray() }.ToJsonString());
        string before = document.ToJson();
        var result = PromptCompiler.Compile(document, availableNodes: new HashSet<string>());
        Assert.False(result.Success); Assert.Contains(result.Diagnostics, d => d.Code == "unavailable_node");
        Assert.Equal(before, WorkflowDocument.Parse(document.ToJson()).ToJson());
    }

    private static JsonObject Node(int id, string type, JsonArray widgets, string[] names, string outputType) => new()
    {
        ["id"] = id, ["type"] = type, ["widgets_values"] = widgets,
        ["inputs"] = new JsonArray(names.Select(name => (JsonNode)new JsonObject
        { ["name"] = name, ["type"] = name == "image" ? "IMAGE" : "INT", ["link"] = null }).ToArray()),
        ["outputs"] = new JsonArray(new JsonObject { ["name"] = outputType, ["type"] = outputType, ["links"] = new JsonArray() })
    };
}
