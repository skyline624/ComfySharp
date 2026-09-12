using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class PromptLiteralTests
{
    // Explicit test-only frontend binding for the real node's wildcard source input.
    // The product PreviewAny definition retains its existing zero-widget layout.
    private static readonly IReadOnlyDictionary<string, NodeDefinition> Definitions =
        new Dictionary<string, NodeDefinition>(PromptCompiler.BaseDefinitions)
        { ["PreviewAny"] = new("PreviewAny", [new("source")]) };

    [Theory]
    [InlineData("null", "null")]
    [InlineData("true", "true")]
    [InlineData("12", "12")]
    [InlineData("1.25", "1.25")]
    [InlineData("\"text\"", "\"text\"")]
    [InlineData("{}", "{}")]
    [InlineData("{\"x\":[1,2]}", "{\"x\":[1,2]}")]
    [InlineData("{\"nested\":{\"__value__\":1}}", "{\"nested\":{\"__value__\":1}}")]
    [InlineData("{\"__VALUE__\":1}", "{\"__VALUE__\":1}")]
    [InlineData("[]", "{\"__value__\":[]}")]
    [InlineData("[1,2]", "{\"__value__\":[1,2]}")]
    [InlineData("[\"absent\",0]", "{\"__value__\":[\"absent\",0]}")]
    [InlineData("{\"__value__\":null}", "{\"__value__\":null}")]
    [InlineData("{\"__value__\":1,\"sibling\":2}", "{\"__value__\":1,\"sibling\":2}")]
    [InlineData("{\"__value__\":{\"__value__\":[1]},\"sibling\":false}", "{\"__value__\":{\"__value__\":[1]},\"sibling\":false}")]
    public void RealCompilerWrapsOnlyArraysAndTransmitsObjectsAcrossFormatsAndSerialization(string literal, string expected)
    {
        foreach (double version in new[] { 0.4, 1.0 })
        foreach (bool named in new[] { false, true })
        {
            var document = Document(literal, version, named);
            string before = document.ToJson();
            var result = PromptCompiler.Compile(document, Definitions);
            Assert.True(result.Success, string.Join("; ", result.Diagnostics));
            var inputs = result.Prompt!["1"]!["inputs"]!.AsObject();
            Assert.True(inputs.ContainsKey("source")); // Explicit null is not an omitted widget.
            Equal(JsonNode.Parse(expected), inputs["source"]);
            Assert.Equal(before, document.ToJson());
            var reopened = WorkflowDocument.Parse(before);
            var second = PromptCompiler.Compile(reopened, Definitions);
            Assert.True(second.Success, string.Join("; ", second.Diagnostics));
            Equal(result.Prompt, second.Prompt);
            var persisted = reopened.Nodes[0].Data["widgets_values"]!;
            Equal(JsonNode.Parse(literal), named ? persisted["source"] : persisted[0]);
        }
    }

    [Fact]
    public void ExplicitNestedEnvelopeIsClonedWithoutAddingAnAutomaticDictionaryWrapper()
    {
        const string widget = "{\"__value__\":{\"__value__\":[1]},\"sibling\":2}";
        var document = Document(widget, 1, true);
        string before = document.ToJson();
        var first = PromptCompiler.Compile(document, Definitions);
        Assert.True(first.Success);
        first.Prompt!["1"]!["inputs"]!["source"]!["__value__"]!["__value__"]!.AsArray().Add(9);
        Assert.Equal(before, document.ToJson());
        var second = PromptCompiler.Compile(document, Definitions);
        Assert.True(second.Success);
        Equal(JsonNode.Parse(widget), second.Prompt!["1"]!["inputs"]!["source"]);
        Assert.False(JsonNode.DeepEquals(first.Prompt, second.Prompt));
    }

    [Theory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public void RealConnectionOverridesReservedLiteralAndUndoRestoresTheWidget(double version)
    {
        const string widget = "{\"__value__\":1,\"sibling\":2}";
        var root = Document(widget, version, false).Snapshot();
        root["nodes"]!.AsArray().Add(JsonNode.Parse("""
            {"id":2,"type":"PrimitiveString","widgets_values":["connected"],
             "outputs":[{"name":"STRING","type":"STRING","links":[]}]}
            """));
        var document = WorkflowDocument.Parse(root.ToJsonString());
        document.Connect(new("2"), 0, new("1"), 0);
        string connected = document.ToJson();
        var result = PromptCompiler.Compile(document, Definitions);
        Assert.True(result.Success, string.Join("; ", result.Diagnostics));
        Equal(new JsonArray("2", 0), result.Prompt!["1"]!["inputs"]!["source"]);
        Assert.IsType<JsonArray>(result.Prompt["1"]!["inputs"]!["source"]);
        Assert.Equal(connected, document.ToJson());
        document.Undo();
        var unlinked = PromptCompiler.Compile(document, Definitions);
        Assert.True(unlinked.Success);
        Equal(JsonNode.Parse(widget), unlinked.Prompt!["1"]!["inputs"]!["source"]);
        document.Redo();
        Equal(result.Prompt, PromptCompiler.Compile(WorkflowDocument.Parse(document.ToJson()), Definitions).Prompt);
    }

    private static WorkflowDocument Document(string literal, double version, bool named)
    {
        var root = JsonNode.Parse("""
            {"version":1,"nodes":[{"id":1,"type":"PreviewAny",
              "inputs":[{"name":"source","type":"*","link":null}],"outputs":[]}],"links":[]}
            """)!.AsObject();
        root["version"] = version;
        root["nodes"]![0]!["widgets_values"] = named
            ? new JsonObject { ["source"] = JsonNode.Parse(literal) } : new JsonArray(JsonNode.Parse(literal));
        return WorkflowDocument.Parse(root.ToJsonString());
    }
    private static void Equal(JsonNode? expected, JsonNode? actual) =>
        Assert.True(JsonNode.DeepEquals(expected, actual), $"Expected {expected}; actual {actual}");
}
