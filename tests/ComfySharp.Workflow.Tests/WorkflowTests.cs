using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;
public class WorkflowTests
{
    [Theory]
    [InlineData("1")]
    [InlineData("1.0")]
    [InlineData("1e0")]
    public void NumericIdSpellingsResolveTheSameLinkAndPreserveSourceToken(string spelling)
    {
        var json = """{"version":1,"nodes":[{"id":SOURCE_ID,"type":"PrimitiveString","widgets_values":["hello"],"outputs":[{"name":"STRING","type":"STRING","links":[1]}]},{"id":2,"type":"StringLength","widgets_values":{},"inputs":[{"name":"string","type":"STRING","link":1}]}],"links":[{"id":1,"origin_id":1,"origin_slot":0,"target_id":2,"target_slot":0,"type":"STRING"}]}""".Replace("SOURCE_ID", spelling);
        var document = WorkflowDocument.Parse(json);
        Assert.Equal(new NodeId("1"), document.Nodes[0].Id);
        Assert.Equal(spelling, document.Snapshot()["nodes"]![0]!["id"]!.ToJsonString());
        var compiled = PromptCompiler.Compile(document);
        Assert.True(compiled.Success);
        Assert.NotNull(compiled.Prompt!["1"]);
        Assert.Equal("1", compiled.Prompt["2"]!["inputs"]!["string"]![0]!.GetValue<string>());
        document.Move(new("1"), 30, 40);
        Assert.Equal(spelling, document.Snapshot()["nodes"]![0]!["id"]!.ToJsonString());
    }
    [Theory]
    [InlineData("1", "1.0")]
    [InlineData("1.0", "1e0")]
    [InlineData("\"1\"", "1e0")]
    [InlineData("0", "-0.0")]
    public void CanonicalNumericDuplicateIdsAreRejected(string first, string second)
    {
        var json = """{"version":1,"nodes":[{"id":FIRST},{"id":SECOND}]}""".Replace("FIRST", first).Replace("SECOND", second);
        Assert.Throws<FormatException>(() => WorkflowDocument.Parse(json));
    }
    [Fact] public void NamedWidgetsRejectMissingRequiredValues()
    {
        var empty = WorkflowDocument.Parse("""{"version":1,"nodes":[{"id":1,"type":"PrimitiveString","widgets_values":{}}]}""");
        var result = PromptCompiler.Compile(empty);
        Assert.False(result.Success); Assert.Null(result.Prompt);
        Assert.Contains(result.Diagnostics, d => d.Code == "missing_widgets" && d.Message.Contains("value"));
        var incomplete = WorkflowDocument.Parse("""{"version":1,"nodes":[{"id":1,"type":"StringConcatenate","widgets_values":{"string_a":"a","string_b":"b"}}]}""");
        result = PromptCompiler.Compile(incomplete);
        Assert.Null(result.Prompt);
        Assert.Contains(result.Diagnostics, d => d.Code == "missing_widgets" && d.Message.Contains("delimiter"));
    }
    [Fact] public void NamedWidgetsMayOmitNonserializedClientControls()
    {
        var document = WorkflowDocument.Parse("""{"version":1,"nodes":[{"id":1,"type":"PrimitiveInt","widgets_values":{"value":42}}]}""");
        var result = PromptCompiler.Compile(document);
        Assert.True(result.Success);
        Assert.Equal(42, result.Prompt!["1"]!["inputs"]!["value"]!.GetValue<int>());
        Assert.Null(result.Prompt["1"]!["inputs"]!["control_after_generate"]);
    }
    private const string Legacy = """
    {"version":0.4,"last_node_id":2,"last_link_id":7,"vendor":{"opaque":[1,{"x":true}]},"nodes":[
    {"id":"custom:1","type":"ThirdParty","pos":{"0":10,"1":20,"extension":"retained"},"size":[200,90],"mode":0,"widgets_values":["secret"],"outputs":[{"name":"value","type":"IMAGE","links":[7]}],"vendorNode":{"x":1}},
    {"id":2,"type":"SaveImage","pos":[400,20],"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":7}],"widgets_values":["out"]}],"links":[[7,"custom:1",0,2,0,"IMAGE"]],"extra":{"viewport":{"x":6}}}
    """;
    [Fact] public void UnknownWorkflowDataAndIdTypesRoundTrip()
    {
        var doc = WorkflowDocument.Parse(Legacy);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(Legacy), JsonNode.Parse(doc.ToJson())));
        doc.Move(new("custom:1"), 44, 55);
        var data = doc.Snapshot(); Assert.Equal("retained", data["nodes"]![0]!["pos"]!["extension"]!.GetValue<string>());
        Assert.Equal("custom:1", data["nodes"]![0]!["id"]!.GetValue<string>());
        Assert.Equal(2, data["nodes"]![1]!["id"]!.GetValue<int>());
        doc.Undo(); Assert.True(JsonNode.DeepEquals(JsonNode.Parse(Legacy), doc.Snapshot()));
        Assert.False(doc.IsDirty); doc.Redo(); Assert.Equal(44, doc.Nodes[0].X); Assert.True(doc.IsDirty);
    }
    [Fact] public void VersionOneObjectLinksRetainExtensionFields()
    {
        var root = JsonNode.Parse(Legacy)!.AsObject(); root["version"] = 1;
        root["links"] = new JsonArray(new JsonObject { ["id"] = 7, ["origin_id"] = "custom:1", ["origin_slot"] = "0", ["target_id"] = 2, ["target_slot"] = 0, ["type"] = "IMAGE", ["parentId"] = 42, ["vendor"] = "kept" });
        root["definitions"] = new JsonObject { ["subgraphs"] = new JsonArray(new JsonObject { ["opaque"] = 9 }) };
        var doc = WorkflowDocument.Parse(root.ToJsonString());
        Assert.Equal(new NodeId("custom:1"), doc.Links[0].Source);
        Assert.True(JsonNode.DeepEquals(root, JsonNode.Parse(doc.ToJson())));
        Assert.Contains(PromptCompiler.Compile(doc).Diagnostics, d => d.Code == "unsupported_subgraphs");
    }
    [Fact] public void DeleteDisconnectAndUndoRestoreBothEndpoints()
    {
        var doc = WorkflowDocument.Parse(Legacy); doc.Delete(new("custom:1"));
        Assert.Single(doc.Nodes); Assert.Empty(doc.Links); Assert.Null(doc.Nodes[0].Data["inputs"]![0]!["link"]);
        doc.Undo(); Assert.Equal(2, doc.Nodes.Count); Assert.Single(doc.Links);
        doc.Disconnect(7); Assert.Empty(doc.Links); doc.Connect(new("custom:1"), 0, new("2"), 0);
        Assert.Single(doc.Links); Assert.Equal(doc.Links[0].Id, doc.Nodes[1].Data["inputs"]![0]!["link"]!.GetValue<long>());
    }
    [Fact] public void FailedConnectionIsAtomic()
    {
        var doc = WorkflowDocument.Parse(Legacy); var before = doc.ToJson();
        Assert.ThrowsAny<Exception>(() => doc.Connect(new("custom:1"), 100, new("2"), 0)); Assert.Equal(before, doc.ToJson()); Assert.False(doc.CanUndo);
    }
    [Fact] public void UnknownAndBypassedNodesNeverProducePartialPrompt()
    {
        var doc = WorkflowDocument.Parse(Legacy); var result = PromptCompiler.Compile(doc);
        Assert.False(result.Success); Assert.Null(result.Prompt); Assert.Contains(result.Diagnostics, d => d.Code == "unsupported_node");
        var root = JsonNode.Parse(Legacy)!; root["nodes"]![1]!["mode"] = 4;
        Assert.Contains(PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString())).Diagnostics, d => d.Code == "unsupported_mode");
    }
    [Fact] public void ExplicitWidgetsOmitClientControlsAndWrapArrayLiterals()
    {
        var doc = WorkflowDocument.Parse("""{"version":0.4,"nodes":[{"id":8,"type":"KSampler","widgets_values":[42,"randomize",20,7.5,"euler","normal",1]}],"links":[]} """);
        var result = PromptCompiler.Compile(doc); Assert.True(result.Success);
        Assert.Equal(42, result.Prompt!["8"]!["inputs"]!["seed"]!.GetValue<int>());
        Assert.Null(result.Prompt["8"]!["inputs"]!["control_after_generate"]);
        var unavailable = PromptCompiler.Compile(doc, availableNodes: new HashSet<string>()); Assert.Contains(unavailable.Diagnostics, d => d.Code == "unavailable_node");
        doc.SetWidgets(new("8"), new JsonArray(42)); Assert.Contains(PromptCompiler.Compile(doc).Diagnostics, d => d.Code == "widget_layout");
        var arrayDoc = WorkflowDocument.Parse("""{"version":1,"nodes":[{"id":"x","type":"Array","widgets_values":[[1,2]]}]}""");
        var definitions = new Dictionary<string, NodeDefinition> { ["Array"] = new("Array", [new("values")]) };
        Assert.IsType<JsonArray>(PromptCompiler.Compile(arrayDoc, definitions).Prompt!["x"]!["inputs"]!["values"]!["__value__"]);
    }
    [Fact] public void CompilerUsesStringLinkIdsAndRejectsDanglingLinks()
    {
        var definitions = new Dictionary<string, NodeDefinition>(PromptCompiler.BaseDefinitions) { ["ThirdParty"] = new("ThirdParty", [new("value")]) };
        var doc = WorkflowDocument.Parse(Legacy); var result = PromptCompiler.Compile(doc, definitions);
        Assert.True(result.Success); Assert.Equal("custom:1", result.Prompt!["2"]!["inputs"]!["images"]![0]!.GetValue<string>());
        var root = JsonNode.Parse(Legacy)!; root["links"]![0]![1] = "missing";
        Assert.Contains(PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString()), definitions).Diagnostics, d => d.Code == "invalid_link");
    }
    [Fact] public void CompilerRejectsCyclesAndUnreferencedLinks()
    {
        var doc = WorkflowDocument.Parse("""{"version":1,"nodes":[{"id":1,"type":"Pass","inputs":[{"name":"x","type":"*","link":1}],"outputs":[{"name":"x","type":"*","links":[1]}]}],"links":[{"id":1,"origin_id":1,"origin_slot":0,"target_id":1,"target_slot":0,"type":"*"}]}""");
        var definitions = new Dictionary<string, NodeDefinition> { ["Pass"] = new("Pass", []) };
        Assert.Contains(PromptCompiler.Compile(doc, definitions).Diagnostics, d => d.Code == "cycle");
        var root = doc.Snapshot(); root["nodes"]![0]!["inputs"]![0]!["link"] = null;
        Assert.Contains(PromptCompiler.Compile(WorkflowDocument.Parse(root.ToJsonString()), definitions).Diagnostics, d => d.Code == "inconsistent_link");
    }
}
