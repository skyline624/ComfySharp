using System.Text.Json.Nodes;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class V3ComboProjectionTests
{
    [Fact]
    public void TypedV3ComboKeepsItsOptionsWhileLegacyProjectionRemainsAnArray()
    {
        var registry = BuiltInNodes.CreateRegistry();
        var before = registry.Nodes.Single(n => n.Schema.ClassType == "StringCompare").Schema.Inputs.Single(i => i.Name == "mode").Options!.ToJsonString();
        var info = registry.ToObjectInfo();
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse("""["COMBO",{"multiselect":false,"options":["Starts With","Ends With","Equal"]}]"""),
            info["StringCompare"]!["input"]!["required"]!["mode"]));
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse("""[["Both","Left","Right"],{}]"""),
            info["StringTrim"]!["input"]!["required"]!["mode"]));
        Assert.Equal(before, registry.Nodes.Single(n => n.Schema.ClassType == "StringCompare").Schema.Inputs.Single(i => i.Name == "mode").Options!.ToJsonString());
        Assert.True(JsonNode.DeepEquals(info, registry.ToObjectInfo()));
    }
}
