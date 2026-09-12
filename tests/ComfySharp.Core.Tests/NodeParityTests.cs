using System.Text.Json.Nodes;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>Expected behavior transcribed from pinned upstream source; these do not run a Python oracle.</summary>
public sealed class NodeParityTests
{
    [Theory]
    [InlineData("StringLength", "{\"string\":\"A😀é\"}", "3")]
    [InlineData("StringSubstring", "{\"string\":\"A😀éB\",\"start\":1,\"end\":-1}", "\"😀é\"")]
    [InlineData("StringSubstring", "{\"string\":\"abcd\",\"start\":3,\"end\":1}", "\"\"")]
    [InlineData("StringReplace", "{\"string\":\"😀x\",\"find\":\"\",\"replace\":\"-\"}", "\"-😀-x-\"")]
    [InlineData("StringReplace", "{\"string\":\"\",\"find\":\"\",\"replace\":\"-\"}", "\"-\"")]
    [InlineData("StringTrim", "{\"string\":\"\\u001c a \\u001f\",\"mode\":\"Both\"}", "\"a\"")]
    [InlineData("PrimitiveBoolean", "{\"value\":\"false\"}", "true")]
    [InlineData("PrimitiveBoolean", "{\"value\":{\"__value__\":[]}}", "false")]
    [InlineData("PrimitiveInt", "{\"value\":-3.9}", "-3")]
    [InlineData("PrimitiveFloat", "{\"value\":\"1.25\"}", "1.25")]
    [InlineData("PrimitiveString", "{\"value\":null}", "\"None\"")]
    [InlineData("PrimitiveStringMultiline", "{\"value\":\"a\\nb\"}", "\"a\\nb\"")]
    [InlineData("ComfyNotNode", "{\"value\":{\"__value__\":[]}}", "true")]
    [InlineData("ComfyNotNode", "{\"value\":\"false\"}", "false")]
    [InlineData("JsonExtractString", "{\"json_string\":\"{\\\"a\\\":true}\",\"key\":\"a\"}", "\"True\"")]
    [InlineData("JsonExtractString", "{\"json_string\":\"{\\\"a\\\":null}\",\"key\":\"a\"}", "\"\"")]
    [InlineData("JsonExtractString", "{\"json_string\":\"malformed\",\"key\":\"a\"}", "\"\"")]
    public async Task Ported_nodes_match_reference_examples(string type, string inputs, string expected)
    {
        var prompt = new JsonObject { ["1"] = new JsonObject { ["class_type"] = type, ["inputs"] = JsonNode.Parse(inputs) } };
        var result = await new EngineService(BuiltInNodes.CreateRegistry()).ExecuteAsync(prompt, ["1"]);
        Assert.Equal("success", result.Status);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(expected), result.Outputs["1"][0][0]));
    }

    [Fact]
    public void Catalog_has_upstream_ids_ports_combo_and_lazy_metadata()
    {
        var catalog = BuiltInNodes.CreateRegistry().ToObjectInfo();
        Assert.Equal(18, catalog.Count);
        Assert.Equal("STRING", catalog["StringConcatenate"]!["input"]!["required"]!["string_a"]![0]!.GetValue<string>());
        Assert.Equal("Both", catalog["StringTrim"]!["input"]!["required"]!["mode"]![0]![0]!.GetValue<string>());
        Assert.True(catalog["ComfySwitchNode"]!["input"]!["optional"]!["on_true"]![1]!["lazy"]!.GetValue<bool>());
        Assert.Equal("COMFY_MATCHTYPE_V3", catalog["ComfySwitchNode"]!["output"]![0]!.GetValue<string>());
        Assert.Equal("switch", catalog["ComfySwitchNode"]!["output_matchtypes"]![0]!.GetValue<string>());
    }

    [Theory]
    [InlineData("{\"a\":1,\"a\":2}", "2")]
    [InlineData("{\"a\":1,\"a\":null}", "")]
    [InlineData("{\"a\":{\"x\":1,\"y\":3,\"x\":2}}", "{'x': 2, 'y': 3}")]
    [InlineData("{\"a\":[{\"x\":1,\"x\":2},{\"b\":false,\"b\":true}]}", "[{'x': 2}, {'b': True}]")]
    public async Task Json_extraction_uses_last_duplicate_key_at_every_depth(string json, string expected)
    {
        var prompt = new JsonObject { ["1"] = new JsonObject
        {
            ["class_type"] = "JsonExtractString",
            ["inputs"] = new JsonObject { ["json_string"] = json, ["key"] = "a" }
        } };
        var result = await new EngineService(BuiltInNodes.CreateRegistry()).ExecuteAsync(prompt, ["1"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(expected, result.Outputs["1"][0][0]!.GetValue<string>());
    }
}
