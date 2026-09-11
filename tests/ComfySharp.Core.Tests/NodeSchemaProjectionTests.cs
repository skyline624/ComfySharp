using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class NodeSchemaProjectionTests
{
    [Fact]
    public void Object_info_preserves_grouped_input_order_options_and_optional_metadata_as_snapshots()
    {
        var options = new JsonObject { ["default"] = 3, ["advanced"] = true, ["round"] = false };
        var aliases = new[] { "source alias" };
        var schema = new NodeSchema("TestSchema", "Display", "test",
            [new("first", "INT", Options: options), new("optional", "*", Required: false, Lazy: true),
             new("second", "COMBO", Options: new() { ["options"] = new JsonArray("a", "b"), ["default"] = "a" })],
            [new("SIGMAS"), new("INT", "count", IsList: true)], Description: "Source description", SearchAliases: aliases,
            PythonModule: "comfy_extras.source_module");
        var registry = new NodeRegistry(); registry.Register(new SchemaNode(schema));
        var info = registry.ToObjectInfo()["TestSchema"]!.AsObject();
        Assert.Equal(new[] { "first", "second" }, info["input_order"]!["required"]!.AsArray().Select(n => n!.GetValue<string>()));
        Assert.Equal("optional", Assert.Single(info["input_order"]!["optional"]!.AsArray())!.GetValue<string>());
        Assert.True(info["input"]!["optional"]!["optional"]![1]!["lazy"]!.GetValue<bool>());
        Assert.False(info["input"]!["required"]!["first"]![1]!["round"]!.GetValue<bool>());
        Assert.Equal("a", info["input"]!["required"]!["second"]![0]![0]!.GetValue<string>());
        Assert.Equal("SIGMAS", info["output_name"]![0]!.GetValue<string>());
        Assert.Equal("count", info["output_name"]![1]!.GetValue<string>());
        Assert.Equal("Source description", info["description"]!.GetValue<string>());
        Assert.Equal("comfy_extras.source_module", info["python_module"]!.GetValue<string>());
        options["default"] = 10; aliases[0] = "changed alias";
        Assert.Equal(3, info["input"]!["required"]!["first"]![1]!["default"]!.GetValue<int>());
        Assert.Equal("source alias", info["search_aliases"]![0]!.GetValue<string>());
        info["input"]!["required"]!["first"]![1]!["round"] = true;
        Assert.False(options["round"]!.GetValue<bool>());
    }

    private sealed class SchemaNode(NodeSchema schema) : IRuntimeNode
    {
        public NodeSchema Schema { get; } = schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) =>
            throw new InvalidOperationException("Schema projection does not execute nodes.");
    }
}
