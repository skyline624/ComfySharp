using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class UnsignedSeedTests
{
    [Theory]
    [InlineData("0", 0UL)]
    [InlineData("9223372036854775808", 9223372036854775808UL)]
    [InlineData("18446744073709551615", ulong.MaxValue)]
    [InlineData("\"18446744073709551615\"", ulong.MaxValue)]
    [InlineData("true", 1UL)]
    [InlineData("1.9", 1UL)]
    public async Task Unsigned_schema_validates_and_executes_exact_seed_values(string literal, ulong expected)
    {
        var registry = new NodeRegistry(); registry.Register(new SeedEcho()); using var engine = new EngineService(registry);
        var graph = JsonNode.Parse("{\"1\":{\"class_type\":\"SeedEcho\",\"inputs\":{\"seed\":" + literal + "}}}")!.AsObject();
        Assert.True(engine.Validate(graph, ["1"]).IsValid);
        var result = await engine.ExecuteAsync(graph, ["1"]); Assert.Equal("success", result.Status);
        Assert.Equal(expected, result.Outputs["1"][0][0]!.GetValue<ulong>());
    }
    [Theory]
    [InlineData("-1")]
    [InlineData("18446744073709551616")]
    [InlineData("\"18446744073709551616\"")]
    public void Out_of_range_unsigned_seeds_are_rejected_before_execution(string literal)
    {
        var registry = new NodeRegistry(); registry.Register(new SeedEcho()); using var engine = new EngineService(registry);
        var graph = JsonNode.Parse("{\"1\":{\"class_type\":\"SeedEcho\",\"inputs\":{\"seed\":" + literal + "}}}")!.AsObject();
        var validation = engine.Validate(graph, ["1"]); Assert.False(validation.IsValid);
        Assert.Contains(validation.Diagnostics, d => d.Code == "invalid_input_type");
    }
    private sealed class SeedEcho : IRuntimeNode
    {
        public NodeSchema Schema { get; } = new("SeedEcho", "SeedEcho", "tests",
            [new("seed", "INT", Options: new() { ["min"] = 0, ["max"] = ulong.MaxValue })], [new("INT")]);
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken token)
            => ValueTask.FromResult(new NodeExecutionOutput([inputs["seed"]]));
    }
}
