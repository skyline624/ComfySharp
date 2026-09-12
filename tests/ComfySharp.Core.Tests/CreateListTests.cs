using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class CreateListTests
{
    private static JsonObject Prompt(string json) => JsonNode.Parse(json)!.AsObject();
    private static RuntimeNode Source(string name, Func<RuntimeNodeContext, RuntimeValue> produce, bool list = false) =>
        new(new(name, name, "test", [], [new("*", IsList: list)]), (c, _, _) => ValueTask.FromResult(new NodeExecutionOutput([produce(c)])));

    [Fact]
    public async Task RealGraphConcatenatesByTemplateOrderThenMapsDownstream()
    {
        var engine = new EngineService(BuiltInNodes.CreateRegistry());
        var result = await engine.ExecuteAsync(Prompt("""
            {"a":{"class_type":"PrimitiveString","inputs":{"value":"alpha"}},
             "b":{"class_type":"PrimitiveString","inputs":{"value":"🌍"}},
             "list":{"class_type":"CreateList","inputs":{"inputs.input2":["b",0],"inputs.input0":["a",0]}},
             "length":{"class_type":"StringLength","inputs":{"string":["list",0]}}}
            """), ["list", "length"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        Assert.Equal(new[] { "alpha", "🌍" }, result.Outputs["list"][0].Select(v => v!.GetValue<string>()));
        Assert.Equal(new[] { 5, 1 }, result.Outputs["length"][0].Select(v => v!.GetValue<int>()));
    }

    [Fact]
    public async Task ConcatenatesExecutionListsOnceAndPreservesLiteralContainers()
    {
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(Source("Two", c => c.List([c.Json(new JsonArray()), c.Json(new JsonArray(1, 2))]), true));
        registry.Register(Source("Empty", c => c.List([]), true));
        registry.Register(Source("One", c => c.Json(new JsonObject { ["x"] = 3 })));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"two":{"class_type":"Two","inputs":{}},"empty":{"class_type":"Empty","inputs":{}},"one":{"class_type":"One","inputs":{}},
             "list":{"class_type":"CreateList","inputs":{"inputs.input3":["one",0],"inputs.input1":["empty",0],"inputs.input0":["two",0]}}}
            """), ["list"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(new[] { "[]", "[1,2]", "{\"x\":3}" }, result.Outputs["list"][0].Select(v => v.ToJson()!.ToJsonString()));
    }

    [Fact]
    public async Task EmptyExecutionListIsAValidRequiredConnection()
    {
        var registry = BuiltInNodes.CreateRegistry(); registry.Register(Source("Empty", c => c.List([]), true));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Empty","inputs":{}},"list":{"class_type":"CreateList","inputs":{"inputs.input0":["p",0]}}}
            """), ["list"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Outputs["list"][0]);
    }

    [Theory]
    [InlineData("inputs.input10")]
    [InlineData("inputs.input01")]
    [InlineData("inputs")]
    public async Task ExtrasAreRejectedBeforeAnyProducerRuns(string extra)
    {
        int calls = 0; var registry = BuiltInNodes.CreateRegistry();
        registry.Register(Source("Producer", c => { calls++; return c.Json(null); }));
        var prompt = Prompt("""{"p":{"class_type":"Producer","inputs":{}},"list":{"class_type":"CreateList","inputs":{"inputs.input0":["p",0]}}}""");
        prompt["list"]!["inputs"]![extra] = new JsonArray("p", 0);
        using var result = await new EngineService(registry).ExecuteValuesAsync(prompt, ["list"]);
        Assert.Equal("error", result.Status); Assert.Equal(0, calls);
        Assert.Contains(result.Diagnostics, d => d.Code == "unsupported_dynamic_input" && d.InputName == extra && d.NodeId == "list");
    }

    [Fact]
    public void AConnectedOptionalPortDoesNotReplaceRequiredPortZero()
    {
        var validation = new EngineService(BuiltInNodes.CreateRegistry()).Validate(Prompt("""
            {"p":{"class_type":"PrimitiveInt","inputs":{"value":1}},"list":{"class_type":"CreateList","inputs":{"inputs.input5":["p",0]}}}
            """), ["list"]);
        Assert.False(validation.IsValid);
        Assert.Contains(validation.Diagnostics, d => d.Code == "required_input_missing" && d.InputName == "inputs.input0");
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("prompt-first")]
    public async Task BlockerSelectionPrecedesGroupingAndUsesPromptOrder(string? message)
    {
        int calls = 0; var create = new CreateListNode(); var registry = new NodeRegistry();
        registry.Register(new RuntimeNode(create.Schema, (c, i, token) => { calls++; return create.ExecuteAsync(c, i, token); }));
        registry.Register(Source("Zero", c => c.Blocker("schema-first")));
        registry.Register(Source("Two", c => c.List([c.Json(JsonValue.Create(1)), c.Blocker(message)]), true));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"zero":{"class_type":"Zero","inputs":{}},"two":{"class_type":"Two","inputs":{}},
             "list":{"class_type":"CreateList","inputs":{"inputs.input2":["two",0],"inputs.input0":["zero",0]}}}
            """), ["list", "zero"]);
        Assert.Equal(0, calls); Assert.Null(result.Outputs["list"][0][0].Blocker.Message);
        Assert.Equal("schema-first", result.Outputs["zero"][0][0].Blocker.Message);
        if (message is null) Assert.Empty(result.Diagnostics);
        else Assert.Equal("Execution Blocked: " + message, Assert.Single(result.Diagnostics).Message);
    }

    [Fact]
    public async Task BlockersInsideContainerItemsAreNotRecursivelyConsumed()
    {
        int calls = 0; var create = new CreateListNode(); var registry = new NodeRegistry();
        registry.Register(new RuntimeNode(create.Schema, (c, i, token) => { calls++; return create.ExecuteAsync(c, i, token); }));
        registry.Register(Source("Container", c => c.Map(new Dictionary<string, RuntimeValue> { ["inside"] = c.Blocker("kept") })));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Container","inputs":{}},"list":{"class_type":"CreateList","inputs":{"inputs.input0":["p",0]}}}
            """), ["list"]);
        Assert.Equal(1, calls); Assert.Empty(result.Diagnostics);
        Assert.Equal("kept", result.Outputs["list"][0][0].Properties["inside"].Blocker.Message);
    }

    [Fact]
    public async Task FanoutAndDuplicateItemsShareTheProducerAndSurviveUntilLastOutputOwner()
    {
        int calls = 0; var resource = new CountingResource(); var registry = BuiltInNodes.CreateRegistry();
        registry.Register(Source("Owned", c => { calls++; return c.Own(resource); }));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Owned","inputs":{}},
             "a":{"class_type":"CreateList","inputs":{"inputs.input0":["p",0],"inputs.input1":["p",0]}},
             "b":{"class_type":"CreateList","inputs":{"inputs.input0":["a",0]}}}
            """), ["a", "b"]);
        Assert.Equal("success", result.Status); Assert.Equal(1, calls);
        Assert.Equal(2, result.Outputs["b"][0].Count);
        using var keeper = result.Outputs["b"][0][1].Retain();
        result.Dispose(); Assert.Equal(0, resource.Disposals); Assert.Same(resource, keeper.GetNative<CountingResource>());
        keeper.Dispose(); Assert.Equal(1, resource.Disposals);
    }

    [Fact]
    public async Task CancellationAfterInputsAreResolvedReleasesResourcesAndPublishesNoList()
    {
        var resource = new CountingResource(); var registry = BuiltInNodes.CreateRegistry();
        registry.Register(Source("Owned", c => c.Own(resource))); using var cancellation = new CancellationTokenSource();
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Owned","inputs":{}},"list":{"class_type":"CreateList","inputs":{"inputs.input0":["p",0]}}}
            """), ["list"], e =>
            {
                if (e.Type == "executing" && e.NodeId == "list") cancellation.Cancel();
                return ValueTask.CompletedTask;
            }, cancellation.Token);
        Assert.Equal("cancelled", result.Status); Assert.Empty(result.Outputs); Assert.Equal(1, resource.Disposals);
    }

    [Fact]
    public async Task PartialConcatenationFailureReleasesClonedLeasesWithoutStealingInputs()
    {
        var resource = new CountingResource(); using var caller = new RuntimeNodeContext(); using var invocation = new RuntimeNodeContext();
        var list = caller.List([caller.Own(resource), caller.Json(null)]);
        var map = caller.Map(new Dictionary<string, RuntimeValue> { ["input0"] = list });
        map.Properties["input0"].Items[1].Dispose();
        var node = new CreateListNode();
        await Assert.ThrowsAsync<ObjectDisposedException>(async () => await node.ExecuteAsync(invocation,
            new Dictionary<string, RuntimeValue> { ["inputs"] = map }, CancellationToken.None));
        invocation.Dispose(); Assert.Equal(0, resource.Disposals);
        Assert.Same(resource, list.Items[0].GetNative<CountingResource>());
        caller.Dispose(); Assert.Equal(1, resource.Disposals);
    }

    [Fact]
    public async Task DirectCallWithUnfinalizedShapeOrCancelledTokenFailsExplicitly()
    {
        using var context = new RuntimeNodeContext(); var node = new CreateListNode();
        await Assert.ThrowsAsync<ArgumentException>(async () => await node.ExecuteAsync(context,
            new Dictionary<string, RuntimeValue> { ["inputs"] = context.Json(new JsonObject()) }, CancellationToken.None));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await node.ExecuteAsync(context,
            new Dictionary<string, RuntimeValue>(), cancelled.Token));
    }

    private sealed class CountingResource : IDisposable { public int Disposals { get; private set; } public void Dispose() => Disposals++; }
    private sealed class RuntimeNode(NodeSchema schema,
        Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, CancellationToken, ValueTask<NodeExecutionOutput>> run) : IRuntimeNode
    {
        public NodeSchema Schema { get; } = schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) => run(context, inputs, cancellationToken);
    }
}
