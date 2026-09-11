using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>Managed behavioral and ownership checks. These are not an executed upstream oracle.</summary>
public sealed class ExecutionBlockerTests
{
    private static JsonObject Prompt(string text) => JsonNode.Parse(text)!.AsObject();
    private static NodeExecutionOutput Out(params RuntimeValue[] values) => new(values);
    private static InputSchema Input(string name, bool lazy = false) => new(name, "*", Lazy: lazy);
    private static TestNode Node(string name, Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, NodeExecutionOutput> run,
        InputSchema[]? inputs = null, OutputSchema[]? outputs = null, bool inputIsList = false,
        Func<IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>>, IReadOnlyCollection<string>>? lazy = null) =>
        new(new(name, name, "test", inputs ?? [], outputs ?? [new("*")], InputIsList: inputIsList),
            (c, i, _) => ValueTask.FromResult(run(c, i)), lazy);

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("reason")]
    public void BlockersAreDistinctFromNullAndRetainIndependentLeases(string? message)
    {
        using var context = new RuntimeNodeContext();
        var value = context.Blocker(message);
        var jsonNull = context.Json(null);
        Assert.Equal(RuntimeValueKind.Blocker, value.Kind);
        Assert.Equal(RuntimeValueKind.Json, jsonNull.Kind);
        Assert.Null(jsonNull.ToJson());
        Assert.Equal(message, value.Blocker.Message);
        Assert.Throws<RuntimeValueProjectionException>(() => value.ToJson());
        using var retained = value.Retain();
        Assert.NotSame(value, retained);
        context.Dispose();
        Assert.Throws<ObjectDisposedException>(() => value.Blocker);
        Assert.Equal(message, retained.Blocker.Message);
        retained.Dispose();
        retained.Dispose();
        Assert.Throws<ObjectDisposedException>(() => retained.Retain());
    }

    [Fact]
    public void ContainersWithBlockersAndResourcesRetainAndReleaseEveryLease()
    {
        var resource = new CountingResource();
        using var context = new RuntimeNodeContext();
        var native = context.Own(resource);
        var block = context.Blocker("kept");
        var map = context.Map(new Dictionary<string, RuntimeValue> { ["native"] = native, ["block"] = block });
        var list = context.List([map, map]);
        using var retained = list.Retain();
        context.Dispose();
        retained.Items[0].Dispose();
        Assert.Equal("kept", retained.Items[1].Properties["block"].Blocker.Message);
        Assert.Same(resource, retained.Items[1].Properties["native"].GetNative<CountingResource>());
        Assert.Equal(0, resource.DisposeCount);
        retained.Dispose();
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task SilentChainSkipsJsonAdaptersAndPreservesIndependentTargetsAndMemo()
    {
        int produced = 0, consumed = 0;
        var registry = new NodeRegistry();
        registry.Register(Node("Block", (c, _) => { produced++; return Out(c.Blocker()); }));
        registry.Register(new JsonConsumer(() => consumed++));
        registry.Register(Node("Ok", (c, _) => Out(c.Json(JsonValue.Create(42)))));
        var events = new List<EngineEvent>();
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Block","inputs":{}},
             "a":{"class_type":"JsonConsumer","inputs":{"value":["p",0]}},
             "b":{"class_type":"JsonConsumer","inputs":{"value":["a",0]}},
             "c":{"class_type":"JsonConsumer","inputs":{"value":["p",0]}},
             "ok":{"class_type":"Ok","inputs":{}}}
            """), ["b", "c", "ok", "p"], e => { events.Add(e); return ValueTask.CompletedTask; });
        Assert.Equal("success", result.Status);
        Assert.Equal(1, produced);
        Assert.Equal(0, consumed);
        Assert.Empty(result.Diagnostics);
        Assert.Equal(42, result.Outputs["ok"][0][0].ToJson()!.GetValue<int>());
        Assert.DoesNotContain(events, e => e.Type == "execution_error");
        Assert.Equal("execution_success", events[^1].Type);
        var first = result.Outputs["b"][0][0];
        using var kept = first.Retain();
        first.Dispose();
        Assert.Null(result.Outputs["c"][0][0].Blocker.Message);
        result.Dispose();
        Assert.Null(kept.Blocker.Message);
    }

    [Theory]
    [InlineData("")]
    [InlineData("stop")]
    public async Task DiagnosticMessagesAreConsumedAtEachFirstConsumerWithoutMutatingTheProducer(string message)
    {
        var registry = new NodeRegistry();
        registry.Register(Node("Block", (c, _) => Out(c.Blocker(message))));
        registry.Register(Node("Never", (_, _) => throw new InvalidOperationException("must not run"), [Input("value")]));
        var events = new List<EngineEvent>();
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Block","inputs":{}},
             "a":{"class_type":"Never","inputs":{"value":["p",0]}},
             "b":{"class_type":"Never","inputs":{"value":["a",0]}},
             "c":{"class_type":"Never","inputs":{"value":["p",0]}}}
            """), ["b", "c", "p"], e => { events.Add(e); return ValueTask.CompletedTask; });
        Assert.Equal("success", result.Status);
        Assert.Equal(new[] { "a", "c" }, result.Diagnostics.Select(d => d.NodeId));
        Assert.All(result.Diagnostics, d => { Assert.Equal("execution_blocked", d.Code); Assert.Equal("Execution Blocked: " + message, d.Message); });
        Assert.Equal(new[] { "a", "c" }, events.Where(e => e.Type == "execution_error").Select(e => e.NodeId));
        Assert.All(events.Where(e => e.Type == "execution_error"), e => Assert.Equal(e.NodeId, e.Identity!.NodeId));
        Assert.Null(result.Outputs["b"][0][0].Blocker.Message);
        Assert.Null(result.Outputs["c"][0][0].Blocker.Message);
        Assert.Equal(message, result.Outputs["p"][0][0].Blocker.Message);
        Assert.Equal("execution_success", events[^1].Type);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("first")]
    public async Task FirstBlockerUsesPromptInputOrderIncludingSilentAndEmptyMessages(string? firstMessage)
    {
        var registry = new NodeRegistry();
        registry.Register(Node("Later", (c, _) => Out(c.Blocker("later"))));
        registry.Register(Node("First", (c, _) => Out(c.Blocker(firstMessage))));
        registry.Register(Node("Never", (_, _) => throw new InvalidOperationException(), [Input("schemaFirst"), Input("promptFirst")]));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"later":{"class_type":"Later","inputs":{}},"first":{"class_type":"First","inputs":{}},
             "out":{"class_type":"Never","inputs":{"promptFirst":["first",0],"schemaFirst":["later",0]}}}
            """), ["out"]);
        Assert.Equal("success", result.Status);
        if (firstMessage is null) Assert.Empty(result.Diagnostics);
        else Assert.Equal("Execution Blocked: " + firstMessage, Assert.Single(result.Diagnostics).Message);
        Assert.Null(result.Outputs["out"][0][0].Blocker.Message);
    }

    [Fact]
    public async Task ListOutputPreservesOneBlockedPositionAndOnlyRealInvocationsProduceUi()
    {
        int listCalls = 0;
        var seen = new List<int>();
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (c, _) => Out(c.List([c.Json(JsonValue.Create(1)), c.Blocker(), c.Json(JsonValue.Create(3))])), outputs: [new("*", IsList: true)]));
        registry.Register(Node("DoubleList", (c, i) => { listCalls++; return Out(c.List([i["value"], i["value"]])); }, [Input("value")], [new("*", IsList: true)]));
        registry.Register(Node("Output", (_, i) =>
        {
            int n = i["value"].ToJson()!.GetValue<int>(); seen.Add(n);
            return new([i["value"]], new JsonObject { ["values"] = new JsonArray(n) });
        }, [Input("value")]));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Source","inputs":{}},"list":{"class_type":"DoubleList","inputs":{"value":["p",0]}},
             "out":{"class_type":"Output","inputs":{"value":["list",0]}}}
            """), ["out", "list"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(2, listCalls);
        Assert.Equal(new[] { 1, 1, 3, 3 }, seen);
        var values = result.Outputs["list"][0];
        Assert.Equal(5, values.Count);
        Assert.Equal(RuntimeValueKind.Blocker, values[2].Kind);
        Assert.Equal("[1,1,3,3]", result.UiOutputs["out"]["values"]!.ToJsonString());
    }

    [Fact]
    public async Task RepeatLastBlockerAndPartialSlotsDoNotSuppressAnUnrelatedSlot()
    {
        int consumerCalls = 0;
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (c, _) => Out(c.List([c.Json(JsonValue.Create(5)), c.Blocker()]),
            c.List([c.Json(JsonValue.Create(10)), c.Json(JsonValue.Create(20)), c.Json(JsonValue.Create(30)), c.Json(JsonValue.Create(40))])),
            outputs: [new("*", IsList: true), new("*", IsList: true)]));
        registry.Register(Node("Combine", (_, i) => { consumerCalls++; return Out(i["other"]); }, [Input("value"), Input("other")]));
        registry.Register(Node("Echo", (_, i) => Out(i["value"]), [Input("value")]));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Source","inputs":{}},
             "mapped":{"class_type":"Combine","inputs":{"value":["p",0],"other":["p",1]}},
             "sibling":{"class_type":"Echo","inputs":{"value":["p",1]}}}
            """), ["mapped", "sibling"]);
        Assert.Equal(1, consumerCalls);
        Assert.Equal(4, result.Outputs["mapped"][0].Count);
        Assert.All(result.Outputs["mapped"][0].Skip(1), v => Assert.Equal(RuntimeValueKind.Blocker, v.Kind));
        Assert.Equal(new[] { 10, 20, 30, 40 }, result.Outputs["sibling"][0].Select(v => v.ToJson()!.GetValue<int>()));
    }

    [Fact]
    public async Task WholeCallBlockExpandsAcrossScalarAndListSlotsAndKeepsProducerUi()
    {
        var registry = new NodeRegistry();
        registry.Register(Node("Whole", (_, _) => NodeExecutionOutput.Blocked("reason", new() { ["notice"] = new JsonArray("producer") }),
            outputs: [new("*"), new("*", IsList: true)]));
        registry.Register(Node("Zero", (_, _) => NodeExecutionOutput.Blocked(), outputs: []));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"whole":{"class_type":"Whole","inputs":{}},"zero":{"class_type":"Zero","inputs":{}}}
            """), ["whole", "zero"]);
        Assert.Equal("success", result.Status);
        Assert.Empty(result.Diagnostics);
        Assert.Empty(result.Outputs["zero"]);
        var slots = result.Outputs["whole"];
        Assert.Equal(2, slots.Count);
        Assert.All(slots, slot => Assert.Equal("reason", Assert.Single(slot).Blocker.Message));
        slots[0][0].Dispose();
        Assert.Equal("reason", slots[1][0].Blocker.Message);
        Assert.Equal("[\"producer\"]", result.UiOutputs["whole"]["notice"]!.ToJsonString());
    }

    [Fact]
    public async Task InputIsListBlocksOnDirectMembersButDoesNotSearchNestedValues()
    {
        int listCalls = 0;
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (c, _) => Out(c.List([c.Json(JsonValue.Create(1)), c.Blocker()]),
            c.List([c.List([c.Blocker()]), c.Map(new Dictionary<string, RuntimeValue> { ["x"] = c.Blocker() })]), c.List([])),
            outputs: [new("*", IsList: true), new("*", IsList: true), new("*", IsList: true)]));
        registry.Register(Node("Count", (c, i) => { listCalls++; return Out(c.Json(JsonValue.Create(i["value"].Items.Count))); },
            [Input("value"), Input("empty")], inputIsList: true));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Source","inputs":{}},
             "blocked":{"class_type":"Count","inputs":{"value":["p",0],"empty":["p",2]}},
             "nested":{"class_type":"Count","inputs":{"value":["p",1],"empty":["p",2]}}}
            """), ["blocked", "nested"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(1, listCalls);
        Assert.Equal(RuntimeValueKind.Blocker, result.Outputs["blocked"][0][0].Kind);
        Assert.Equal(2, result.Outputs["nested"][0][0].ToJson()!.GetValue<int>());
    }

    [Fact]
    public async Task MixedEmptyListsStillFailBeforeBlockDetectionForOrdinaryMapping()
    {
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (c, _) => Out(c.List([]), c.List([c.Blocker()])), outputs: [new("*", IsList: true), new("*", IsList: true)]));
        registry.Register(Node("Never", (_, _) => throw new InvalidOperationException("wrong failure"), [Input("empty"), Input("block")]));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Source","inputs":{}},"out":{"class_type":"Never","inputs":{"empty":["p",0],"block":["p",1]}}}
            """), ["out"]);
        Assert.Equal("error", result.Status);
        Assert.Equal("Cannot repeat the last item of an empty execution list.", Assert.Single(result.Diagnostics).Message);
    }

    [Fact]
    public async Task EntirelyBlockedLazyCallDoesNotInvokeHookOrResolveBranches()
    {
        int lazyCalls = 0, branchCalls = 0;
        var registry = new NodeRegistry();
        registry.Register(Node("Block", (c, _) => Out(c.Blocker("still present"))));
        registry.Register(Node("Branch", (_, _) => { branchCalls++; throw new InvalidOperationException(); }));
        registry.Register(Node("Lazy", (_, _) => throw new InvalidOperationException(), [Input("selector"), Input("branch", lazy: true)],
            lazy: _ => { lazyCalls++; return ["branch"]; }));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Block","inputs":{}},"branch":{"class_type":"Branch","inputs":{}},
             "out":{"class_type":"Lazy","inputs":{"selector":["p",0],"branch":["branch",0]}}}
            """), ["out"]);
        Assert.Equal(0, lazyCalls);
        Assert.Equal(0, branchCalls);
        Assert.Equal("Execution Blocked: still present", Assert.Single(result.Diagnostics).Message);
        Assert.Equal("success", result.Status);
    }

    [Fact]
    public async Task LazyRowsAreJointlySlicedBeforeFilteringAndKeepRepeatLastPairings()
    {
        var registry = new NodeRegistry();
        registry.Register(Node("Rows", (c, _) => Out(c.List([c.Json(JsonValue.Create(10)), c.Blocker(), c.Json(JsonValue.Create(30))]),
            c.List([c.Json(JsonValue.Create(100)), c.Json(JsonValue.Create(200))])), outputs: [new("*", IsList: true), new("*", IsList: true)]));
        registry.Register(Node("Chosen", (c, _) => Out(c.Json(JsonValue.Create(7)))));
        registry.Register(Node("Unused", (_, _) => throw new InvalidOperationException("unselected lazy branch")));
        int lazyCalls = 0, calls = 0;
        registry.Register(Node("Lazy", (_, i) => { calls++; return Out(i["chosen"]); },
            [Input("a"), Input("b"), Input("chosen", true), Input("unused", true)], lazy: rows =>
            {
                lazyCalls++;
                Assert.Equal(new[] { 10, 30 }, rows["a"].Select(v => v.ToJson()!.GetValue<int>()));
                Assert.Equal(new[] { 100, 200 }, rows["b"].Select(v => v.ToJson()!.GetValue<int>()));
                return ["chosen"];
            }));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Rows","inputs":{}},"chosen":{"class_type":"Chosen","inputs":{}},"unused":{"class_type":"Unused","inputs":{}},
             "out":{"class_type":"Lazy","inputs":{"a":["p",0],"b":["p",1],"chosen":["chosen",0],"unused":["unused",0]}}}
            """), ["out"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(1, lazyCalls);
        Assert.Equal(2, calls);
        Assert.Equal(RuntimeValueKind.Blocker, result.Outputs["out"][0][1].Kind);
    }

    [Fact]
    public async Task SelectedLazyBlockerSkipsSwitchAndLeavesUnselectedProducerUnexecuted()
    {
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(Node("Block", (c, _) => Out(c.Blocker("selected"))));
        registry.Register(Node("Unused", (_, _) => throw new InvalidOperationException("unselected")));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"block":{"class_type":"Block","inputs":{}},"unused":{"class_type":"Unused","inputs":{}},
             "out":{"class_type":"ComfySwitchNode","inputs":{"switch":true,"on_true":["block",0],"on_false":["unused",0]}}}
            """), ["out"]);
        Assert.Equal("success", result.Status);
        Assert.Equal("Execution Blocked: selected", Assert.Single(result.Diagnostics).Message);
        Assert.Null(result.Outputs["out"][0][0].Blocker.Message);
    }

    [Fact]
    public async Task AsyncProducerCanReturnBlockerWithoutStartingItsConsumer()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int calls = 0;
        var registry = new NodeRegistry();
        registry.Register(new TestNode(new("Async", "Async", "test", [], [new("*")]), async (c, _, token) =>
        {
            entered.SetResult();
            await release.Task.WaitAsync(token);
            return Out(c.Blocker());
        }));
        registry.Register(Node("Consumer", (_, i) => { calls++; return Out(i["value"]); }, [Input("value")]));
        var task = new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Async","inputs":{}},"out":{"class_type":"Consumer","inputs":{"value":["p",0]}}}
            """), ["out"]);
        try
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.False(task.IsCompleted);
            Assert.Equal(0, calls);
            release.TrySetResult();
            using var result = await task.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal("success", result.Status);
            Assert.Equal(0, calls);
        }
        finally
        {
            release.TrySetResult();
            using var cleanup = await task.WaitAsync(TimeSpan.FromSeconds(5));
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task CancellationAtBlockedBoundaryWinsOverSuccessfulPropagation(bool verbose)
    {
        using var cancelled = new CancellationTokenSource();
        var events = new List<string>();
        var registry = new NodeRegistry();
        registry.Register(Node("Block", (c, _) => Out(c.Blocker(verbose ? "stop" : null))));
        registry.Register(Node("Never", (_, _) => throw new InvalidOperationException(), [Input("value")]));
        var graph = Prompt("""{"p":{"class_type":"Block","inputs":{}},"out":{"class_type":"Never","inputs":{"value":["p",0]}}}""");
        var engine = new EngineService(registry);
        using var result = await engine.ExecuteValuesAsync(graph, ["out"], async e =>
        {
            events.Add(e.Type);
            if (e.NodeId == "out" && e.Type == (verbose ? "execution_error" : "executing"))
            {
                await Task.Yield();
                cancelled.Cancel();
            }
        }, cancelled.Token);
        Assert.Equal("cancelled", result.Status);
        Assert.Empty(result.Outputs);
        Assert.Single(events, e => e == "execution_interrupted");
        Assert.DoesNotContain("execution_success", events);
        using var next = await engine.ExecuteValuesAsync(graph, ["out"]);
        Assert.Equal("success", next.Status);
        using var preCancelled = await engine.ExecuteValuesAsync(graph, ["out"], cancellationToken: cancelled.Token);
        Assert.Equal("cancelled", preCancelled.Status);
    }

    [Fact]
    public async Task CancellationWhileAwaitingProducerReclaimsResourcesInsteadOfCreatingABlocker()
    {
        var resource = new CountingResource();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var registry = new NodeRegistry();
        registry.Register(new TestNode(new("Wait", "Wait", "test", [], [new("*")]), async (c, _, token) =>
        {
            c.Own(resource); entered.SetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return Out(c.Blocker());
        }));
        using var cancelled = new CancellationTokenSource();
        var task = new EngineService(registry).ExecuteValuesAsync(Prompt("""{"p":{"class_type":"Wait","inputs":{}}}"""), ["p"], cancellationToken: cancelled.Token);
        try
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
            cancelled.Cancel();
            using var result = await task.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal("cancelled", result.Status);
            Assert.Empty(result.Outputs);
            Assert.Empty(result.Diagnostics);
            Assert.Equal(1, resource.DisposeCount);
        }
        finally
        {
            cancelled.Cancel();
            using var cleanup = await task.WaitAsync(TimeSpan.FromSeconds(5));
        }
    }

    [Fact]
    public async Task MappedExceptionsReleasePriorOutputsAndKeepIndependentTargetAlive()
    {
        var resource = new CountingResource();
        var registry = new NodeRegistry();
        registry.Register(Node("Rows", (c, _) => Out(c.List([c.Json(JsonValue.Create(1)), c.Blocker(), c.Json(JsonValue.Create(3))])), outputs: [new("*", IsList: true)]));
        registry.Register(Node("Fail", (c, i) =>
        {
            if (i["value"].ToJson()!.GetValue<int>() == 3) throw new InvalidOperationException("ordinary failure");
            return Out(c.Own(resource));
        }, [Input("value")]));
        registry.Register(Node("Ok", (c, _) => Out(c.Json(JsonValue.Create(9)))));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"p":{"class_type":"Rows","inputs":{}},"fail":{"class_type":"Fail","inputs":{"value":["p",0]}},"ok":{"class_type":"Ok","inputs":{}}}
            """), ["fail", "ok"]);
        Assert.Equal("error", result.Status);
        Assert.Equal("execution_error", Assert.Single(result.Diagnostics).Code);
        Assert.Equal("ordinary failure", result.Diagnostics[0].Message);
        Assert.False(result.Outputs.ContainsKey("fail"));
        Assert.Equal(9, result.Outputs["ok"][0][0].ToJson()!.GetValue<int>());
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task UiExecutionSkipsBlockedOutputWhileJsonProjectionRejectsRawMarker()
    {
        int calls = 0;
        var registry = new NodeRegistry();
        registry.Register(Node("Block", (c, _) => Out(c.Blocker())));
        registry.Register(Node("Output", (_, i) => { calls++; return new([i["value"]], new() { ["data"] = new JsonArray("must not appear") }); }, [Input("value")]));
        var engine = new EngineService(registry);
        var graph = Prompt("""{"p":{"class_type":"Block","inputs":{}},"out":{"class_type":"Output","inputs":{"value":["p",0]}}}""");
        var ui = await engine.ExecuteUiAsync(graph, ["out"]);
        Assert.Equal("success", ui.Status);
        Assert.Empty(ui.Outputs);
        Assert.Empty(ui.Diagnostics);
        Assert.Equal(0, calls);
        var json = await engine.ExecuteAsync(graph, ["out"]);
        Assert.Equal("error", json.Status);
        Assert.Equal("runtime_value_not_json", Assert.Single(json.Diagnostics).Code);
        Assert.Empty(json.Outputs);
        Assert.Equal(0, calls);
    }

    [Fact]
    public async Task JsonObjectResemblingABlockerRemainsOrdinaryData()
    {
        var registry = new NodeRegistry();
        registry.Register(Node("Echo", (_, i) => Out(i["value"]), [Input("value")]));
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""
            {"out":{"class_type":"Echo","inputs":{"value":{"kind":"Blocker","message":null}}}}
            """), ["out"]);
        Assert.Equal("success", result.Status);
        Assert.Equal("Blocker", result.Outputs["out"][0][0]!["kind"]!.GetValue<string>());
    }

    private sealed class TestNode(NodeSchema schema,
        Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, CancellationToken, ValueTask<NodeExecutionOutput>> run,
        Func<IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>>, IReadOnlyCollection<string>>? lazy = null) : IRuntimeNode
    {
        public NodeSchema Schema => schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) =>
            run(context, inputs, cancellationToken);
        public IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> resolvedInputs) => lazy?.Invoke(resolvedInputs) ?? [];
    }

    private sealed class JsonConsumer(Action called) : INode
    {
        public NodeSchema Schema { get; } = new("JsonConsumer", "JsonConsumer", "test", [Input("value")], [new("*")]);
        public ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> inputs, CancellationToken cancellationToken)
        { called(); return ValueTask.FromResult<IReadOnlyList<JsonNode?>>([inputs["value"]]); }
    }

    private sealed class CountingResource : IDisposable
    {
        public int DisposeCount { get; private set; }
        public void Dispose() => DisposeCount++;
    }
}
