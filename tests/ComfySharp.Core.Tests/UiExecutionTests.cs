using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

// Counting resources and test-only nodes prove the engine contract, not inference compatibility.
public sealed class UiExecutionTests
{
    private static JsonObject Prompt(string json) => JsonNode.Parse(json)!.AsObject();
    private static JsonObject Text(string text) => new() { ["text"] = new JsonArray(text) };
    private static string First(JsonObject ui) => ui["text"]![0]!.GetValue<string>();
    private static RuntimeNode Node(string name, Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, NodeExecutionOutput> execute,
        InputSchema[]? inputs = null, OutputSchema[]? outputs = null, bool outputNode = false) =>
        new(new(name, name, "test", inputs ?? [], outputs ?? [new("*")], OutputNode: outputNode),
            (context, values, _) => ValueTask.FromResult(execute(context, values)));

    [Fact]
    public async Task Native_slots_and_ancestor_ui_are_independent_snapshots_in_results_and_events()
    {
        var resource = new Resource();
        var sourceUi = Text("original");
        var events = new List<EngineEvent>();
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (context, _) => new([context.Own(resource)], sourceUi)));
        registry.Register(Node("Alias", (_, inputs) =>
        {
            sourceUi["text"]![0] = "node mutation after handoff";
            return new([inputs["value"]]);
        }, [new("value", "*")]));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"source":{"class_type":"Source","inputs":{}},"alias":{"class_type":"Alias","inputs":{"value":["source",0]}}}
            """), ["alias"], e =>
            {
                events.Add(e);
                if (e.Output is not null) e.Output["text"]![0] = "event mutation";
                return ValueTask.CompletedTask;
            });

        Assert.Equal("success", result.Status);
        Assert.Equal("alias", Assert.Single(result.Outputs).Key);
        Assert.Same(resource, result.Outputs["alias"][0][0].GetNative<Resource>());
        Assert.Equal("source", Assert.Single(result.UiOutputs).Key);
        Assert.Equal("original", First(result.UiOutputs["source"]));
        Assert.Equal(new NodeExecutionIdentity("source", "source"), result.Meta["source"]);
        var executed = Assert.Single(events, e => e.Type == "executed");
        Assert.Equal("source", executed.NodeId);
        Assert.Equal(result.Meta["source"], executed.Identity);
        result.UiOutputs["source"]["text"]![0] = "result mutation";
        Assert.Equal("event mutation", First(executed.Output!));
        Assert.Equal(0, resource.DisposeCount);
        result.Dispose();
        Assert.Equal(1, resource.DisposeCount);
        Assert.Equal("result mutation", First(result.UiOutputs["source"]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Ui_boundary_does_not_project_native_slots_and_disposes_before_terminal_event(bool withUi)
    {
        var resource = new Resource();
        var events = new List<EngineEvent>();
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (context, _) => new([context.Own(resource)], withUi ? Text("preview") : null)));
        var result = await new EngineService(registry).ExecuteUiAsync(
            Prompt("""{"source":{"class_type":"Source","inputs":{}}}"""), ["source"], e =>
            {
                events.Add(e);
                if (e.Type == "execution_success") Assert.Equal(1, resource.DisposeCount);
                return ValueTask.CompletedTask;
            });
        Assert.Equal("success", result.Status);
        Assert.Empty(result.Diagnostics);
        Assert.Equal(1, resource.DisposeCount);
        Assert.Equal(withUi ? 1 : 0, result.Outputs.Count);
        Assert.Equal(withUi ? 1 : 0, result.Meta.Count);
        Assert.Equal(withUi ? 1 : 0, events.Count(e => e.Type == "executed"));
        if (withUi) Assert.Equal("preview", First(result.Outputs["source"]));
        Assert.Equal("execution_success", events[^1].Type);
    }

    [Fact]
    public async Task An_output_node_can_produce_ui_with_zero_execution_slots()
    {
        var registry = new NodeRegistry();
        registry.Register(Node("UiOnly", (_, _) => new([], Text("visible")), outputs: [], outputNode: true));
        var result = await new EngineService(registry).ExecuteUiAsync(
            Prompt("""{"out":{"class_type":"UiOnly","inputs":{}}}"""));
        Assert.Equal("success", result.Status);
        Assert.Equal("visible", First(result.Outputs["out"]));
    }

    [Fact]
    public async Task Mapping_copies_reused_ui_documents_and_concatenates_first_return_keys_only()
    {
        var registry = new NodeRegistry();
        registry.Register(Node("List", (context, _) => new([context.List([context.Json(JsonValue.Create(1)), context.Json(JsonValue.Create(2))])]),
            outputs: [new("INT", IsList: true)]));
        var reused = Text("uninitialized");
        registry.Register(Node("Map", (_, inputs) =>
        {
            var value = inputs["value"].ToJson()!.GetValue<int>();
            reused["text"]![0] = value.ToString(System.Globalization.CultureInfo.InvariantCulture);
            if (value == 2) reused["later_only"] = new JsonArray("not merged");
            return new([inputs["value"]], reused);
        }, [new("value", "INT")]));
        var result = await new EngineService(registry).ExecuteUiAsync(Prompt("""
            {"source":{"class_type":"List","inputs":{}},"out":{"class_type":"Map","inputs":{"value":["source",0]}}}
            """), ["out"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(new[] { "1", "2" }, result.Outputs["out"]["text"]!.AsArray().Select(n => n!.GetValue<string>()));
        Assert.False(result.Outputs["out"].ContainsKey("later_only"));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Null_ui_is_omitted_but_empty_first_ui_controls_the_merge(bool firstEmpty)
    {
        var registry = new NodeRegistry();
        registry.Register(Node("List", (context, _) => new([context.List([context.Json(JsonValue.Create(1)), context.Json(JsonValue.Create(2))])]),
            outputs: [new("INT", IsList: true)]));
        registry.Register(Node("Map", (_, inputs) => new([inputs["value"]], inputs["value"].ToJson()!.GetValue<int>() == 1
            ? firstEmpty ? new JsonObject() : null : Text("second")), [new("value", "INT")]));
        var events = new List<EngineEvent>();
        var result = await new EngineService(registry).ExecuteUiAsync(Prompt("""
            {"source":{"class_type":"List","inputs":{}},"out":{"class_type":"Map","inputs":{"value":["source",0]}}}
            """), ["out"], e => { events.Add(e); return ValueTask.CompletedTask; });
        Assert.Equal("success", result.Status);
        Assert.Equal(firstEmpty ? 0 : 1, result.Outputs.Count);
        Assert.Equal(firstEmpty ? 0 : 1, events.Count(e => e.Type == "executed"));
        if (!firstEmpty) Assert.Equal("second", First(result.Outputs["out"]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Missing_or_non_array_ui_entry_fails_the_mapped_node_and_reclaims_native_outputs(bool nonArray)
    {
        var resources = new List<Resource>();
        var registry = new NodeRegistry();
        registry.Register(Node("List", (context, _) => new([context.List([context.Json(JsonValue.Create(1)), context.Json(JsonValue.Create(2))])]),
            outputs: [new("INT", IsList: true)]));
        registry.Register(Node("Map", (context, inputs) =>
        {
            var resource = new Resource(); resources.Add(resource);
            var ui = inputs["value"].ToJson()!.GetValue<int>() == 1 ? Text("first")
                : nonArray ? new JsonObject { ["text"] = 42 } : new JsonObject();
            return new([context.Own(resource)], ui);
        }, [new("value", "INT")]));
        var events = new List<EngineEvent>();
        var result = await new EngineService(registry).ExecuteUiAsync(Prompt("""
            {"source":{"class_type":"List","inputs":{}},"out":{"class_type":"Map","inputs":{"value":["source",0]}}}
            """), ["out"], e => { events.Add(e); return ValueTask.CompletedTask; });
        Assert.Equal("error", result.Status);
        Assert.Empty(result.Outputs); Assert.Empty(result.Meta);
        Assert.Contains(result.Diagnostics, d => d.NodeId == "out" && d.Message.Contains("UI output 'text'", StringComparison.Ordinal));
        Assert.DoesNotContain(events, e => e.Type is "executed" or "execution_success");
        Assert.Equal(2, resources.Count);
        Assert.All(resources, resource => Assert.Equal(1, resource.DisposeCount));
    }

    [Fact]
    public async Task Failed_target_retains_completed_ancestor_and_independent_ui_with_all_resources_disposed()
    {
        var resource = new Resource();
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (context, _) => new([context.Own(resource)], Text("ancestor"))));
        registry.Register(Node("Fail", (_, _) => throw new InvalidOperationException("broken consumer"), [new("value", "*")]));
        registry.Register(Node("Ui", (_, _) => new([], Text("independent")), outputs: []));
        var result = await new EngineService(registry).ExecuteUiAsync(Prompt("""
            {"source":{"class_type":"Source","inputs":{}},"bad":{"class_type":"Fail","inputs":{"value":["source",0]}},
             "ok":{"class_type":"Ui","inputs":{}}}
            """), ["bad", "ok"]);
        Assert.Equal("error", result.Status);
        Assert.Equal(2, result.Outputs.Count);
        Assert.Equal("ancestor", First(result.Outputs["source"]));
        Assert.Equal("independent", First(result.Outputs["ok"]));
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task Cancellation_preserves_completed_ui_and_disposes_every_resource_before_interruption()
    {
        var source = new Resource();
        var pending = new Resource();
        var registry = new NodeRegistry();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        registry.Register(Node("Source", (context, _) => new([context.Own(source)], Text("completed"))));
        registry.Register(new RuntimeNode(new("Wait", "Wait", "test", [], [new("*")]), async (context, _, token) =>
        {
            context.Own(pending);
            entered.SetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            throw new InvalidOperationException("unreachable");
        }));
        using var cancellation = new CancellationTokenSource();
        var events = new List<string>();
        var engine = new EngineService(registry);
        var task = engine.ExecuteUiAsync(Prompt("""
            {"source":{"class_type":"Source","inputs":{}},"wait":{"class_type":"Wait","inputs":{}}}
            """), ["source", "wait"], e =>
            {
                events.Add(e.Type);
                if (e.Type == "execution_interrupted")
                {
                    Assert.Equal(1, source.DisposeCount); Assert.Equal(1, pending.DisposeCount);
                }
                return ValueTask.CompletedTask;
            }, cancellation.Token);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        cancellation.Cancel();
        var result = await task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal("cancelled", result.Status);
        Assert.Equal("completed", First(result.Outputs["source"]));
        Assert.Equal("execution_interrupted", events[^1]);
        Assert.DoesNotContain("execution_success", events);
    }

    [Theory]
    [InlineData("values")]
    [InlineData("json")]
    [InlineData("ui")]
    public async Task Memo_disposer_failure_cannot_emit_success_or_leak_retained_target(string boundary)
    {
        var source = new Resource(throwOnDispose: true);
        var target = new Resource();
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (context, _) => new([context.Own(source)])));
        registry.Register(Node("Target", (context, _) => new([context.Own(target)], Text("target")), [new("value", "*")]));
        var events = new List<string>();
        var prompt = Prompt("""
            {"source":{"class_type":"Source","inputs":{}},"target":{"class_type":"Target","inputs":{"value":["source",0]}}}
            """);
        var engine = new EngineService(registry);
        ValueTask Event(EngineEvent e) { events.Add(e.Type); return ValueTask.CompletedTask; }
        await Assert.ThrowsAsync<AggregateException>(async () =>
        {
            if (boundary == "values") { using var result = await engine.ExecuteValuesAsync(prompt, ["target"], Event); }
            else if (boundary == "json") await engine.ExecuteAsync(prompt, ["target"], Event);
            else await engine.ExecuteUiAsync(prompt, ["target"], Event);
        });
        Assert.DoesNotContain("execution_success", events);
        Assert.Equal(1, source.DisposeCount); Assert.Equal(1, target.DisposeCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Result_disposer_failure_cannot_emit_success_at_a_serializable_boundary(bool ui)
    {
        var resource = new Resource(throwOnDispose: true);
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (context, _) => new([context.Own(resource)], Text("preview"))));
        var engine = new EngineService(registry);
        var events = new List<string>();
        var prompt = Prompt("""{"source":{"class_type":"Source","inputs":{}}}""");
        ValueTask Event(EngineEvent e) { events.Add(e.Type); return ValueTask.CompletedTask; }
        await Assert.ThrowsAsync<AggregateException>(async () =>
        {
            if (ui) await engine.ExecuteUiAsync(prompt, ["source"], Event);
            else await engine.ExecuteAsync(prompt, ["source"], Event);
        });
        Assert.DoesNotContain("execution_success", events);
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task Ui_terminal_callback_failure_occurs_after_native_disposal()
    {
        var resource = new Resource();
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (context, _) => new([context.Own(resource)], Text("preview"))));
        await Assert.ThrowsAsync<InvalidOperationException>(() => new EngineService(registry).ExecuteUiAsync(
            Prompt("""{"source":{"class_type":"Source","inputs":{}}}"""), ["source"], e =>
            {
                if (e.Type == "execution_success")
                {
                    Assert.Equal(1, resource.DisposeCount);
                    throw new InvalidOperationException("transport closed");
                }
                return ValueTask.CompletedTask;
            }));
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task Terminal_transport_cancellation_does_not_attempt_to_complete_already_disposed_leases_again()
    {
        var resource = new Resource();
        var registry = new NodeRegistry();
        using var cancellation = new CancellationTokenSource();
        registry.Register(Node("Source", (context, _) => new([context.Own(resource)])));
        var events = new List<string>();
        await Assert.ThrowsAsync<OperationCanceledException>(() => new EngineService(registry).ExecuteValuesAsync(
            Prompt("""{"source":{"class_type":"Source","inputs":{}}}"""), ["source"], e =>
            {
                events.Add(e.Type);
                if (e.Type == "execution_success")
                {
                    cancellation.Cancel();
                    throw new OperationCanceledException(cancellation.Token);
                }
                return ValueTask.CompletedTask;
            }, cancellation.Token));
        Assert.DoesNotContain("execution_interrupted", events);
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public void Ui_snapshots_support_parsed_and_programmatic_json_without_calling_arbitrary_getters()
    {
        var ui = new JsonObject
        {
            ["items"] = new JsonArray(null, true, 2, 1.5, "é"),
            ["parsed"] = JsonNode.Parse("""{"nested":[1,{"text":"x"}]}""")
        };
        Assert.True(JsonNode.DeepEquals(ui, UiDocument.Snapshot(ui)));
        var probe = new SerializationProbe();
        Assert.Throws<InvalidOperationException>(() => UiDocument.Snapshot(new() { ["unsafe"] = JsonValue.Create(probe) }));
        Assert.Equal(0, probe.GetterCalls);
        Assert.Throws<InvalidOperationException>(() => UiDocument.Snapshot(new() { ["invalid"] = double.NaN }));
    }

    [Fact]
    public async Task Non_json_ui_fails_before_executed_or_success_and_reclaims_the_native_slot()
    {
        var resource = new Resource();
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (context, _) => new([context.Own(resource)], new() { ["text"] = new JsonArray(double.PositiveInfinity) })));
        var events = new List<string>();
        var result = await new EngineService(registry).ExecuteUiAsync(
            Prompt("""{"source":{"class_type":"Source","inputs":{}}}"""), ["source"], e => { events.Add(e.Type); return ValueTask.CompletedTask; });
        Assert.Equal("error", result.Status);
        Assert.Empty(result.Outputs);
        Assert.DoesNotContain("executed", events); Assert.DoesNotContain("execution_success", events);
        Assert.Equal(1, resource.DisposeCount);
    }

    private sealed class Resource(bool throwOnDispose = false) : IDisposable
    {
        public int DisposeCount { get; private set; }
        public void Dispose()
        {
            DisposeCount++;
            if (throwOnDispose) throw new InvalidOperationException("resource disposal failed");
        }
    }
    private sealed class SerializationProbe
    {
        public int GetterCalls;
        public string Value { get { GetterCalls++; throw new InvalidOperationException("must never serialize this object"); } }
    }
    private sealed class RuntimeNode(NodeSchema schema,
        Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, CancellationToken, ValueTask<NodeExecutionOutput>> execute) : IRuntimeNode
    {
        public NodeSchema Schema { get; } = schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs,
            CancellationToken cancellationToken) => execute(context, inputs, cancellationToken);
    }
}
