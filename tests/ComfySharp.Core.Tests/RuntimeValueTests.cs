using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>Managed counting resources verify ownership contracts only, not tensor operations or model parity.</summary>
public sealed class RuntimeValueTests
{
    private static JsonObject Prompt(string json) => JsonNode.Parse(json)!.AsObject();
    private static NodeSchema Schema(string name, InputSchema[]? inputs = null, OutputSchema[]? outputs = null, bool inputIsList = false) =>
        new(name, name, "test", inputs ?? [], outputs ?? [new("*")], InputIsList: inputIsList);
    private static RuntimeNode Node(string name, Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, IReadOnlyList<RuntimeValue>> run,
        InputSchema[]? inputs = null, OutputSchema[]? outputs = null, bool inputIsList = false) =>
        new(Schema(name, inputs, outputs, inputIsList), (context, values, _) => ValueTask.FromResult(run(context, values)));

    [Fact]
    public void Literal_arrays_execution_lists_and_native_maps_have_distinct_kinds_and_independent_leases()
    {
        var resource = new CountingResource();
        using var context = new RuntimeNodeContext();
        var literal = context.Json(new JsonArray("a", "b"));
        var native = context.Own(resource);
        var map = context.Map(new Dictionary<string, RuntimeValue> { ["samples"] = native, ["metadata"] = literal });
        var list = context.List([map, map]);
        Assert.Equal(RuntimeValueKind.Json, literal.Kind);
        Assert.Equal(RuntimeValueKind.Map, map.Kind);
        Assert.Equal(RuntimeValueKind.List, list.Kind);
        Assert.Throws<RuntimeValueProjectionException>(() => list.ToJson());
        using var retained = list.Retain();
        context.Dispose();
        Assert.Throws<ObjectDisposedException>(() => literal.ToJson());
        Assert.Throws<ObjectDisposedException>(() => native.GetNative<CountingResource>());
        Assert.Same(resource, retained.Items[1].Properties["samples"].GetNative<CountingResource>());
        Assert.Equal(0, resource.DisposeCount);
        retained.Items[0].Dispose();
        Assert.Same(resource, retained.Items[1].Properties["samples"].GetNative<CountingResource>());
        retained.Dispose();
        retained.Dispose();
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public void Json_factories_and_projections_never_expose_mutable_snapshots()
    {
        using var context = new RuntimeNodeContext();
        var original = new JsonObject { ["nested"] = new JsonArray("original") };
        var value = context.Json(original);
        original["nested"]![0] = "source change";
        var projected = value.ToJson();
        projected!["nested"]![0] = "consumer change";
        Assert.Equal("original", value.ToJson()!["nested"]![0]!.GetValue<string>());
    }

    [Fact]
    public void Partial_container_factory_failure_releases_every_temporary_lease()
    {
        var resource = new CountingResource();
        using var context = new RuntimeNodeContext();
        var live = context.Own(resource);
        var invalid = context.Json(null);
        invalid.Dispose();
        Assert.Throws<ObjectDisposedException>(() => context.List([live, invalid]));
        Assert.Throws<ObjectDisposedException>(() => context.Map(new Dictionary<string, RuntimeValue> { ["live"] = live, ["invalid"] = invalid }));
        context.Dispose();
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public void Repeated_registration_in_one_context_does_not_create_two_resource_owners()
    {
        var resource = new CountingResource();
        using var context = new RuntimeNodeContext();
        var first = context.Own(resource);
        var second = context.Own(resource);
        first.Dispose();
        Assert.Same(resource, second.GetNative<CountingResource>());
        Assert.Equal(0, resource.DisposeCount);
        context.Dispose();
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task Fanout_duplicate_targets_and_alias_slots_share_resources_with_separate_result_leases()
    {
        var resource = new CountingResource();
        var calls = 0;
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (c, _) => { calls++; var value = c.Own(resource); return [value, value]; }, outputs: [new("*"), new("*")]));
        registry.Register(Node("Alias", (_, i) => { Assert.Equal(0, resource.DisposeCount); return [i["value"], i["value"]]; },
            [new("value", "*")], [new("*"), new("*")]));
        var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"source":{"class_type":"Source","inputs":{}},
             "left":{"class_type":"Alias","inputs":{"value":["source",0]}},
             "right":{"class_type":"Alias","inputs":{"value":["source",1]}}}
            """), ["left", "right", "source", "left"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(1, calls);
        Assert.Equal(3, result.Outputs.Count);
        var first = result.Outputs["left"][0][0];
        var second = result.Outputs["left"][1][0];
        Assert.NotSame(first, second);
        first.Dispose();
        Assert.Same(resource, second.GetNative<CountingResource>());
        using var external = result.Outputs["right"][0][0].Retain();
        result.Dispose();
        result.Dispose();
        Assert.Throws<ObjectDisposedException>(() => result.Outputs);
        Assert.Throws<ObjectDisposedException>(() => second.GetNative<CountingResource>());
        Assert.Equal(0, resource.DisposeCount);
        Assert.Same(resource, external.GetNative<CountingResource>());
        external.Dispose();
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task Async_failure_reclaims_allocations_and_independent_target_still_executes()
    {
        var resource = new CountingResource();
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new RuntimeNode(Schema("Fail"), async (c, _, _) =>
        {
            c.Own(resource);
            await Task.Yield();
            throw new InvalidOperationException("after allocation");
        }));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"bad":{"class_type":"Fail","inputs":{}},"ok":{"class_type":"PrimitiveInt","inputs":{"value":9}}}
            """), ["bad", "ok"]);
        Assert.Equal("error", result.Status);
        Assert.Equal(1, resource.DisposeCount);
        Assert.Equal(9L, result.Outputs["ok"][0][0].ToJson()!.GetValue<long>());
        Assert.Contains(result.Diagnostics, d => d.NodeId == "bad" && d.Message == "after allocation");
    }

    [Fact]
    public async Task Cancellation_reclaims_unreturned_allocations_without_affecting_successor()
    {
        var resource = new CountingResource();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new RuntimeNode(Schema("Wait"), async (c, _, token) =>
        {
            var value = c.Own(resource);
            entered.SetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return [value];
        }));
        var engine = new EngineService(registry);
        using var cancellation = new CancellationTokenSource();
        var running = engine.ExecuteValuesAsync(Prompt("""{"wait":{"class_type":"Wait","inputs":{}}}"""), ["wait"], cancellationToken: cancellation.Token);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        cancellation.Cancel();
        using var result = await running.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal("cancelled", result.Status);
        Assert.Equal(1, resource.DisposeCount);
        using var next = await engine.ExecuteValuesAsync(Prompt("""{"ok":{"class_type":"PrimitiveInt","inputs":{"value":5}}}"""), ["ok"]);
        Assert.Equal("success", next.Status);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Invalid_output_count_or_list_kind_reclaims_aliases_and_partially_copied_outputs(bool invalidList)
    {
        var resource = new CountingResource();
        var registry = new NodeRegistry();
        registry.Register(Node("Invalid", (c, _) => { var value = c.Own(resource); return [value, value]; },
            outputs: invalidList ? [new("*"), new("*", IsList: true)] : [new("*")]));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""{"bad":{"class_type":"Invalid","inputs":{}}}"""), ["bad"]);
        Assert.Equal("error", result.Status);
        Assert.Empty(result.Outputs);
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task Mapped_failure_disposes_prior_invocation_outputs_and_current_allocations()
    {
        var resources = new List<CountingResource>();
        var registry = new NodeRegistry();
        registry.Register(Node("List", (c, _) => [c.List([c.Json(JsonValue.Create(1)), c.Json(JsonValue.Create(2))])], outputs: [new("INT", IsList: true)]));
        registry.Register(Node("Map", (c, i) =>
        {
            var resource = new CountingResource(); resources.Add(resource);
            var value = c.Own(resource);
            if (i["value"].ToJson()!.GetValue<int>() == 2) throw new InvalidOperationException("second invocation");
            return [value];
        }, [new("value", "INT")]));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"list":{"class_type":"List","inputs":{}},"map":{"class_type":"Map","inputs":{"value":["list",0]}}}
            """), ["map"]);
        Assert.Equal("error", result.Status);
        Assert.Equal(2, resources.Count);
        Assert.All(resources, r => Assert.Equal(1, r.DisposeCount));
    }

    [Fact]
    public async Task Lazy_hooks_borrow_native_values_and_switch_preserves_native_maps()
    {
        var resource = new CountingResource();
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(Node("Source", (c, _) => [c.Map(new Dictionary<string, RuntimeValue> { ["samples"] = c.Own(resource) })]));
        registry.Register(Node("Fail", (_, _) => throw new InvalidOperationException("unselected branch")));
        RuntimeValue? hookBorrow = null;
        registry.Register(new RuntimeNode(Schema("Probe", [new("value", "*")]), (_, i, _) =>
        {
            Assert.Throws<ObjectDisposedException>(() => hookBorrow!.Properties);
            Assert.Same(resource, i["value"].Properties["samples"].GetNative<CountingResource>());
            return ValueTask.FromResult<IReadOnlyList<RuntimeValue>>([i["value"]]);
        }, values =>
        {
            hookBorrow = values["value"][0];
            Assert.Same(resource, hookBorrow.Properties["samples"].GetNative<CountingResource>());
            return [];
        }));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"source":{"class_type":"Source","inputs":{}},"bad":{"class_type":"Fail","inputs":{}},
             "switch":{"class_type":"ComfySwitchNode","inputs":{"switch":true,"on_true":["source",0],"on_false":["bad",0]}},
             "probe":{"class_type":"Probe","inputs":{"value":["switch",0]}}}
            """), ["probe"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(0, resource.DisposeCount);
        Assert.Same(resource, result.Outputs["probe"][0][0].Properties["samples"].GetNative<CountingResource>());
        result.Dispose();
        Assert.Equal(1, resource.DisposeCount);
    }

    [Theory]
    [InlineData("executed")]
    [InlineData("execution_success")]
    [InlineData("execution_error")]
    public async Task Event_callback_failure_reclaims_native_values(string failedEvent)
    {
        var resource = new CountingResource();
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (c, _) =>
        {
            var value = c.Own(resource);
            if (failedEvent == "execution_error") throw new InvalidOperationException("node failure");
            return [value];
        }));
        var task = new EngineService(registry).ExecuteValuesAsync(Prompt("""{"source":{"class_type":"Source","inputs":{}}}"""), ["source"], e =>
            e.Type == failedEvent ? throw new InvalidOperationException("event failure") : ValueTask.CompletedTask);
        if (failedEvent == "executed")
        {
            using var result = await task;
            Assert.Equal("error", result.Status);
        }
        else await Assert.ThrowsAsync<InvalidOperationException>(async () => await task);
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task Json_projection_diagnoses_native_target_preserves_json_target_and_disposes_every_native_lease()
    {
        var resource = new CountingResource();
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(Node("Source", (c, _) => [c.Map(new Dictionary<string, RuntimeValue> { ["samples"] = c.Own(resource) })]));
        var events = new List<string>();
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""
            {"native":{"class_type":"Source","inputs":{}},"json":{"class_type":"PrimitiveInt","inputs":{"value":3}}}
            """), ["native", "json"], e => { events.Add(e.Type); return ValueTask.CompletedTask; });
        Assert.Equal("error", result.Status);
        Assert.Single(result.Outputs);
        Assert.Equal(3L, result.Outputs["json"][0][0]!.GetValue<long>());
        Assert.Contains(result.Diagnostics, d => d.Code == "runtime_value_not_json" && d.TargetId == "native");
        Assert.DoesNotContain("execution_success", events);
        Assert.Equal("execution_failed", events[^1]);
        Assert.Equal(1, resource.DisposeCount);
    }

    [Fact]
    public async Task Json_projection_callback_exception_still_disposes_the_owned_result()
    {
        var resource = new CountingResource();
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (c, _) => [c.Own(resource)]));
        await Assert.ThrowsAsync<InvalidOperationException>(() => new EngineService(registry).ExecuteAsync(
            Prompt("""{"source":{"class_type":"Source","inputs":{}}}"""), ["source"],
            e => e.Type == "execution_error" ? throw new InvalidOperationException("projection event failed") : ValueTask.CompletedTask));
        Assert.Equal(1, resource.DisposeCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Cancellation_keeps_completed_target_leases_until_result_disposal_and_callback_failure_reclaims_them(bool callbackFails)
    {
        var completed = new CountingResource();
        var interrupted = new CountingResource();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (c, _) => [c.Own(completed)]));
        registry.Register(new RuntimeNode(Schema("Wait"), async (c, _, token) =>
        {
            c.Own(interrupted);
            entered.SetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return [];
        }));
        using var cancellation = new CancellationTokenSource();
        var running = new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"source":{"class_type":"Source","inputs":{}},"wait":{"class_type":"Wait","inputs":{}}}
            """), ["source", "wait"], e => e.Type == "execution_interrupted" && callbackFails
                ? throw new InvalidOperationException("interruption event failed") : ValueTask.CompletedTask, cancellation.Token);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        cancellation.Cancel();
        if (callbackFails) await Assert.ThrowsAsync<InvalidOperationException>(async () => await running.WaitAsync(TimeSpan.FromSeconds(5)));
        else
        {
            using var result = await running.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal("cancelled", result.Status);
            Assert.Equal(0, completed.DisposeCount);
            Assert.Same(completed, result.Outputs["source"][0][0].GetNative<CountingResource>());
        }
        Assert.Equal(1, completed.DisposeCount);
        Assert.Equal(1, interrupted.DisposeCount);
    }

    [Fact]
    public async Task Disposal_failure_reclaims_other_memo_and_result_resources_before_propagating()
    {
        var source = new CountingResource(throwOnDispose: true);
        var target = new CountingResource();
        var registry = new NodeRegistry();
        registry.Register(Node("Source", (c, _) => [c.Own(source)]));
        registry.Register(Node("Target", (c, _) => [c.Own(target)], [new("value", "*")]));
        await Assert.ThrowsAsync<AggregateException>(() => new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"source":{"class_type":"Source","inputs":{}},"target":{"class_type":"Target","inputs":{"value":["source",0]}}}
            """), ["target"]));
        Assert.Equal(1, source.DisposeCount);
        Assert.Equal(1, target.DisposeCount);
    }

    [Fact]
    public async Task Concurrent_invocation_allocations_register_every_resource_and_share_duplicate_owners()
    {
        var shared = new CountingResource();
        var resources = Enumerable.Range(0, 64).Select(_ => new CountingResource()).ToArray();
        using var context = new RuntimeNodeContext();
        var values = await Task.WhenAll(resources.Select(resource => Task.Run(() =>
            context.List([context.Own(shared), context.Own(resource)]))));
        Assert.All(values, value => Assert.Same(shared, value.Items[0].GetNative<CountingResource>()));
        using var retained = values[0].Retain();
        context.Dispose();
        Assert.Equal(0, shared.DisposeCount);
        Assert.Equal(0, resources[0].DisposeCount);
        Assert.All(resources.Skip(1), value => Assert.Equal(1, value.DisposeCount));
        retained.Dispose();
        Assert.Equal(1, shared.DisposeCount);
        Assert.All(resources, value => Assert.Equal(1, value.DisposeCount));
    }

    private sealed class CountingResource(bool throwOnDispose = false) : IDisposable
    {
        public int DisposeCount { get; private set; }
        public void Dispose()
        {
            DisposeCount++;
            if (throwOnDispose) throw new InvalidOperationException("resource disposal failed");
        }
    }
    private sealed class RuntimeNode(NodeSchema schema,
        Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, CancellationToken, ValueTask<IReadOnlyList<RuntimeValue>>> execute,
        Func<IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>>, IReadOnlyCollection<string>>? lazy = null) : IRuntimeNode
    {
        public NodeSchema Schema { get; } = schema;
        public ValueTask<IReadOnlyList<RuntimeValue>> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) =>
            execute(context, inputs, cancellationToken);
        public IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> resolvedInputs) => lazy?.Invoke(resolvedInputs) ?? [];
    }
}
