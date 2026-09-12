using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class NodeObjectCacheTests
{
    private static JsonObject Prompt(params (string Id, string Type)[] nodes) => new(nodes.Select(n =>
        new KeyValuePair<string, JsonNode?>(n.Id, new JsonObject { ["class_type"] = n.Type, ["inputs"] = new JsonObject() })));
    private static EngineService Engine(params Factory[] factories)
    {
        var registry = new NodeRegistry(); foreach (var factory in factories) registry.Register(factory); return new(registry);
    }
    private static int Value(ExecutionResult result, string id, int index = 0)
    { Assert.Equal("success", result.Status); return result.Outputs[id][0][index]!.GetValue<int>(); }

    [Fact]
    public async Task Same_id_and_type_reuse_an_instance_across_input_changes_and_distinct_ids_are_separate()
    {
        var factory = new Factory("Counter"); using var engine = Engine(factory);
        var prompt = Prompt(("a", "Counter"), ("b", "Counter"));
        Assert.Equal(1, Value(await engine.ExecuteAsync(prompt), "a"));
        Assert.Equal(2, factory.Instances.Count);
        prompt["a"]!["inputs"]!["value"] = "changed";
        var result = await engine.ExecuteAsync(prompt);
        Assert.Equal(2, Value(result, "a")); Assert.Equal(2, Value(result, "b"));
        Assert.Equal(2, factory.Instances.Count); Assert.Equal(0, factory.DefinitionDisposals);
    }

    [Fact]
    public async Task Unselected_nodes_are_not_created_but_existing_listed_nodes_remain_cached()
    {
        var factory = new Factory("Counter"); using var engine = Engine(factory);
        var prompt = Prompt(("a", "Counter"), ("b", "Counter"));
        Assert.Equal(1, Value(await engine.ExecuteAsync(prompt, ["a"]), "a"));
        Assert.Single(factory.Instances);
        Assert.Equal(1, Value(await engine.ExecuteAsync(prompt, ["b"]), "b"));
        Assert.Equal(0, factory.Instances[0].Disposals);
        Assert.Equal(2, Value(await engine.ExecuteAsync(prompt, ["a"]), "a"));
        Assert.Equal(2, factory.Instances.Count);
    }

    [Fact]
    public async Task Removing_a_node_or_changing_its_type_retires_its_previous_object()
    {
        var first = new Factory("First"); var second = new Factory("Second"); using var engine = Engine(first, second);
        await engine.ExecuteAsync(Prompt(("a", "First")));
        await engine.ExecuteAsync(Prompt(("a", "Second")));
        Assert.Equal(1, first.Instances[0].Disposals);
        await engine.ExecuteAsync(Prompt(("a", "First")));
        Assert.Equal(1, second.Instances[0].Disposals); Assert.Equal(2, first.Instances.Count);
        await engine.ExecuteAsync(Prompt(("new", "First")));
        Assert.Equal(1, first.Instances[1].Disposals);
    }

    [Fact]
    public async Task Invalid_submission_does_not_clear_an_existing_object()
    {
        var factory = new Factory("Counter"); using var engine = Engine(factory);
        var prompt = Prompt(("a", "Counter"));
        await engine.ExecuteAsync(prompt);
        Assert.Equal("error", (await engine.ExecuteAsync(Prompt(("unknown", "Unavailable")))).Status);
        Assert.Equal(2, Value(await engine.ExecuteAsync(prompt), "a"));
        Assert.Single(factory.Instances); Assert.Equal(0, factory.Instances[0].Disposals);
    }

    [Fact]
    public async Task Lazy_selection_and_all_mapped_invocations_use_the_same_object()
    {
        var factory = new Factory("Counter", lazy: true); using var engine = Engine(factory);
        engine.Registry.Register(new Values());
        var prompt = Prompt(("a", "Counter"), ("values", "Values"));
        prompt["a"]!["inputs"]!["value"] = new JsonArray("values", 0);
        var result = await engine.ExecuteAsync(prompt, ["a"]);
        Assert.Equal(new[] { 1, 2, 3 }, Enumerable.Range(0, 3).Select(i => Value(result, "a", i)));
        var instance = Assert.Single(factory.Instances); Assert.Equal(1, instance.LazyCalls); Assert.Equal(3, instance.Calls);
    }

    [Fact]
    public async Task An_active_object_is_not_disposed_when_a_second_prompt_removes_it()
    {
        var factory = new Factory("Counter"); using var engine = Engine(factory);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var resume = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var first = engine.ExecuteAsync(Prompt(("a", "Counter")), onEvent: async e =>
        {
            if (e.Type == "executing") { entered.SetResult(); await resume.Task; }
        });
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        var active = Assert.Single(factory.Instances);
        Assert.Equal(1, Value(await engine.ExecuteAsync(Prompt(("b", "Counter"))), "b"));
        Assert.Equal(0, active.Disposals);
        resume.SetResult(); Assert.Equal(1, Value(await first.WaitAsync(TimeSpan.FromSeconds(5)), "a"));
        Assert.Equal(1, active.Disposals); Assert.Equal(0, factory.Instances[1].Disposals);
    }

    [Fact]
    public async Task Closing_engine_defers_disposal_until_active_work_finishes_and_prevents_new_work()
    {
        var factory = new Factory("Counter"); var engine = Engine(factory);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var resume = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        bool releasedAtTerminal = false;
        var task = engine.ExecuteAsync(Prompt(("a", "Counter")), onEvent: async e =>
        {
            if (e.Type == "executing") { entered.SetResult(); await resume.Task; }
            if (e.Type == "execution_success") releasedAtTerminal = factory.Instances[0].Disposals == 1;
        });
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5)); engine.Dispose();
        Assert.Equal(0, factory.Instances[0].Disposals);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => engine.ExecuteAsync(Prompt(("b", "Counter"))));
        resume.SetResult(); Assert.Equal(1, Value(await task.WaitAsync(TimeSpan.FromSeconds(5)), "a"));
        Assert.True(releasedAtTerminal); Assert.Equal(1, factory.Instances[0].Disposals);
        engine.Dispose(); Assert.Equal(1, factory.Instances[0].Disposals); Assert.Equal(0, factory.DefinitionDisposals);
    }

    [Fact]
    public async Task Cancelled_execution_keeps_the_object_but_does_not_replay_the_body()
    {
        var factory = new Factory("Counter"); using var engine = Engine(factory); using var cancel = new CancellationTokenSource();
        var prompt = Prompt(("a", "Counter"));
        var cancelled = await engine.ExecuteAsync(prompt, onEvent: e =>
        { if (e.Type == "executing") cancel.Cancel(); return ValueTask.CompletedTask; }, cancellationToken: cancel.Token);
        Assert.Equal("cancelled", cancelled.Status); Assert.Empty(cancelled.Outputs);
        Assert.Equal(0, Assert.Single(factory.Instances).Calls);
        Assert.Equal(1, Value(await engine.ExecuteAsync(prompt), "a")); Assert.Single(factory.Instances);
    }

    [Fact]
    public async Task Closing_cache_attempts_all_disposals_even_if_one_throws()
    {
        var factory = new Factory("Counter") { FailDispose = true }; var engine = Engine(factory);
        await engine.ExecuteAsync(Prompt(("a", "Counter"), ("b", "Counter")));
        var error = Assert.Throws<AggregateException>(engine.Dispose);
        Assert.Equal(2, error.InnerExceptions.Count); Assert.All(factory.Instances, i => Assert.Equal(1, i.Disposals));
        engine.Dispose(); Assert.Equal(0, factory.DefinitionDisposals);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Invalid_factory_returns_fail_without_transferring_definition_ownership(bool self)
    {
        var factory = new Factory("Counter") { ReturnDefinition = self, WrongType = !self }; using var engine = Engine(factory);
        var result = await engine.ExecuteAsync(Prompt(("a", "Counter")));
        Assert.Equal("error", result.Status); Assert.Empty(result.Outputs); Assert.Equal(0, factory.DefinitionDisposals);
        if (!self) Assert.Equal(1, Assert.Single(factory.Instances).Disposals);
    }

    private sealed class Factory(string type, bool lazy = false) : IRuntimeNodeFactory, IDisposable
    {
        public NodeSchema Schema { get; } = new(type, type, "test", [new("value", "*", Required: false, Lazy: lazy)], [new("INT")], OutputNode: true);
        public List<Instance> Instances { get; } = [];
        public int DefinitionDisposals { get; private set; }
        public bool FailDispose { get; init; }
        public bool ReturnDefinition { get; init; }
        public bool WrongType { get; init; }
        public IRuntimeNode CreateInstance()
        {
            if (ReturnDefinition) return this;
            var instance = new Instance(this, WrongType ? Schema with { ClassType = "Wrong" } : Schema, lazy);
            Instances.Add(instance); return instance;
        }
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) =>
            throw new InvalidOperationException("The test definition must not execute instead of its instance.");
        public void Dispose() => DefinitionDisposals++;
    }
    private sealed class Instance(Factory factory, NodeSchema schema, bool lazy) : IRuntimeNode, IDisposable
    {
        public NodeSchema Schema => schema;
        public int Calls { get; private set; }
        public int LazyCalls { get; private set; }
        public int Disposals { get; private set; }
        public IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> resolvedInputs)
        { LazyCalls++; Assert.Equal(0, Disposals); return lazy ? ["value"] : []; }
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        { Assert.Equal(0, Disposals); return ValueTask.FromResult(new NodeExecutionOutput([context.Json(++Calls)])); }
        public void Dispose() { Disposals++; if (factory.FailDispose) throw new InvalidOperationException("test disposal failure"); }
    }
    private sealed class Values : INode
    {
        public NodeSchema Schema { get; } = new("Values", "Values", "test", [], [new("*", IsList: true)]);
        public ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> inputs, CancellationToken cancellationToken) =>
            ValueTask.FromResult<IReadOnlyList<JsonNode?>>([new JsonArray(1, 2, 3)]);
    }
}
