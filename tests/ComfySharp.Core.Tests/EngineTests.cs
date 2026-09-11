using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class EngineTests
{
    private static JsonObject Prompt(string text) => JsonNode.Parse(text)!.AsObject();
    private static EngineService Engine() => new(BuiltInNodes.CreateRegistry());
    private static JsonNode? Value(ExecutionResult result, string target = "1") => result.Outputs[target][0][0];

    [Theory]
    [InlineData("{}", "missing_node")]
    [InlineData("{\"1\":{\"class_type\":\"CheckpointLoaderSimple\",\"inputs\":{}}}", "unknown_node")]
    [InlineData("{\"1\":{\"class_type\":\"PrimitiveString\",\"inputs\":{}}}", "required_input_missing")]
    [InlineData("{\"1\":{\"class_type\":\"PrimitiveInt\",\"inputs\":{\"value\":\"abc\"}}}", "invalid_input_type")]
    [InlineData("{\"1\":{\"class_type\":\"PrimitiveString\",\"inputs\":{\"value\":[\"1\",0]}}}", "cycle")]
    [InlineData("{\"1\":{\"class_type\":\"PrimitiveString\",\"inputs\":{\"value\":[\"2\",0]}}}", "missing_node")]
    [InlineData("{\"1\":{\"class_type\":\"PrimitiveString\",\"inputs\":{\"value\":[\"2\",0.5]}}}", "invalid_link")]
    public void Invalid_graphs_are_diagnosed_before_execution(string json, string code)
    {
        var result = Engine().Validate(Prompt(json), ["1"]);
        Assert.False(result.IsValid);
        Assert.Contains(result.Diagnostics, d => d.Code == code && d.TargetId == "1");
    }

    [Fact]
    public void Linked_types_and_output_indices_are_checked()
    {
        var prompt = Prompt("""{"1":{"class_type":"PrimitiveInt","inputs":{"value":3}},"2":{"class_type":"StringLength","inputs":{"string":["1",0]}},"3":{"class_type":"PrimitiveInt","inputs":{"value":["1",1]}}}""");
        var result = Engine().Validate(prompt, ["2", "3"]);
        Assert.Contains(result.Diagnostics, d => d.Code == "type_mismatch");
        Assert.Contains(result.Diagnostics, d => d.Code == "invalid_output_index");
    }

    [Fact]
    public void Integer_bounds_preserve_precision_at_int64_limit()
    {
        var result = Engine().Validate(Prompt("""{"1":{"class_type":"PrimitiveInt","inputs":{"value":-9223372036854775808}}}"""), ["1"]);
        Assert.False(result.IsValid);
        Assert.Contains(result.Diagnostics, d => d.Code == "invalid_input_type");
    }

    [Fact]
    public async Task Default_targets_execute_only_declared_output_nodes()
    {
        var registry = new NodeRegistry();
        registry.Register(new OutputTestNode());
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""{"1":{"class_type":"TestOutput","inputs":{}},"unrelated":{"class_type":"Unknown","inputs":{}}}"""));
        Assert.Equal("success", result.Status);
        Assert.Equal("output", Value(result)!.GetValue<string>());
    }

    [Fact]
    public async Task Invalid_independent_target_does_not_prevent_valid_output()
    {
        var result = await Engine().ExecuteAsync(Prompt("""{"1":{"class_type":"PrimitiveString","inputs":{"value":"ok"}},"2":{"class_type":"Missing","inputs":{}}}"""), ["1", "2"]);
        Assert.Equal("success", result.Status);
        Assert.Equal("ok", Value(result)!.GetValue<string>());
        Assert.Contains(result.Diagnostics, d => d.Code == "unknown_node" && d.TargetId == "2");
    }

    [Fact]
    public void No_fake_output_nodes_are_registered()
    {
        var engine = Engine();
        Assert.All(engine.Registry.Nodes, n => Assert.False(n.Schema.OutputNode));
        Assert.Contains(engine.Validate(Prompt("""{"1":{"class_type":"PrimitiveInt","inputs":{"value":1}}}""")).Diagnostics, d => d.Code == "no_outputs");
    }

    [Fact]
    public async Task Wrapped_link_shaped_array_is_a_literal_and_prompt_remains_unchanged()
    {
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new TestNode("Echo", [new("value", "*")], [new("*")], i => [i["value"]]));
        var prompt = Prompt("""{"1":{"class_type":"Echo","inputs":{"value":{"__value__":["absent",0]}}}}""");
        var original = prompt.ToJsonString();
        var result = await new EngineService(registry).ExecuteAsync(prompt, ["1"]);
        Assert.Equal("success", result.Status);
        Assert.Equal("[\"absent\",0]", Value(result)!.ToJsonString());
        Assert.Equal(original, prompt.ToJsonString());
    }

    [Fact]
    public async Task List_outputs_flatten_and_mapping_repeats_last_item()
    {
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new TestNode("List", [new("items", "ARRAY")], [new("STRING", IsList: true)], i => [i["items"]]));
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""
            {"a":{"class_type":"List","inputs":{"items":{"__value__":["a","b","c"]}}},
             "b":{"class_type":"List","inputs":{"items":{"__value__":["x","y"]}}},
             "1":{"class_type":"StringConcatenate","inputs":{"string_a":["a",0],"string_b":["b",0],"delimiter":"-"}}}
            """), ["1"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(new[] { "a-x", "b-y", "c-y" }, result.Outputs["1"][0].Select(v => v!.GetValue<string>()));
    }

    [Fact]
    public async Task Input_is_list_receives_the_whole_execution_list_once()
    {
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new TestNode("List", [], [new("INT", IsList: true)], _ => [new JsonArray(1, 2, 3)]));
        registry.Register(new TestNode("Count", [new("value", "INT")], [new("INT")], i => [JsonValue.Create(i["value"]!.AsArray().Count)], inputIsList: true));
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""{"a":{"class_type":"List","inputs":{}},"1":{"class_type":"Count","inputs":{"value":["a",0]}}}"""), ["1"]);
        Assert.Equal(3, Value(result)!.GetValue<int>());
        Assert.Single(result.Outputs["1"][0]);
    }

    [Fact]
    public async Task Empty_list_output_is_preserved_for_list_consumers()
    {
        var registry = new NodeRegistry();
        registry.Register(new TestNode("Empty", [], [new("INT", IsList: true)], _ => [new JsonArray()]));
        registry.Register(new TestNode("Count", [new("value", "INT")], [new("INT")], i => [JsonValue.Create(i["value"]!.AsArray().Count)], inputIsList: true));
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""{"a":{"class_type":"Empty","inputs":{}},"1":{"class_type":"Count","inputs":{"value":["a",0]}}}"""), ["1"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(0, Value(result)!.GetValue<int>());
    }

    [Fact]
    public async Task List_output_slots_flatten_across_mapped_invocations()
    {
        var registry = new NodeRegistry();
        registry.Register(new TestNode("List", [], [new("INT", IsList: true)], _ => [new JsonArray(1, 2)]));
        registry.Register(new TestNode("Double", [new("value", "INT")], [new("INT", IsList: true)], i => [new JsonArray(i["value"]!.DeepClone(), i["value"]!.DeepClone())]));
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""{"a":{"class_type":"List","inputs":{}},"1":{"class_type":"Double","inputs":{"value":["a",0]}}}"""), ["1"]);
        Assert.Equal(new[] { 1, 1, 2, 2 }, result.Outputs["1"][0].Select(v => v!.GetValue<int>()));
    }

    [Fact]
    public async Task Lazy_switch_never_invokes_unselected_branch()
    {
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new TestNode("Throw", [], [new("STRING")], _ => throw new InvalidOperationException("Should not run")));
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""
            {"bad":{"class_type":"Throw","inputs":{}},"1":{"class_type":"ComfySwitchNode","inputs":{"switch":true,"on_true":"selected","on_false":["bad",0]}}}
            """), ["1"]);
        Assert.Equal("success", result.Status);
        Assert.Equal("selected", Value(result)!.GetValue<string>());
    }

    [Fact]
    public async Task Missing_optional_switch_branch_returns_null()
    {
        var result = await Engine().ExecuteAsync(Prompt("""{"1":{"class_type":"ComfySwitchNode","inputs":{"switch":false,"on_true":"unused"}}}"""), ["1"]);
        Assert.Equal("success", result.Status);
        Assert.Null(Value(result));
    }

    [Fact]
    public async Task Cancellation_is_cooperative_and_does_not_poison_next_run()
    {
        var registry = BuiltInNodes.CreateRegistry();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        registry.Register(new AsyncNode(entered));
        var engine = new EngineService(registry);
        using var cancellation = new CancellationTokenSource();
        var events = new List<string>();
        var running = engine.ExecuteAsync(Prompt("""{"1":{"class_type":"Wait","inputs":{}}}"""), ["1"], e => { events.Add(e.Type); return ValueTask.CompletedTask; }, cancellation.Token);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        cancellation.Cancel();
        Assert.Equal("cancelled", (await running.WaitAsync(TimeSpan.FromSeconds(5))).Status);
        Assert.Contains("execution_interrupted", events);
        Assert.DoesNotContain("execution_success", events);
        var next = await engine.ExecuteAsync(Prompt("""{"1":{"class_type":"PrimitiveInt","inputs":{"value":7}}}"""), ["1"]);
        Assert.Equal("success", next.Status);
    }

    [Fact]
    public async Task Shared_dependency_runs_once_per_job_and_changed_inputs_are_not_stale()
    {
        var calls = 0;
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new TestNode("Counted", [new("value", "STRING")], [new("STRING")], i => { calls++; return [i["value"]]; }));
        var engine = new EngineService(registry);
        var prompt = Prompt("""{"a":{"class_type":"Counted","inputs":{"value":"a"}},"1":{"class_type":"StringConcatenate","inputs":{"string_a":["a",0],"string_b":["a",0],"delimiter":""}}}""");
        Assert.Equal("aa", Value(await engine.ExecuteAsync(prompt, ["1"]))!.GetValue<string>());
        Assert.Equal(1, calls);
        prompt["a"]!["inputs"]!["value"] = "b";
        Assert.Equal("bb", Value(await engine.ExecuteAsync(prompt, ["1"]))!.GetValue<string>());
        Assert.Equal(2, calls);
    }

    [Fact]
    public async Task Consumer_cannot_mutate_shared_literal_from_another_branch()
    {
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new TestNode("Array", [], [new("ARRAY")], _ => [new JsonArray("original")]));
        registry.Register(new TestNode("Mutate", [new("value", "ARRAY")], [new("ARRAY")], i => { i["value"]![0] = "changed"; return [i["value"]]; }));
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""{"a":{"class_type":"Array","inputs":{}},"1":{"class_type":"Mutate","inputs":{"value":["a",0]}}}"""), ["1", "a"]);
        Assert.Equal("original", result.Outputs["a"][0][0]![0]!.GetValue<string>());
    }

    [Fact]
    public async Task Lazy_hook_cannot_mutate_nested_memo_values_or_execution_list_containers()
    {
        var registry = new NodeRegistry();
        registry.Register(new TestNode("Object", [], [new("DICT")], _ =>
            [new JsonObject { ["nested"] = new JsonObject { ["field"] = "original" } }]));
        registry.Register(new MutatingLazyHookNode());
        registry.Register(new TestNode("Read", [new("value", "DICT")], [new("STRING")], i => [i["value"]!["nested"]!["field"]]));
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""
            {"source":{"class_type":"Object","inputs":{}},
             "mutator":{"class_type":"MutatingLazyHook","inputs":{"value":["source",0]}},
             "reader":{"class_type":"Read","inputs":{"value":["source",0]}}}
            """), ["mutator", "reader", "source"]);
        Assert.Equal("success", result.Status);
        Assert.Equal("original", Value(result, "mutator")!.GetValue<string>());
        Assert.Equal("original", Value(result, "reader")!.GetValue<string>());
        Assert.Equal("original", Value(result, "source")!["nested"]!["field"]!.GetValue<string>());
    }

    [Fact]
    public async Task Execution_error_has_node_identity_and_other_targets_still_execute()
    {
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new TestNode("Fail", [], [new("STRING")], _ => throw new InvalidOperationException("broken")));
        var result = await new EngineService(registry).ExecuteAsync(Prompt("""{"bad":{"class_type":"Fail","inputs":{}},"1":{"class_type":"PrimitiveString","inputs":{"value":"ok"}}}"""), ["bad", "1"]);
        Assert.Equal("error", result.Status);
        Assert.Equal("ok", Value(result)!.GetValue<string>());
        Assert.Contains(result.Diagnostics, d => d.NodeId == "bad" && d.Message == "broken" && d.Code == "execution_error");
    }

    [Fact]
    public async Task Snapshot_prevents_caller_changes_during_events()
    {
        var prompt = Prompt("""{"1":{"class_type":"PrimitiveString","inputs":{"value":"before"}}}""");
        var result = await Engine().ExecuteAsync(prompt, ["1"], _ => { prompt["1"]!["inputs"]!["value"] = "after"; return ValueTask.CompletedTask; });
        Assert.Equal("before", Value(result)!.GetValue<string>());
    }

    private sealed class TestNode(string id, InputSchema[] inputs, OutputSchema[] outputs,
        Func<IReadOnlyDictionary<string, JsonNode?>, IReadOnlyList<JsonNode?>> execute, bool inputIsList = false) : INode
    {
        public NodeSchema Schema { get; } = new(id, id, "test", inputs, outputs, InputIsList: inputIsList);
        public ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> values, CancellationToken cancellationToken) => ValueTask.FromResult(execute(values));
    }
    private sealed class AsyncNode(TaskCompletionSource entered) : INode
    {
        public NodeSchema Schema { get; } = new("Wait", "Wait", "test", [], [new("STRING")]);
        public async ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> inputs, CancellationToken cancellationToken)
        {
            entered.SetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            return [JsonValue.Create("impossible")];
        }
    }
    private sealed class OutputTestNode : INode
    {
        public NodeSchema Schema { get; } = new("TestOutput", "Test output", "test", [], [new("STRING")], OutputNode: true);
        public ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> inputs, CancellationToken cancellationToken) => ValueTask.FromResult<IReadOnlyList<JsonNode?>>([JsonValue.Create("output")]);
    }
    private sealed class MutatingLazyHookNode : INode
    {
        public NodeSchema Schema { get; } = new("MutatingLazyHook", "Mutating hook", "test", [new("value", "DICT")], [new("STRING")]);
        public IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<JsonNode?>> resolvedInputs)
        {
            var values = resolvedInputs["value"];
            values[0]!["nested"]!["field"] = "mutated";
            if (values is IList<JsonNode?> writable) writable[0] = new JsonObject { ["nested"] = new JsonObject { ["field"] = "replaced" } };
            return [];
        }
        public ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> inputs, CancellationToken cancellationToken) =>
            ValueTask.FromResult<IReadOnlyList<JsonNode?>>([inputs["value"]!["nested"]!["field"]]);
    }
}
