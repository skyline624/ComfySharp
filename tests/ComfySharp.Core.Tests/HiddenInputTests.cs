using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class HiddenInputTests
{
    private static NodeSchema Schema(bool lists = false) => new("Capture", "Capture", "test", [], [new("*")],
        OutputNode: true, InputIsList: lists,
        HiddenInputs: [new("original", "PROMPT"), new("png", "EXTRA_PNGINFO"), new("id", "UNIQUE_ID")]);
    private static JsonObject Prompt() => JsonNode.Parse("""
        {"outer:7":{"class_type":"Capture","inputs":{},"_meta":{"title":"Saved image"}},
         "unselected":{"class_type":"Unavailable","inputs":{"value":42}}}
        """)!.AsObject();
    private static JsonObject Extra() => JsonNode.Parse("""
        {"extra_pnginfo":{"workflow":{"version":1,"unknown":{"keep":[1,"é",null]}},"tag":"first"},"client_id":"not-png"}
        """)!.AsObject();
    private static EngineService Engine(NodeSchema? schema = null,
        Func<IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>>, IReadOnlyCollection<string>>? lazy = null)
    {
        var registry = new NodeRegistry(); registry.Register(new Capture(schema ?? Schema(), lazy));
        return new(registry);
    }

    [Fact]
    public void Object_info_exposes_ordered_hidden_kinds_without_widgets()
    {
        var info = Engine().Registry.ToObjectInfo()["Capture"]!;
        Assert.Equal("{\"original\":\"PROMPT\",\"png\":\"EXTRA_PNGINFO\",\"id\":\"UNIQUE_ID\"}", info["input"]!["hidden"]!.ToJsonString());
        Assert.Equal("[\"original\",\"png\",\"id\"]", info["input_order"]!["hidden"]!.ToJsonString());
        Assert.Empty(info["input"]!["required"]!.AsObject());
        Assert.Empty(info["input"]!["optional"]!.AsObject());
    }

    [Theory]
    [InlineData("DYNPROMPT")]
    [InlineData("AUTH_TOKEN_COMFY_ORG")]
    [InlineData("unknown")]
    public void Unimplemented_hidden_kinds_are_rejected_at_registration(string kind) =>
        Assert.Throws<NotSupportedException>(() => Engine(Schema() with { HiddenInputs = [new("x", kind)] }));

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Ambiguous_hidden_names_are_rejected(int variant)
    {
        var schema = Schema() with
        {
            Inputs = [new("visible", "*")],
            HiddenInputs = variant switch
            {
                0 => [new("", "PROMPT")],
                1 => [new("visible", "PROMPT")],
                _ => [new("same", "PROMPT"), new("same", "UNIQUE_ID")]
            }
        };
        Assert.Throws<ArgumentException>(() => Engine(schema));
    }

    [Fact]
    public void Legacy_arguments_do_not_claim_V3_hidden_context() =>
        Assert.Throws<NotSupportedException>(() => Engine(Schema() with { V3ObjectInfo = true }));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Metadata_reaches_json_and_ui_boundaries_with_original_prompt_and_composed_id(bool ui)
    {
        var prompt = Prompt(); var extra = Extra(); var engine = Engine();
        JsonObject captured;
        if (ui)
        {
            var result = await engine.ExecuteUiAsync(prompt, ["outer:7"], extraData: extra);
            Assert.Equal("success", result.Status);
            captured = result.Outputs["outer:7"]["captures"]![0]!.AsObject();
        }
        else
        {
            var result = await engine.ExecuteAsync(prompt, ["outer:7"], extraData: extra);
            Assert.Equal("success", result.Status);
            captured = result.Outputs["outer:7"][0][0]!.AsObject();
        }
        Assert.True(JsonNode.DeepEquals(prompt, captured["original"]));
        Assert.True(JsonNode.DeepEquals(extra["extra_pnginfo"], captured["png"]));
        Assert.Equal("outer:7", captured["id"]!.GetValue<string>());
        Assert.False(captured.ContainsKey("client_id"));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("{}")]
    [InlineData("{\"extra_pnginfo\":null}")]
    [InlineData("{\"EXTRA_PNGINFO\":{\"wrong-case\":true}}")]
    public async Task Missing_png_metadata_is_present_as_null(string? json)
    {
        var result = await Engine().ExecuteAsync(Prompt(), ["outer:7"], extraData: json is null ? null : JsonNode.Parse(json)!.AsObject());
        Assert.Equal("success", result.Status);
        var value = result.Outputs["outer:7"][0][0]!.AsObject();
        Assert.True(value.ContainsKey("png")); Assert.Null(value["png"]);
    }

    [Fact]
    public async Task Caller_hidden_values_cannot_replace_job_metadata_or_create_dependencies()
    {
        var prompt = Prompt();
        prompt["outer:7"]!["inputs"] = JsonNode.Parse("""{"id":"spoofed","png":["missing",0],"original":null}""");
        var result = await Engine().ExecuteAsync(prompt, ["outer:7"], extraData: Extra());
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        var value = result.Outputs["outer:7"][0][0]!.AsObject();
        Assert.Equal(new[] { "id", "png", "original" }, value.Select(p => p.Key));
        Assert.Equal("outer:7", value["id"]!.GetValue<string>());
        Assert.Equal("first", value["png"]!["tag"]!.GetValue<string>());
        Assert.True(JsonNode.DeepEquals(prompt, value["original"]));
    }

    [Fact]
    public async Task Caller_mutation_after_first_await_cannot_change_prompt_or_metadata()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var resume = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var prompt = Prompt(); var extra = Extra();
        var beforePrompt = prompt.DeepClone(); var beforePng = extra["extra_pnginfo"]!.DeepClone();
        var task = Engine().ExecuteAsync(prompt, ["outer:7"], async e =>
        {
            if (e.Type == "execution_start") { entered.SetResult(); await resume.Task; }
        }, extraData: extra);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        prompt.Clear(); extra["extra_pnginfo"]!["workflow"] = "changed"; extra.Clear();
        resume.SetResult();
        var result = await task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal("success", result.Status);
        var captured = result.Outputs["outer:7"][0][0]!;
        Assert.True(JsonNode.DeepEquals(beforePrompt, captured["original"]));
        Assert.True(JsonNode.DeepEquals(beforePng, captured["png"]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Hidden_metadata_is_available_to_lazy_selection_and_obeys_input_list_wrapping(bool lists)
    {
        var calls = 0;
        var schema = Schema(lists) with { Inputs = [new("optional", "*", Required: false, Lazy: true)] };
        var engine = Engine(schema, resolved =>
        {
            calls++;
            Assert.Equal("outer:7", Assert.Single(resolved["id"]).ToJson()!.GetValue<string>());
            Assert.Equal("first", Assert.Single(resolved["png"]).ToJson()!["tag"]!.GetValue<string>());
            return ["optional"];
        });
        var prompt = Prompt(); prompt["outer:7"]!["inputs"]!["optional"] = 17;
        var result = await engine.ExecuteAsync(prompt, ["outer:7"], extraData: Extra());
        Assert.Equal("success", result.Status); Assert.Equal(1, calls);
        var captured = result.Outputs["outer:7"][0][0]!;
        Assert.Equal(lists ? "[\"outer:7\"]" : "\"outer:7\"", captured["id"]!.ToJsonString());
        Assert.Equal(lists ? "[17]" : "17", captured["optional"]!.ToJsonString());
        Assert.True(JsonNode.DeepEquals(Extra()["extra_pnginfo"], lists ? captured["png"]![0] : captured["png"]));
    }

    [Fact]
    public async Task Overlapping_jobs_on_one_engine_keep_separate_metadata()
    {
        var engine = Engine();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var resume = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var first = engine.ExecuteAsync(Prompt(), ["outer:7"], async e =>
        {
            if (e.Type == "executing") { entered.SetResult(); await resume.Task; }
        }, extraData: Extra());
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        var secondExtra = Extra(); secondExtra["extra_pnginfo"]!["tag"] = "second";
        var second = await engine.ExecuteAsync(Prompt(), ["outer:7"], extraData: secondExtra);
        resume.SetResult();
        var firstResult = await first.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal("success", firstResult.Status); Assert.Equal("success", second.Status);
        Assert.Equal("first", firstResult.Outputs["outer:7"][0][0]!["png"]!["tag"]!.GetValue<string>());
        Assert.Equal("second", second.Outputs["outer:7"][0][0]!["png"]!["tag"]!.GetValue<string>());
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(0, true)]
    [InlineData(3, false)]
    [InlineData(3, true)]
    public async Task Metadata_is_a_single_execution_value_when_other_inputs_produce_lists(int count, bool lists)
    {
        var registry = new NodeRegistry();
        registry.Register(new Values(count));
        registry.Register(new Capture(Schema(lists) with { Inputs = [new("values", "*")] }, null));
        var prompt = Prompt();
        prompt["source"] = JsonNode.Parse("""{"class_type":"Values","inputs":{}}""");
        prompt["outer:7"]!["inputs"]!["values"] = new JsonArray("source", 0);
        var result = await new EngineService(registry).ExecuteAsync(prompt, ["outer:7"], extraData: Extra());
        if (count == 0 && !lists)
        {
            // The singleton hidden value participates in mapping: an empty visible list has no last item.
            Assert.Equal("error", result.Status);
            Assert.Contains(result.Diagnostics, d => d.Message.Contains("empty execution list", StringComparison.Ordinal));
            Assert.Empty(result.Outputs);
            return;
        }
        Assert.Equal("success", result.Status);
        var captures = result.Outputs["outer:7"][0];
        Assert.Equal(lists ? 1 : count, captures.Count);
        for (int i = 0; i < captures.Count; i++)
        {
            var capture = captures[i]!;
            Assert.Equal(lists ? "[\"outer:7\"]" : "\"outer:7\"", capture["id"]!.ToJsonString());
            Assert.True(JsonNode.DeepEquals(prompt, lists ? capture["original"]![0] : capture["original"]));
            if (lists) Assert.Equal(Enumerable.Range(0, count), capture["values"]!.AsArray().Select(v => v!.GetValue<int>()));
            else Assert.Equal(i, capture["values"]!.GetValue<int>());
        }
    }

    private sealed class Values(int count) : IRuntimeNode
    {
        public NodeSchema Schema { get; } = new("Values", "Values", "test", [], [new("*", IsList: true)]);
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) =>
            ValueTask.FromResult(new NodeExecutionOutput([context.List(Enumerable.Range(0, count).Select(i => context.Json(i)))]));
    }

    private sealed class Capture(NodeSchema schema,
        Func<IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>>, IReadOnlyCollection<string>>? lazy) : IRuntimeNode
    {
        public NodeSchema Schema => schema;
        public IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> resolvedInputs) => lazy?.Invoke(resolvedInputs) ?? [];
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            var captured = new JsonObject(inputs.Select(p => new KeyValuePair<string, JsonNode?>(p.Key, p.Value.ToJson())));
            return ValueTask.FromResult(new NodeExecutionOutput([context.Json(captured)],
                new JsonObject { ["captures"] = new JsonArray(captured) }));
        }
    }
}
