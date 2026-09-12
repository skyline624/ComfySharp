using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>Managed contract tests. These probes exercise binding, not a port of StringFormat or source oracle.</summary>
public sealed class AutogrowNamesTests
{
    private static InputSchema Group(AutogrowNamesTemplate template, string name = "values") =>
        new(name, "COMFY_AUTOGROW_V3", Autogrow: template);
    private static NodeSchema Schema(AutogrowNamesTemplate template, bool list = false) =>
        new("NamesProbe", "NamesProbe", "test", [Group(template)], [new("*")], InputIsList: list, V3ObjectInfo: true);
    private static AutogrowNamesTemplate Names(int min = 0) => new(new("value", "*"), ["z", "a", "b"], min);
    private static JsonObject Prompt(string json) => JsonNode.Parse(json)!.AsObject();
    private static RuntimeNode Source(string name, Func<RuntimeNodeContext, RuntimeValue> produce, bool list = false) =>
        new(new(name, name, "test", [], [new("*", IsList: list)]), (c, _, _) => ValueTask.FromResult(new NodeExecutionOutput([produce(c)])));
    private static RuntimeNode Probe(AutogrowNamesTemplate template, bool list = false) =>
        new(Schema(template, list), (_, values, _) => ValueTask.FromResult(new NodeExecutionOutput([values["values"]])));

    [Fact]
    public void NamesAndPrototypeOptionsAreImmutableSnapshots()
    {
        var callerNames = new List<string> { "z", "a", "b" };
        var options = new JsonObject { ["tooltip"] = "original" };
        var template = new AutogrowNamesTemplate(new("value", "*", Options: options), callerNames);
        callerNames[0] = "changed"; callerNames.Clear(); options["tooltip"] = "changed";
        template.Input.Options!["tooltip"] = "getter change";
        Assert.Equal(new[] { "z", "a", "b" }, template.Names);
        Assert.Same(template.Names, template.MemberNames);
        Assert.Throws<NotSupportedException>(() => ((IList<string>)template.Names)[0] = "changed");
        var expanded = NodeInputExpansion.Expand(Schema(template), []);
        var leaves = expanded.Inputs;
        leaves[0].Options!["tooltip"] = "leaf change";
        Assert.Equal("leaf change", leaves[0].Options!["tooltip"]!.GetValue<string>());
        Assert.All(leaves.Skip(1), input => Assert.Equal("original", input.Options!["tooltip"]!.GetValue<string>()));
        Assert.Equal("original", template.Input.Options!["tooltip"]!.GetValue<string>());
        // Each Inputs access returns a fresh snapshot; a new expansion must also remain independent.
        Assert.All(expanded.Inputs, input => Assert.Equal("original", input.Options!["tooltip"]!.GetValue<string>()));
        Assert.All(NodeInputExpansion.Expand(Schema(template), []).Inputs,
            input => Assert.Equal("original", input.Options!["tooltip"]!.GetValue<string>()));
    }

    [Fact]
    public void TruncationPrecedesValidationAndDoesNotSortOrFoldNames()
    {
        var names = Enumerable.Range(0, 100).Select(i => "n" + i).Concat(["", "n0", "invalid.path"]).ToArray();
        var template = new AutogrowNamesTemplate(new("value", "*"), names, 101);
        Assert.Equal(100, template.Names.Count); Assert.Equal("n99", template.Names[^1]);
        Assert.All(NodeInputExpansion.Expand(Schema(template), []).Inputs, i => Assert.True(i.Required));
        var ordinal = new AutogrowNamesTemplate(new("value", "*"), ["z", "A", "a", "image_10", "image_2"]);
        Assert.Equal(new[] { "values.z", "values.A", "values.a", "values.image_10", "values.image_2" },
            NodeInputExpansion.Expand(Schema(ordinal), []).Inputs.Select(i => i.Name));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("a.b")]
    [InlineData("a-b")]
    [InlineData("é")]
    public void UnsupportedEffectiveNameFailsExplicitly(string? name)
    {
        var error = Assert.Throws<ArgumentException>(() => new AutogrowNamesTemplate(new("value", "*"), [name!]));
        Assert.Equal("names", error.ParamName); Assert.Contains("Unsupported Autogrow names", error.Message);
    }

    [Fact]
    public void DuplicateEffectiveNamesAreRejectedRatherThanDeduplicated()
    {
        Assert.Throws<ArgumentException>(() => new AutogrowNamesTemplate(new("value", "*"), ["a", "b", "a"], 1));
        var names = Enumerable.Range(0, 99).Select(i => "n" + i).Append("n0").ToArray();
        Assert.Throws<ArgumentException>(() => new AutogrowNamesTemplate(new("value", "*"), names));
        Assert.Throws<ArgumentNullException>(() => new AutogrowNamesTemplate(new("value", "*"), null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AutogrowNamesTemplate(new("value", "*"), [], -1));
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(3, 3)]
    [InlineData(5, 3)]
    public void RequiredPartitionFollowsNamePositionWithoutClampingMinimum(int min, int required)
    {
        var template = Names(min);
        var expanded = NodeInputExpansion.Expand(Schema(template), ["values.b", "values.z"]);
        Assert.Equal(min, template.Min);
        Assert.Equal(new[] { "values.z", "values.a", "values.b" }, expanded.Inputs.Select(i => i.Name));
        Assert.Equal(Enumerable.Range(0, 3).Select(i => i < required), expanded.Inputs.Select(i => i.Required));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(1000)]
    public async Task EmptyNamesAreValidAndBindOneEmptyMap(int min)
    {
        var template = new AutogrowNamesTemplate(new("value", "*"), [], min);
        Assert.Empty(NodeInputExpansion.Expand(Schema(template), []).Inputs);
        var registry = new NodeRegistry(); registry.Register(Probe(template));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""{"p":{"class_type":"NamesProbe","inputs":{}}}"""), ["p"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        Assert.Empty(Assert.Single(result.Outputs["p"][0]).Properties);
    }

    [Fact]
    public void OptionalPrototypeAndObjectInfoPreserveOriginalNamesTemplate()
    {
        var options = new JsonObject { ["template"] = new JsonObject { ["template_id"] = "type", ["allowed_types"] = "*" } };
        var template = new AutogrowNamesTemplate(new("item", "COMFY_MATCHTYPE_V3", Required: false, Options: options), ["z", "a"], 9);
        Assert.All(NodeInputExpansion.Expand(Schema(template), []).Inputs, i => Assert.False(i.Required));
        var registry = new NodeRegistry(); registry.Register(Probe(template));
        var info = registry.ToObjectInfo()["NamesProbe"]!;
        var actual = info["input"]!["required"]!["values"]![1]!["template"]!.AsObject();
        Assert.Equal(new[] { "input", "names", "min" }, actual.Select(p => p.Key));
        Assert.Equal(new[] { "z", "a" }, actual["names"]!.AsArray().Select(n => n!.GetValue<string>()));
        Assert.Equal(9, actual["min"]!.GetValue<int>());
        Assert.Empty(actual["input"]!["required"]!.AsObject());
        Assert.True(JsonNode.DeepEquals(actual["input"]!["optional"]!["item"]![1], options));
        Assert.False(actual.ContainsKey("max")); Assert.False(actual.ContainsKey("prefix"));
        actual["names"]![0] = "mutated";
        Assert.Equal("z", NodeInputExpansion.TemplateOptions(Group(template))["template"]!["names"]![0]!.GetValue<string>());
    }

    [Theory]
    [InlineData("force_input")]
    [InlineData("default")]
    [InlineData("rawLink")]
    public void UnqualifiedPrototypeAndGroupOptionsAreRejected(string option)
    {
        Assert.Throws<ArgumentException>(() => new AutogrowNamesTemplate(new("value", "*", Options: new() { [option] = true }), ["a"]));
        var group = Group(Names()) with { Options = new() { [option] = true } };
        var error = Assert.Throws<NodeInputExpansionException>(() => NodeInputExpansion.TemplateOptions(group));
        Assert.Equal("unsupported_dynamic_template", error.Code);
    }

    [Fact]
    public void NestedWidgetsLazyAndOptionalGroupsRemainOutsideTheContract()
    {
        Assert.Throws<ArgumentException>(() => new AutogrowNamesTemplate(new("value", "STRING"), ["a"]));
        Assert.Throws<ArgumentException>(() => new AutogrowNamesTemplate(new("value", "*", Lazy: true), ["a"]));
        Assert.Throws<ArgumentException>(() => new AutogrowNamesTemplate(Group(Names()), ["a"]));
        Assert.Throws<ArgumentException>(() => new AutogrowNamesTemplate(new("value", "COMFY_MATCHTYPE_V3"), ["a"]));
        foreach (var group in new[] { Group(Names()) with { Required = false }, Group(Names()) with { Lazy = true } })
            Assert.Equal("unsupported_dynamic_template", Assert.Throws<NodeInputExpansionException>(() => NodeInputExpansion.TemplateOptions(group)).Code);
        var mixed = Schema(Names()) with { Inputs = [Group(Names()), new("static", "*", Lazy: true)] };
        Assert.Equal("unsupported_dynamic_template", Assert.Throws<NodeInputExpansionException>(() => NodeInputExpansion.Expand(mixed, [])).Code);
    }

    [Theory]
    [InlineData("values.A")]
    [InlineData("values.a.child")]
    [InlineData("values.unknown")]
    [InlineData("values")]
    public async Task UnknownFlatNamesAreRejectedBeforeAnyProducerRuns(string extra)
    {
        int calls = 0; var registry = new NodeRegistry(); registry.Register(Probe(Names()));
        registry.Register(Source("Source", c => { calls++; return c.Json(null); }));
        var prompt = Prompt("""{"s":{"class_type":"Source","inputs":{}},"p":{"class_type":"NamesProbe","inputs":{"values.z":["s",0]}}}""");
        prompt["p"]!["inputs"]![extra] = new JsonArray("s", 0);
        using var result = await new EngineService(registry).ExecuteValuesAsync(prompt, ["p"]);
        Assert.Equal("error", result.Status); Assert.Equal(0, calls);
        Assert.Contains(result.Diagnostics, d => d.Code == "unsupported_dynamic_input" && d.InputName == extra);
    }

    [Fact]
    public void StaticLeafCollisionsAndDuplicateGroupsAreRejected()
    {
        foreach (var inputs in new IReadOnlyList<InputSchema>[]
        {
            [Group(Names()), new("values.a", "*")], [new("values.a", "*"), Group(Names())], [Group(Names()), Group(Names())]
        })
            Assert.Equal("invalid_dynamic_template", Assert.Throws<NodeInputExpansionException>(() =>
                NodeInputExpansion.Expand(Schema(Names()) with { Inputs = inputs }, [])).Code);
    }

    [Fact]
    public void RequiredMissingIsDistinctFromExplicitNullAndOptionalOmission()
    {
        var registry = new NodeRegistry(); registry.Register(Probe(Names(1))); var engine = new EngineService(registry);
        var missing = engine.Validate(Prompt("""{"p":{"class_type":"NamesProbe","inputs":{"values.b":null}}}"""), ["p"]);
        Assert.False(missing.IsValid);
        Assert.Contains(missing.Diagnostics, d => d.Code == "required_input_missing" && d.InputName == "values.z");
        var explicitNull = engine.Validate(Prompt("""{"p":{"class_type":"NamesProbe","inputs":{"values.z":null}}}"""), ["p"]);
        Assert.True(explicitNull.IsValid); Assert.Empty(explicitNull.Diagnostics);
    }

    [Fact]
    public async Task ScalarMappingUsesNamedOrderHolesRepeatLastAndStaticArguments()
    {
        var registry = new NodeRegistry();
        registry.Register(Source("Long", c => c.List([c.Json(JsonValue.Create(1)), c.Json(JsonValue.Create(2))]), true));
        registry.Register(Source("Short", c => c.List([c.Json(JsonValue.Create(10))]), true));
        var schema = Schema(Names()) with { Inputs = [Group(Names()), new("label", "STRING"), Group(new(new("v", "*"), ["q"], 0), "other")] };
        var seen = new List<string>();
        registry.Register(new RuntimeNode(schema, (c, args, _) =>
        {
            Assert.Equal(new[] { "label", "values", "other" }, args.Keys);
            Assert.Equal(new[] { "z", "b" }, args["values"].Properties.Keys);
            Assert.All(args["values"].Properties.Values, v => Assert.Equal(RuntimeValueKind.Json, v.Kind));
            Assert.Empty(args["other"].Properties);
            seen.Add(args["label"].ToJson()!.GetValue<string>());
            return ValueTask.FromResult(new NodeExecutionOutput([c.Json(args["values"].ToJson())]));
        }));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"l":{"class_type":"Long","inputs":{}},"s":{"class_type":"Short","inputs":{}},
             "p":{"class_type":"NamesProbe","inputs":{"values.b":["s",0],"label":"row","values.z":["l",0]}}}
            """), ["p"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        Assert.Equal(new[] { "row", "row" }, seen);
        Assert.Equal(new[] { "{\"z\":1,\"b\":10}", "{\"z\":2,\"b\":10}" }, result.Outputs["p"][0].Select(v => v.ToJson()!.ToJsonString()));
    }

    [Fact]
    public async Task InputIsListBindsWholeExecutionListsWithoutFlatteningLiteralItems()
    {
        var registry = new NodeRegistry(); registry.Register(Probe(Names(), true));
        registry.Register(Source("Items", c => c.List([c.Json(new JsonArray(1, 2)), c.Json(null)]), true));
        registry.Register(Source("Empty", c => c.List([]), true));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"s":{"class_type":"Items","inputs":{}},"e":{"class_type":"Empty","inputs":{}},
             "p":{"class_type":"NamesProbe","inputs":{"values.b":["e",0],"values.z":["s",0]}}}
            """), ["p"]);
        Assert.Equal("success", result.Status);
        var map = Assert.Single(result.Outputs["p"][0]).Properties;
        Assert.Equal(new[] { "z", "b" }, map.Keys); Assert.Empty(map["b"].Items);
        Assert.Equal(new[] { "[1,2]", "null" }, map["z"].Items.Select(v => v.ToJson()?.ToJsonString() ?? "null"));
    }

    [Fact]
    public async Task EmptyMappedListKeepsItsLivePathWithNullValue()
    {
        var registry = new NodeRegistry(); registry.Register(Probe(Names(1)));
        registry.Register(Source("Empty", c => c.List([]), true));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"s":{"class_type":"Empty","inputs":{}},"p":{"class_type":"NamesProbe","inputs":{"values.z":["s",0]}}}
            """), ["p"]);
        Assert.Equal("success", result.Status);
        var map = Assert.Single(result.Outputs["p"][0]).Properties;
        Assert.Equal(new[] { "z" }, map.Keys); Assert.Null(map["z"].ToJson());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FlatBlockerSelectionPrecedesGroupingInPromptOrder(bool list)
    {
        int calls = 0; var registry = new NodeRegistry();
        registry.Register(new RuntimeNode(Schema(Names(), list), (_, _, _) => { calls++; throw new InvalidOperationException("Must not execute"); }));
        registry.Register(Source("Z", c => c.Blocker("schema-first")));
        registry.Register(Source("B", c => c.Blocker("prompt-first")));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"z":{"class_type":"Z","inputs":{}},"b":{"class_type":"B","inputs":{}},
             "p":{"class_type":"NamesProbe","inputs":{"values.b":["b",0],"values.z":["z",0]}}}
            """), ["p"]);
        Assert.Equal(0, calls); Assert.Equal("Execution Blocked: prompt-first", Assert.Single(result.Diagnostics).Message);
        Assert.Null(Assert.Single(result.Outputs["p"][0]).Blocker.Message);
    }

    [Fact]
    public async Task NestedBlockersAreKeptAsValuesWithoutRecursiveConsumption()
    {
        var registry = new NodeRegistry(); registry.Register(Probe(Names()));
        registry.Register(Source("Container", c => c.Map(new Dictionary<string, RuntimeValue> { ["inner"] = c.Blocker("nested") })));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"s":{"class_type":"Container","inputs":{}},"p":{"class_type":"NamesProbe","inputs":{"values.a":["s",0]}}}
            """), ["p"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        Assert.Equal("nested", result.Outputs["p"][0][0].Properties["a"].Properties["inner"].Blocker.Message);
    }

    [Fact]
    public async Task AliasedMembersAndFanoutSurviveUntilLastIndependentOwner()
    {
        var resource = new CountingResource(); int calls = 0;
        var registry = new NodeRegistry(); registry.Register(Probe(Names()));
        registry.Register(Source("Owned", c => { calls++; return c.Own(resource); }));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"s":{"class_type":"Owned","inputs":{}},"p":{"class_type":"NamesProbe","inputs":{"values.z":["s",0],"values.b":["s",0]}},
             "q":{"class_type":"NamesProbe","inputs":{"values.a":["p",0]}}}
            """), ["p", "q"]);
        Assert.Equal("success", result.Status); Assert.Equal(1, calls);
        using var kept = result.Outputs["q"][0][0].Properties["a"].Properties["b"].Retain();
        result.Dispose(); Assert.Equal(0, resource.Disposals); Assert.Same(resource, kept.GetNative<CountingResource>());
        kept.Dispose(); Assert.Equal(1, resource.Disposals);
    }

    [Fact]
    public void BindingFailureReleasesPartialRetainsWithoutStealingCallerOwnership()
    {
        var resource = new CountingResource(); using var caller = new RuntimeNodeContext(); using var invocation = new RuntimeNodeContext();
        var first = caller.Own(resource); var dead = caller.Json(null); dead.Dispose();
        var expanded = NodeInputExpansion.Expand(Schema(Names()), ["values.z", "values.b"]);
        Assert.Throws<ObjectDisposedException>(() => expanded.BindArguments(invocation,
            new Dictionary<string, RuntimeValue> { ["values.z"] = first, ["values.b"] = dead }));
        invocation.Dispose(); Assert.Equal(0, resource.Disposals); Assert.Same(resource, first.GetNative<CountingResource>());
        caller.Dispose(); Assert.Equal(1, resource.Disposals);
    }

    [Fact]
    public async Task CancellationBeforeGroupingReleasesResolvedProducerAndSkipsBody()
    {
        var resource = new CountingResource(); using var cancelled = new CancellationTokenSource();
        var registry = new NodeRegistry(); int calls = 0;
        registry.Register(Source("Owned", c => c.Own(resource)));
        registry.Register(new RuntimeNode(Schema(Names()), (_, _, _) => { calls++; throw new InvalidOperationException("Must not execute"); }));
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"s":{"class_type":"Owned","inputs":{}},"p":{"class_type":"NamesProbe","inputs":{"values.a":["s",0]}}}
            """), ["p"], e => { if (e.Type == "executing" && e.NodeId == "p") cancelled.Cancel(); return ValueTask.CompletedTask; }, cancelled.Token);
        Assert.Equal("cancelled", result.Status); Assert.Empty(result.Outputs); Assert.Equal(0, calls); Assert.Equal(1, resource.Disposals);
    }

    [Fact]
    public void CancelledBinderDoesNotAcquireAnyCallerResource()
    {
        var resource = new CountingResource(); using var caller = new RuntimeNodeContext(); using var invocation = new RuntimeNodeContext();
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        var expanded = NodeInputExpansion.Expand(Schema(Names()), ["values.z"]);
        Assert.ThrowsAny<OperationCanceledException>(() => expanded.BindArguments(invocation,
            new Dictionary<string, RuntimeValue> { ["values.z"] = caller.Own(resource) }, cancelled.Token));
        caller.Dispose(); Assert.Equal(1, resource.Disposals); invocation.Dispose(); Assert.Equal(1, resource.Disposals);
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
