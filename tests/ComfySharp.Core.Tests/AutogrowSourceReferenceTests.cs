using System.Security.Cryptography;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>
/// Exact observations from the published source collector, restricted to its 18 comparable cases.
/// The other 18 observations remain in the raw fixture; required-input bypass, unsupported
/// templates and source-only constructor behavior are not accepted engine behavior.
/// Source cache records become real typed producer outputs. A literal JSON array is explicitly
/// wrapped in __value__ for the validated prompt grammar (the source also supports this wrapper).
/// The collector observes get_input_data without PromptExecutor validation: execution values,
/// not raw HTTP acceptance or bytes, are compared.
/// Cache infrastructure, hidden V3 bookkeeping and source dynamic_paths have no public engine
/// equivalents. Their observable grouping is checked through real BindArguments and node calls.
/// </summary>
public sealed class AutogrowSourceReferenceTests
{
    private const string ReferenceSha = "04f2f2d1cea2a66460a832d471f751fcf3effd45ad4bc36a6508d540a02d2924";
    private const string ProtocolSha = "772b018f80b35dd7e1d4f4655ab6d58ccbc62c34fcc915f5a93e9cb8ad1d164b";
    private static readonly string[] ComparableIds =
    [
        "one-port", "concat-two", "reverse-gap", "ten-ports", "empty-execution-list",
        "literal-empty-item", "nested-items", "duplicate-link", "blocker-reverse-order",
        "blocker-silent-first", "blocker-empty-message", "nested-blocker-item",
        "prefix-min-zero-empty", "prefix-max-one", "prefix-max-hundred", "prefix-min-over-max",
        "optional-prototype", "different-prefix"
    ];

    [Theory]
    [InlineData("one-port")]
    [InlineData("concat-two")]
    [InlineData("reverse-gap")]
    [InlineData("ten-ports")]
    [InlineData("empty-execution-list")]
    [InlineData("literal-empty-item")]
    [InlineData("nested-items")]
    [InlineData("duplicate-link")]
    [InlineData("blocker-reverse-order")]
    [InlineData("blocker-silent-first")]
    [InlineData("blocker-empty-message")]
    [InlineData("nested-blocker-item")]
    [InlineData("prefix-min-zero-empty")]
    [InlineData("prefix-max-one")]
    [InlineData("prefix-max-hundred")]
    [InlineData("prefix-min-over-max")]
    [InlineData("optional-prototype")]
    [InlineData("different-prefix")]
    public async Task PublishedSourceMetadataGroupingCallsAndOutputsMatchRealEngine(string id)
    {
        var reference = Resource("autogrow.reference.json", 279905, ReferenceSha);
        var protocol = Resource("autogrow.protocol.json", 39714, ProtocolSha);
        Assert.True(reference["executed"]!.GetValue<bool>());
        Assert.Equal("ae241a0ae32f4e053f0bc5754e8d4d0ebfc7d5f4", reference["collectorCommit"]!.GetValue<string>());
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", reference["backendCommit"]!.GetValue<string>());
        Assert.Equal(ProtocolSha, reference["protocolSha256"]!.GetValue<string>());
        Assert.Equal("autogrow-prefix-createlist-v1", reference["protocolId"]!.GetValue<string>());
        Assert.Equal(9, reference["sourceFiles"]!.AsArray().Count);
        Assert.Equal(72, reference["sourceFiles"]!.AsArray().Sum(s => s!["declarations"]!.AsArray().Count));
        var observations = reference["referenceOutputs"]!.AsArray();
        Assert.Equal(36, observations.Count);
        Assert.Equal(ComparableIds, observations.Where(c => c!["portClassification"]!.GetValue<string>() == "comparable")
            .Select(c => c!["id"]!.GetValue<string>()));
        Assert.Equal(36, protocol["cases"]!.AsArray().Count);
        var specification = Assert.Single(protocol["cases"]!.AsArray(), c => c!["id"]!.GetValue<string>() == id)!.AsObject();
        var expected = Assert.Single(observations, c => c!["id"]!.GetValue<string>() == id)!.AsObject();
        Assert.Equal("comparable", specification["portClassification"]!.GetValue<string>());
        Assert.Empty(specification["permittedSourceErrors"]!.AsArray());
        Assert.Equal("returned", expected["status"]!.GetValue<string>());
        Equal(specification["promptInputs"], expected["inputsAfter"]);
        Equal(specification["cache"], expected["cacheAfter"]);
        Equal(new JsonArray("off", "on", "off"), expected["repeats"]!["sequence"]);
        Assert.True(expected["repeats"]!["bitIdentical"]!.GetValue<bool>());
        var repeatHashes = expected["repeats"]!["sha256"]!.AsArray().Select(h => h!.GetValue<string>()).ToArray();
        Assert.Equal(3, repeatHashes.Length);
        Assert.Single(repeatHashes.Distinct(StringComparer.Ordinal));

        string specificationBefore = specification.ToJsonString();
        var promptInputs = specification["promptInputs"]!.AsObject();
        var cache = specification["cache"]!.AsObject();
        var create = new CreateListNode();
        var schema = Schema(specification, create.Schema);
        var expanded = NodeInputExpansion.Expand(schema, promptInputs.Select(p => p.Key));
        CompareFinalized(expected, expanded.Inputs);
        var calls = new JsonArray();
        var nested = new JsonArray();
        var events = new List<EngineEvent>();
        var producerCalls = new Dictionary<string, int>(StringComparer.Ordinal);
        var registry = new NodeRegistry();
        registry.Register(new Node(schema, (context, inputs, token) =>
        {
            // Borrow only during the callback. Snapshots contain no RuntimeValue wrappers.
            var group = inputs["inputs"].Properties;
            calls.Add(new JsonObject { ["inputs"] = Encode(inputs["inputs"]), ["order"] = Names(group.Keys) });
            nested.Add(new JsonObject { ["values"] = EncodeInputs(inputs),
                ["rootOrder"] = Names(inputs.Keys), ["groupOrder"] = Names(group.Keys) });
            return create.ExecuteAsync(context, inputs, token);
        }));
        var info = registry.ToObjectInfo()[schema.ClassType]!;
        Equal(expected["objectInfo"], info);
        Equal(expected["inputTypes"], info["input"]);
        var prompt = new JsonObject();
        foreach (var (sourceId, cached) in cache)
        {
            string capturedId = sourceId;
            var slots = cached!.AsArray();
            producerCalls.Add(sourceId, 0);
            registry.Register(new Node(new(sourceId, sourceId, "reference", [],
                slots.Select(_ => new OutputSchema("*", IsList: true)).ToArray()), (context, _, _) =>
            {
                producerCalls[capturedId]++;
                return ValueTask.FromResult(new NodeExecutionOutput(slots.Select(slot =>
                    context.List(slot!.AsArray().Select(v => Decode(context, v)))).ToArray()));
            }));
            prompt[sourceId] = new JsonObject { ["class_type"] = sourceId, ["inputs"] = new JsonObject() };
        }
        var arguments = new JsonObject();
        foreach (var (name, value) in promptInputs)
            arguments[name] = value is JsonArray && !IsLink(value)
                ? new JsonObject { ["__value__"] = value.DeepClone() } : value?.DeepClone();
        prompt[id] = new JsonObject { ["class_type"] = schema.ClassType, ["inputs"] = arguments };
        string promptBefore = prompt.ToJsonString();
        using var result = await new EngineService(registry).ExecuteValuesAsync(prompt, [id], e =>
        { events.Add(e); return ValueTask.CompletedTask; });
        Assert.Equal("success", result.Status);
        Equal(expected["outputs"], new JsonArray(result.Outputs[id].Select(slot =>
            (JsonNode?)new JsonArray(slot.Select(Encode).ToArray())).ToArray()));
        Equal(expected["observation"]!["createListCalls"], calls);
        Equal(expected["observation"]!["nestedInputReturns"], nested);
        Assert.Empty(expected["ui"]!.AsObject());
        Assert.DoesNotContain(events, e => e.Type == "executed" && e.NodeId == id);
        Assert.False(expected["hasSubgraph"]!.GetValue<bool>());
        CompareReports(expected["reports"]!.AsArray(), events, result.Diagnostics);
        foreach (var (sourceId, count) in producerCalls)
        {
            bool referenced = promptInputs.Any(p => IsLink(p.Value) && p.Value![0]!.GetValue<string>() == sourceId);
            Assert.Equal(referenced ? 1 : 0, count); // Engine dependency cache; no equality claim for source cache-read counts.
        }
        Assert.Equal(promptBefore, prompt.ToJsonString());
        Assert.Equal(specificationBefore, specification.ToJsonString());

        // Exercise the public expansion/binder independently with the protocol's runtime inputs.
        // Source blocked cases intentionally never call the binder; their empty callback capture
        // above verifies that EngineService checks blockers before grouping.
        if (nested.Count != 0)
        {
            using var context = new RuntimeNodeContext();
            var flat = new Dictionary<string, RuntimeValue>(StringComparer.Ordinal);
            foreach (var (name, value) in promptInputs)
            {
                flat.Add(name, IsLink(value)
                    ? context.List(cache[value![0]!.GetValue<string>()]![value[1]!.GetValue<int>()]!.AsArray().Select(v => Decode(context, v)))
                    : context.List([context.Json(value)]));
            }
            Equal(expected["flatInputs"], EncodeInputs(flat));
            Equal(expected["flatOrder"], Names(flat.Keys));
            var bound = expanded.BindArguments(context, flat);
            Equal(Assert.Single(expected["observation"]!["nestedInputReturns"]!.AsArray())!["values"], EncodeInputs(bound));
            Equal(Assert.Single(expected["observation"]!["nestedInputReturns"]!.AsArray())!["groupOrder"], Names(bound["inputs"].Properties.Keys));
        }
    }

    private static NodeSchema Schema(JsonObject specification, NodeSchema create)
    {
        if (specification["kind"]!.GetValue<string>() == "create-list") return create;
        var template = specification["template"]!.AsObject();
        var prototype = create.Inputs[0].Autogrow!.Input with
        { Required = !(template["prototypeOptions"]?["optional"]?.GetValue<bool>() ?? false) };
        var group = new InputSchema("inputs", "COMFY_AUTOGROW_V3", Autogrow: new(prototype,
            template["prefix"]?.GetValue<string>() ?? "input", template["min"]?.GetValue<int>() ?? 1,
            template["max"]?.GetValue<int>() ?? 10));
        return new("FixturePrefix", "Fixture Prefix", "laboratory", [group], create.Outputs,
            InputIsList: true, Description: "", PythonModule: "laboratory.autogrow_prefix", V3ObjectInfo: true);
    }

    private static void CompareFinalized(JsonObject expected, IReadOnlyList<InputSchema> inputs)
    {
        var required = new JsonObject(); var optional = new JsonObject();
        foreach (var input in inputs)
            (input.Required ? required : optional).Add(input.Name,
                new JsonArray(input.Type, input.Options?.DeepClone() ?? new JsonObject()));
        // Source finalization preserves both partitions, including an empty optional map.
        var finalized = new JsonObject { ["required"] = required, ["optional"] = optional };
        var order = new JsonObject { ["required"] = Names(required.Select(p => p.Key)),
            ["optional"] = Names(optional.Select(p => p.Key)) };
        Equal(expected["finalized"], finalized);
        Equal(expected["finalizedOrder"], order);
    }

    private static void CompareReports(JsonArray expected, List<EngineEvent> events, IReadOnlyList<EngineDiagnostic> diagnostics)
    {
        var errors = events.Where(e => e.Type == "execution_error").ToArray();
        Assert.Equal(expected.Count, errors.Length);
        Assert.Equal(expected.Count, diagnostics.Count);
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.Equal(expected[i]!["event"]!.GetValue<string>(), errors[i].Type);
            Assert.Equal(expected[i]!["payload"]!["node_id"]!.GetValue<string>(), errors[i].NodeId);
            Assert.Equal(expected[i]!["payload"]!["exception_message"]!.GetValue<string>(), errors[i].Message);
            Assert.Equal("execution_blocked", diagnostics[i].Code);
            Assert.Equal(errors[i].NodeId, diagnostics[i].NodeId);
            Assert.Equal(errors[i].Message, diagnostics[i].Message);
        }
    }

    private static JsonObject Resource(string name, int bytes, string hash)
    {
        using var stream = typeof(AutogrowSourceReferenceTests).Assembly.GetManifestResourceStream("ComfySharp.Core.Tests.Fixtures." + name);
        Assert.NotNull(stream);
        using var buffer = new MemoryStream(); stream.CopyTo(buffer);
        byte[] raw = buffer.ToArray();
        Assert.Equal(bytes, raw.Length);
        Assert.Equal(hash, Convert.ToHexString(SHA256.HashData(raw)).ToLowerInvariant());
        return JsonNode.Parse(raw)!.AsObject();
    }
    private static bool IsLink(JsonNode? value) => value is JsonArray { Count: 2 } a &&
        a[0] is JsonValue name && name.TryGetValue<string>(out _) && a[1] is JsonValue index && index.TryGetValue<int>(out var i) && i >= 0;
    private static RuntimeValue Decode(RuntimeNodeContext context, JsonNode? value) => value switch
    {
        JsonObject o when o.Count == 1 && o.ContainsKey("$blocker") => context.Blocker(o["$blocker"]?.GetValue<string>()),
        JsonObject o => context.Map(o.ToDictionary(p => p.Key, p => Decode(context, p.Value), StringComparer.Ordinal)),
        JsonArray a => context.List(a.Select(v => Decode(context, v))),
        _ => context.Json(value)
    };
    private static JsonNode? Encode(RuntimeValue value) => value.Kind switch
    {
        RuntimeValueKind.Blocker => new JsonObject { ["$blocker"] = value.Blocker.Message },
        RuntimeValueKind.List => new JsonArray(value.Items.Select(Encode).ToArray()),
        RuntimeValueKind.Map => new JsonObject(value.Properties.Select(p => KeyValuePair.Create(p.Key, Encode(p.Value)))),
        _ => value.ToJson()
    };
    private static JsonObject EncodeInputs(IReadOnlyDictionary<string, RuntimeValue> inputs) =>
        new(inputs.Select(p => KeyValuePair.Create(p.Key, Encode(p.Value))));
    private static JsonArray Names(IEnumerable<string> names) => new(names.Select(s => (JsonNode?)JsonValue.Create(s)).ToArray());
    private static void Equal(JsonNode? expected, JsonNode? actual) =>
        Assert.True(JsonNode.DeepEquals(expected, actual), $"Expected {expected}; actual {actual}");
    private sealed class Node(NodeSchema schema,
        Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, CancellationToken, ValueTask<NodeExecutionOutput>> run) : IRuntimeNode
    {
        public NodeSchema Schema => schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) => run(context, inputs, cancellationToken);
    }
}
