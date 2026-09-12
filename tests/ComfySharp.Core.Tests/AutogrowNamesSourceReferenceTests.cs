using System.Security.Cryptography;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>
/// Exact comparison of the 25 prospectively comparable Names cases. The other 11 remain
/// in the immutable source resource. The source fixture and this test-only node carry
/// arguments transparently; neither is a production node or a StringFormat port.
/// Source acquisition bypasses whole-prompt validation. Here real typed producers and
/// EngineService exercise the public validated execution path, without a private memo API.
/// The sole source mapper error is compared as an explicit failure boundary, not as an
/// identical Python/.NET exception. All recorded argument orders remain exact.
/// </summary>
public sealed class AutogrowNamesSourceReferenceTests
{
    private const string ReferenceSha = "f3f6772bdf05aa464cc831edb8e929ff179f73df3d01ea2f4546616eafcbe4e3";
    private const string ProtocolSha = "08b18c44494d41fef84d4055f0ef7baab1d092db0f6706af7a5043a14f8f9a8c";
    private static readonly string[] ComparableIds =
    [
        "explicit-order-repeat-last", "input-is-list-order", "optional-hole", "all-required", "min-above-count",
        "empty-names-min-zero", "empty-names-min-one", "single-name", "hundred-names", "truncate-hundred-one",
        "optional-prototype", "match-wildcard", "ordinary-inputs-around-group", "two-groups", "omitted-optional",
        "explicit-null", "empty-list-before-blocker", "empty-input-is-list", "scalar-blocker-prompt-order",
        "scalar-blocker-silent-first", "scalar-blocker-empty-message", "list-blocker-before-grouping",
        "nested-marker-not-recursive", "caller-names-copy", "truncate-ambiguous-tail"
    ];

    [Theory]
    [InlineData("explicit-order-repeat-last")]
    [InlineData("input-is-list-order")]
    [InlineData("optional-hole")]
    [InlineData("all-required")]
    [InlineData("min-above-count")]
    [InlineData("empty-names-min-zero")]
    [InlineData("empty-names-min-one")]
    [InlineData("single-name")]
    [InlineData("hundred-names")]
    [InlineData("truncate-hundred-one")]
    [InlineData("optional-prototype")]
    [InlineData("match-wildcard")]
    [InlineData("ordinary-inputs-around-group")]
    [InlineData("two-groups")]
    [InlineData("omitted-optional")]
    [InlineData("explicit-null")]
    [InlineData("empty-list-before-blocker")]
    [InlineData("empty-input-is-list")]
    [InlineData("scalar-blocker-prompt-order")]
    [InlineData("scalar-blocker-silent-first")]
    [InlineData("scalar-blocker-empty-message")]
    [InlineData("list-blocker-before-grouping")]
    [InlineData("nested-marker-not-recursive")]
    [InlineData("caller-names-copy")]
    [InlineData("truncate-ambiguous-tail")]
    public async Task PublishedNamesMetadataAndArgumentTransportMatchTheExactSourceBoundary(string id)
    {
        var reference = Resource("autogrow-names.reference.json", 296326, ReferenceSha);
        var protocol = Resource("autogrow-names.protocol.json", 47770, ProtocolSha);
        Assert.True(reference["executed"]!.GetValue<bool>());
        Assert.Equal("4312bdafabe2906b74bc51cdca678aa26aeec5ee", reference["collectorCommit"]!.GetValue<string>());
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", reference["backendCommit"]!.GetValue<string>());
        Assert.Equal("autogrow-template-names-v1", reference["protocolId"]!.GetValue<string>());
        Assert.Equal(ProtocolSha, reference["protocolSha256"]!.GetValue<string>());
        Assert.Equal("3.12.10", reference["python"]!.GetValue<string>());
        Equal(protocol["sources"], reference["sourceFiles"]);
        Equal(protocol["helperFiles"], reference["helperFiles"]);
        Assert.Equal(10, reference["sourceFiles"]!.AsArray().Count);
        Assert.Equal(73, reference["sourceFiles"]!.AsArray().Sum(s => s!["declarations"]!.AsArray().Count));
        var records = reference["referenceOutputs"]!.AsArray();
        Assert.Equal(36, records.Count);
        Assert.Equal(protocol["cases"]!.AsArray().Select(c => c!["id"]!.GetValue<string>()), records.Select(c => c!["id"]!.GetValue<string>()));
        Assert.Equal(36, records.Select(c => c!["id"]!.GetValue<string>()).Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(ComparableIds, records.Where(c => c!["portClassification"]!.GetValue<string>() == "comparable")
            .Select(c => c!["id"]!.GetValue<string>()));
        Assert.Equal(6, records.Count(c => c!["portClassification"]!.GetValue<string>() == "unsupported-template"));
        Assert.Equal(1, records.Count(c => c!["portClassification"]!.GetValue<string>() == "constructor-error"));
        Assert.Equal(2, records.Count(c => c!["portClassification"]!.GetValue<string>() == "source-only"));
        Assert.Equal(2, records.Count(c => c!["portClassification"]!.GetValue<string>() == "unsupported-input"));
        var specification = Assert.Single(protocol["cases"]!.AsArray(), c => c!["id"]!.GetValue<string>() == id)!.AsObject();
        var expected = Assert.Single(records, c => c!["id"]!.GetValue<string>() == id)!.AsObject();
        Assert.Equal("comparable", specification["portClassification"]!.GetValue<string>());
        Assert.Equal("fixture-names", specification["kind"]!.GetValue<string>());
        Assert.Equal("fixture.argument-carrier", expected["bodyOrigin"]!.GetValue<string>());
        Equal(specification["promptInputs"], expected["inputsAfter"]);
        Equal(specification["cache"], expected["cacheAfter"]);
        Equal(new JsonArray("off", "on", "off"), expected["repeats"]!["sequence"]);
        Assert.True(expected["repeats"]!["bitIdentical"]!.GetValue<bool>());
        var hashes = expected["repeats"]!["sha256"]!.AsArray();
        Assert.Equal(3, hashes.Count);
        Assert.All(hashes, h => Assert.Matches("^[0-9a-f]{64}$", h!.GetValue<string>()));
        Assert.Single(hashes.Select(h => h!.GetValue<string>()).Distinct(StringComparer.Ordinal));

        string specificationBefore = specification.ToJsonString();
        string expectedBefore = expected.ToJsonString();
        var promptInputs = specification["promptInputs"]!.AsObject();
        var cache = specification["cache"]!.AsObject();
        var (schema, template, callerNames) = Schema(specification);
        Equal(specification["template"]!["names"], expected["callerNamesBefore"]);
        Equal(expected["callerNamesAfter"], Names(callerNames));
        Equal(expected["cachedInputOrder"], Names(template.MemberNames));
        var group = schema.Inputs.Single(i => i.Name == "values");
        var templateInfo = NodeInputExpansion.TemplateOptions(group)["template"]!;
        Equal(expected["templateAtConstruction"], templateInfo);
        Equal(expected["templateAfterCallerMutation"], templateInfo);
        Assert.Equal(new[] { "input", "names", "min" }, templateInfo.AsObject().Select(p => p.Key));
        var expanded = NodeInputExpansion.Expand(schema, promptInputs.Select(p => p.Key));
        CompareFinalized(expected, expanded.Inputs);

        var calls = new JsonArray();
        var events = new List<EngineEvent>();
        var producerCalls = new Dictionary<string, int>(StringComparer.Ordinal);
        var registry = new NodeRegistry();
        registry.Register(new Node(schema, (context, inputs, _) =>
        {
            // One actual C# entry capture. The source separately observed its binder return and body entry.
            calls.Add(Observe(inputs));
            return ValueTask.FromResult(new NodeExecutionOutput([context.Map(inputs)]));
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
                    context.List(slot!.AsArray().Select(value => DecodeCache(context, value)))).ToArray()));
            }));
            prompt[sourceId] = new JsonObject { ["class_type"] = sourceId, ["inputs"] = new JsonObject() };
        }
        var arguments = new JsonObject();
        foreach (var (name, value) in promptInputs)
            // This adaptation is a literal-value boundary, not an assertion of raw HTTP parity.
            arguments.Add(name, value is JsonArray && !IsLink(value)
                ? new JsonObject { ["__value__"] = value.DeepClone() } : value?.DeepClone());
        prompt[id] = new JsonObject { ["class_type"] = schema.ClassType, ["inputs"] = arguments };
        string promptBefore = prompt.ToJsonString();
        var engine = new EngineService(registry);
        var validation = engine.Validate(prompt, [id]);
        Assert.True(validation.IsValid); Assert.Empty(validation.Diagnostics);
        Assert.Equal(new[] { id }, validation.ValidTargets);
        Assert.Empty(calls); Assert.All(producerCalls.Values, count => Assert.Equal(0, count));
        using var result = await engine.ExecuteValuesAsync(prompt, [id], e => { events.Add(e); return ValueTask.CompletedTask; });

        if (id == "empty-list-before-blocker")
        {
            Equal(new JsonArray(new JsonObject { ["stage"] = "mapper", ["type"] = "IndexError" }), specification["permittedSourceErrors"]);
            Assert.Equal("raised", expected["status"]!.GetValue<string>());
            Assert.Equal("mapper", expected["stage"]!.GetValue<string>());
            Assert.Equal("IndexError", expected["errorType"]!.GetValue<string>());
            Assert.Equal("list index out of range", expected["errorMessage"]!.GetValue<string>());
            Assert.Empty(expected["reports"]!.AsArray());
            Assert.Empty(expected["observation"]!["bodyCalls"]!.AsArray());
            Assert.Empty(expected["observation"]!["nestedInputReturns"]!.AsArray());
            Assert.False(expected.ContainsKey("outputs"));
            Assert.Equal("error", result.Status); Assert.Empty(result.Outputs); Assert.Empty(calls);
            var diagnostic = Assert.Single(result.Diagnostics);
            Assert.Equal("execution_error", diagnostic.Code); Assert.Equal(id, diagnostic.NodeId);
            // Explicit local diagnostic: no Python exception/message equivalence is asserted.
            Assert.Equal("Cannot repeat the last item of an empty execution list.", diagnostic.Message);
            var error = Assert.Single(events, e => e.Type == "execution_error");
            Assert.Equal(id, error.NodeId); Assert.Equal(diagnostic.Message, error.Message);
            Assert.DoesNotContain(result.Diagnostics, d => d.Code == "execution_blocked");
        }
        else
        {
            Assert.Empty(specification["permittedSourceErrors"]!.AsArray());
            Assert.Equal("returned", expected["status"]!.GetValue<string>());
            Assert.Equal("success", result.Status);
            // Keep this order assertion exact, including tail/head/values in ordinary-inputs-around-group.
            Equal(expected["observation"]!["bodyCalls"], calls);
            // The two source observations agree; there is only one instrumented C# boundary above.
            Equal(expected["observation"]!["nestedInputReturns"], calls);
            Equal(expected["outputs"], new JsonArray(result.Outputs[id].Select(slot =>
                (JsonNode?)new JsonArray(slot.Select(Encode).ToArray())).ToArray()));
            Assert.Equal(id, Assert.Single(result.Outputs).Key);
            Assert.Empty(expected["ui"]!.AsObject()); Assert.False(expected["hasSubgraph"]!.GetValue<bool>());
            Assert.DoesNotContain(events, e => e.Type == "executed" && e.NodeId == id);
            CompareReports(expected["reports"]!.AsArray(), events, result.Diagnostics);
        }
        foreach (var (sourceId, count) in producerCalls)
        {
            bool referenced = promptInputs.Any(p => IsLink(p.Value) && p.Value![0]!.GetValue<string>() == sourceId);
            Assert.Equal(referenced ? 1 : 0, count); // Per-job producer memo; not source cache-read count parity.
        }
        Assert.Equal(promptBefore, prompt.ToJsonString());

        // Public binder comparison for whole-list calls needs no duplicate scalar-mapper implementation.
        // Blocked cases intentionally never reach binding in either source or real EngineService.
        if (schema.InputIsList && calls.Count != 0)
        {
            using var context = new RuntimeNodeContext();
            var flat = new Dictionary<string, RuntimeValue>(StringComparer.Ordinal);
            foreach (var (name, value) in promptInputs)
                flat.Add(name, IsLink(value)
                    ? context.List(cache[value![0]!.GetValue<string>()]![value[1]!.GetValue<int>()]!.AsArray().Select(v => DecodeCache(context, v)))
                    : context.List([context.Json(value)]));
            Equal(expected["flatInputs"], EncodeInputs(flat));
            Equal(expected["flatOrder"], Names(flat.Keys));
            var bound = expanded.BindArguments(context, flat);
            Equal(Assert.Single(expected["observation"]!["nestedInputReturns"]!.AsArray()), Observe(bound));
        }
        Assert.Equal(specificationBefore, specification.ToJsonString());
        Assert.Equal(expectedBefore, expected.ToJsonString());
    }

    private static (NodeSchema Schema, AutogrowNamesTemplate Template, List<string> CallerNames) Schema(JsonObject specification)
    {
        var config = specification["template"]!.AsObject();
        var caller = config["names"]!.AsArray().Select(n => n!.GetValue<string>()).ToList();
        var options = config["prototypeOptions"]?.DeepClone().AsObject() ?? new JsonObject();
        bool optional = options["optional"]?.GetValue<bool>() ?? false;
        options.Remove("optional");
        string type = config["prototype"]?.GetValue<string>() ?? "AnyType";
        Assert.Contains(type, new[] { "AnyType", "MatchType" });
        if (type == "MatchType") options["template"] = new JsonObject { ["template_id"] = "type", ["allowed_types"] = "*" };
        var template = new AutogrowNamesTemplate(new("value", type == "MatchType" ? "COMFY_MATCHTYPE_V3" : "*",
            Required: !optional, Options: options), caller, config["min"]?.GetValue<int>() ?? 1);
        if (config["mutateCallerNames"] is JsonArray mutation)
        {
            caller.Clear(); caller.AddRange(mutation.Select(n => n!.GetValue<string>()));
        }
        // All comparable groups are required and non-lazy; unsupported records never enter this builder.
        Assert.True(config["groupOptions"] is null || config["groupOptions"]!.AsObject().Count == 0);
        var inputs = new List<InputSchema>();
        var order = config["ordinaryInputs"]?.AsArray().Select(n => n!.GetValue<string>()).ToArray() ?? ["@group"];
        foreach (var name in order)
        {
            if (name == "@group") inputs.Add(new("values", "COMFY_AUTOGROW_V3", Autogrow: template));
            else if (name == "@other")
            {
                var other = config["otherGroup"]!;
                inputs.Add(new("other", "COMFY_AUTOGROW_V3", Autogrow: new AutogrowNamesTemplate(new("value", "*"),
                    other["names"]!.AsArray().Select(n => n!.GetValue<string>()).ToArray(), other["min"]?.GetValue<int>() ?? 1)));
            }
            else inputs.Add(new(name, "*"));
        }
        return (new("FixtureNames", "Fixture Names", "laboratory", inputs, [new("*", "arguments")],
            InputIsList: config["isInputList"]?.GetValue<bool>() ?? false,
            Description: "", PythonModule: "laboratory.autogrow_names", V3ObjectInfo: true), template, caller);
    }

    private static void CompareFinalized(JsonObject expected, IReadOnlyList<InputSchema> inputs)
    {
        var required = new JsonObject(); var optional = new JsonObject();
        foreach (var input in inputs)
            (input.Required ? required : optional).Add(input.Name, new JsonArray(input.Type, input.Options?.DeepClone() ?? new JsonObject()));
        Equal(expected["finalized"], new JsonObject { ["required"] = required, ["optional"] = optional });
        Equal(expected["finalizedOrder"], new JsonObject { ["required"] = Names(required.Select(p => p.Key)), ["optional"] = Names(optional.Select(p => p.Key)) });
    }

    private static JsonObject Observe(IReadOnlyDictionary<string, RuntimeValue> values)
    {
        var encoded = EncodeInputs(values);
        var groupOrders = new JsonObject();
        foreach (var (key, value) in values)
            if (value.Kind != RuntimeValueKind.Blocker && encoded[key] is JsonObject group)
                groupOrders.Add(key, Names(group.Select(p => p.Key)));
        return new() { ["values"] = encoded, ["rootOrder"] = Names(values.Keys), ["groupOrders"] = groupOrders };
    }

    private static void CompareReports(JsonArray expected, List<EngineEvent> events, IReadOnlyList<EngineDiagnostic> diagnostics)
    {
        var errors = events.Where(e => e.Type == "execution_error").ToArray();
        Assert.Equal(expected.Count, errors.Length); Assert.Equal(expected.Count, diagnostics.Count);
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.Equal(expected[i]!["event"]!.GetValue<string>(), errors[i].Type);
            Assert.Equal(expected[i]!["payload"]!["node_id"]!.GetValue<string>(), errors[i].NodeId);
            Assert.Equal(expected[i]!["payload"]!["exception_message"]!.GetValue<string>(), errors[i].Message);
            Assert.Equal("execution_blocked", diagnostics[i].Code);
            Assert.Equal(errors[i].NodeId, diagnostics[i].NodeId); Assert.Equal(errors[i].Message, diagnostics[i].Message);
        }
    }

    private static JsonObject Resource(string name, int size, string sha)
    {
        using var stream = typeof(AutogrowNamesSourceReferenceTests).Assembly.GetManifestResourceStream("ComfySharp.Core.Tests.Fixtures." + name);
        Assert.NotNull(stream); using var buffer = new MemoryStream(); stream.CopyTo(buffer);
        byte[] raw = buffer.ToArray();
        Assert.Equal(size, raw.Length); Assert.Equal(sha, Convert.ToHexString(SHA256.HashData(raw)).ToLowerInvariant());
        return JsonNode.Parse(raw)!.AsObject();
    }
    private static bool IsLink(JsonNode? value) => value is JsonArray { Count: 2 } a &&
        a[0] is JsonValue name && name.TryGetValue<string>(out _) &&
        a[1] is JsonValue index && index.TryGetValue<int>(out var i) && i >= 0;
    // Only cache fixtures decode markers. Prompt dictionaries remain ordinary input data.
    private static RuntimeValue DecodeCache(RuntimeNodeContext context, JsonNode? value) => value switch
    {
        JsonObject o when o.Count == 1 && o.ContainsKey("$blocker") => context.Blocker(o["$blocker"]?.GetValue<string>()),
        JsonObject o => context.Map(o.ToDictionary(p => p.Key, p => DecodeCache(context, p.Value), StringComparer.Ordinal)),
        JsonArray a => context.List(a.Select(v => DecodeCache(context, v))),
        _ => context.Json(value)
    };
    private static JsonNode? Encode(RuntimeValue value) => value.Kind switch
    {
        RuntimeValueKind.Blocker => new JsonObject { ["$blocker"] = value.Blocker.Message },
        RuntimeValueKind.List => new JsonArray(value.Items.Select(Encode).ToArray()),
        RuntimeValueKind.Map => new JsonObject(value.Properties.Select(p => KeyValuePair.Create(p.Key, Encode(p.Value)))),
        _ => value.ToJson()
    };
    private static JsonObject EncodeInputs(IReadOnlyDictionary<string, RuntimeValue> values) =>
        new(values.Select(p => KeyValuePair.Create(p.Key, Encode(p.Value))));
    private static JsonArray Names(IEnumerable<string> names) => new(names.Select(n => (JsonNode?)JsonValue.Create(n)).ToArray());
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
