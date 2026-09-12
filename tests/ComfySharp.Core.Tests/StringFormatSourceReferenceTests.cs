using System.Security.Cryptography;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>
/// Published source observations, partitioned before collection: 24 exact output cases,
/// 12 source failures compared to explicit local failure boundaries, and 8 local profile refusals.
/// The observer delegates to the actual StringFormatNode; it never formats or synthesizes outputs.
/// Source acquisition is not whole-prompt HTTP validation, and Python diagnostic text is not a local oracle.
/// </summary>
public sealed class StringFormatSourceReferenceTests
{
    private const string ReferenceSha = "a6c1aa98c333851195d0bff97f3e1305835634a1cb922dcc102dabe5e07871e0";
    private const string ProtocolSha = "2eae9a7d3719a225110e032e4b12233300c7ae168a6659d140d87778a34aaefd";

    [Theory]
    [InlineData("constant-empty-group", "comparable", null)]
    [InlineData("escaped-braces", "comparable", null)]
    [InlineData("repeat-ordinal-fields", "comparable", null)]
    [InlineData("mapped-format-and-values", "comparable", null)]
    [InlineData("unused-unsupported-value", "comparable", null)]
    [InlineData("unicode-string-conversion", "comparable", null)]
    [InlineData("default-text-width", "comparable", null)]
    [InlineData("explicit-left-fill", "comparable", null)]
    [InlineData("explicit-right-fill", "comparable", null)]
    [InlineData("center-odd-padding", "comparable", null)]
    [InlineData("astral-fill-codepoint", "comparable", null)]
    [InlineData("zero-precision", "comparable", null)]
    [InlineData("astral-text-precision", "comparable", null)]
    [InlineData("combining-width-precision", "comparable", null)]
    [InlineData("explicit-zero-fill", "comparable", null)]
    [InlineData("zero-width", "comparable", null)]
    [InlineData("none-empty-spec", "comparable", null)]
    [InlineData("true-empty-spec", "comparable", null)]
    [InlineData("false-empty-spec", "comparable", null)]
    [InlineData("int64-min-empty-spec", "comparable", null)]
    [InlineData("int64-max-empty-spec", "comparable", null)]
    [InlineData("none-convert-before-spec", "comparable", null)]
    [InlineData("bool-convert-before-precision", "comparable", null)]
    [InlineData("integer-convert-before-zero-fill", "comparable", null)]
    [InlineData("unmatched-opening", "source-error", "format_syntax")]
    [InlineData("unmatched-closing", "source-error", "format_syntax")]
    [InlineData("missing-named-field", "source-error", "format_missing_field")]
    [InlineData("missing-automatic-position", "source-error", "format_missing_field")]
    [InlineData("missing-explicit-position", "source-error", "format_missing_field")]
    [InlineData("malformed-conversion", "source-error", "format_syntax")]
    [InlineData("missing-before-late-syntax", "source-error", "format_missing_field")]
    [InlineData("conversion-before-nested-spec", "source-error", "format_syntax")]
    [InlineData("repr-and-ascii", "unsupported-profile", "unsupported_format_feature")]
    [InlineData("dictionary-item", "unsupported-profile", "unsupported_format_feature")]
    [InlineData("integer-attribute", "unsupported-profile", "unsupported_format_feature")]
    [InlineData("nested-width", "unsupported-profile", "unsupported_format_feature")]
    [InlineData("float-numeric-spec", "unsupported-profile", "unsupported_format_value")]
    [InlineData("container-empty-spec", "unsupported-profile", "unsupported_format_value")]
    [InlineData("integer-numeric-spec", "unsupported-profile", "unsupported_format_feature")]
    [InlineData("leading-zero-width", "unsupported-profile", "unsupported_format_feature")]
    [InlineData("missing-before-unknown-conversion", "source-error", "format_missing_field")]
    [InlineData("missing-before-unsupported-spec", "source-error", "format_missing_field")]
    [InlineData("malformed-conversion-before-missing", "source-error", "format_syntax")]
    [InlineData("unclosed-field-before-missing", "source-error", "format_syntax")]
    public async Task ActualNodeMatchesThePublishedBoundaryOrItsDeclaredProfileRefusal(string id, string classification, string? localCategory)
    {
        var reference = Resource("string-format.reference.json", 408527, ReferenceSha);
        var protocol = Resource("string-format.protocol.json", 47482, ProtocolSha);
        Assert.True(reference["executed"]!.GetValue<bool>());
        Assert.True(reference["sourceNodeExecution"]!.GetValue<bool>());
        Assert.Equal("52fe346c74c2db14b6c73aec804c215674550f4d", reference["collectorCommit"]!.GetValue<string>());
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", reference["backendCommit"]!.GetValue<string>());
        Assert.Equal("string-format-text-v1", reference["protocolId"]!.GetValue<string>());
        Assert.Equal(ProtocolSha, reference["protocolSha256"]!.GetValue<string>());
        Assert.Equal("3.12.10", reference["python"]!.GetValue<string>());
        Equal(protocol["sources"], reference["sourceFiles"]); Equal(protocol["helperFiles"], reference["helperFiles"]);
        Equal(protocol["profile"], reference["profile"]);
        Assert.Equal(10, reference["sourceFiles"]!.AsArray().Count);
        Assert.Equal(73, reference["sourceFiles"]!.AsArray().Sum(s => s!["declarations"]!.AsArray().Count));
        var cases = protocol["cases"]!.AsArray(); var records = reference["referenceOutputs"]!.AsArray();
        Assert.Equal(44, reference["cases"]!.GetValue<int>()); Assert.Equal(44, records.Count); Assert.Equal(44, cases.Count);
        Assert.Equal(44, cases.Select(c => c!["id"]!.GetValue<string>()).Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(cases.Select(c => c!["id"]!.GetValue<string>()), records.Select(c => c!["id"]!.GetValue<string>()));
        foreach (var (kind, count) in new[] { ("comparable", 24), ("source-error", 12), ("unsupported-profile", 8) })
        {
            Assert.Equal(count, cases.Count(c => c!["portClassification"]!.GetValue<string>() == kind));
            Assert.Equal(count, records.Count(c => c!["portClassification"]!.GetValue<string>() == kind));
        }
        var specification = Assert.Single(cases, c => c!["id"]!.GetValue<string>() == id)!.AsObject();
        var expected = Assert.Single(records, c => c!["id"]!.GetValue<string>() == id)!.AsObject();
        Assert.Equal(classification, specification["portClassification"]!.GetValue<string>());
        Assert.Equal(classification, expected["portClassification"]!.GetValue<string>());
        Assert.Equal("source.StringFormat.execute -> builtin str.format", expected["bodyOrigin"]!.GetValue<string>());
        Equal(specification["promptInputs"], expected["inputsAfter"]); Equal(specification["cache"], expected["cacheAfter"]);
        Equal(new JsonArray("off", "on", "off"), expected["repeats"]!["sequence"]);
        Assert.True(expected["repeats"]!["bitIdentical"]!.GetValue<bool>());
        var hashes = expected["repeats"]!["sha256"]!.AsArray(); Assert.Equal(3, hashes.Count);
        Assert.All(hashes, h => Assert.Matches("^[0-9a-f]{64}$", h!.GetValue<string>()));
        Assert.Single(hashes.Select(h => h!.GetValue<string>()).Distinct(StringComparer.Ordinal));

        string specificationBefore = specification.ToJsonString(), expectedBefore = expected.ToJsonString();
        var registered = BuiltInNodes.CreateRegistry();
        Assert.True(registered.TryGet("StringFormat", out var registeredNode));
        var node = Assert.IsType<StringFormatNode>(registeredNode);
        CompareMetadata(expected, registered.ToObjectInfo()["StringFormat"]!.AsObject(), node.Schema, specification);
        var captures = new List<Capture>(); var registry = new NodeRegistry();
        registry.Register(new ObservedFormat(node, captures));
        var cache = specification["cache"]!.AsObject(); var promptInputs = specification["promptInputs"]!.AsObject();
        var producerCalls = new Dictionary<string, int>(StringComparer.Ordinal); var prompt = new JsonObject();
        foreach (var (sourceId, cached) in cache)
        {
            string capturedId = sourceId; var slots = cached!.AsArray(); producerCalls.Add(sourceId, 0);
            registry.Register(new Producer(new(sourceId, sourceId, "reference", [],
                slots.Select(_ => new OutputSchema("*", IsList: true)).ToArray()), context =>
            {
                producerCalls[capturedId]++;
                return new(slots.Select(slot => context.List(slot!.AsArray().Select(v => CacheValue(context, v)))).ToArray());
            }));
            prompt[sourceId] = new JsonObject { ["class_type"] = sourceId, ["inputs"] = new JsonObject() };
        }
        var arguments = new JsonObject();
        foreach (var (name, value) in promptInputs)
            arguments.Add(name, value is JsonArray && !IsLink(value)
                ? new JsonObject { ["__value__"] = value.DeepClone() } : value?.DeepClone());
        prompt[id] = new JsonObject { ["class_type"] = "StringFormat", ["inputs"] = arguments };
        string promptBefore = prompt.ToJsonString(); var engine = new EngineService(registry);
        var validation = engine.Validate(prompt, [id]); Assert.True(validation.IsValid); Assert.Empty(validation.Diagnostics);
        Assert.Empty(captures); Assert.All(producerCalls.Values, count => Assert.Equal(0, count));
        var events = new List<EngineEvent>();
        using var result = await engine.ExecuteValuesAsync(prompt, [id], e => { events.Add(e); return ValueTask.CompletedTask; });

        // These managed snapshots observe the real wrapper entry once. The source has two hooks.
        // No native/context-owned RuntimeValue is held after an invocation completes.
        var sourceBodies = expected["observation"]!["bodyCalls"]!.AsArray();
        var sourceBound = expected["observation"]!["nestedInputReturns"]!.AsArray();
        Assert.Equal(sourceBodies.Count, captures.Count); Assert.Equal(sourceBound.Count, captures.Count);
        for (int i = 0; i < captures.Count; i++)
        {
            Equal(sourceBodies[i]!["values"], captures[i].Arguments["values"]);
            Equal(sourceBodies[i]!["f_string"], captures[i].Arguments["f_string"]);
            Equal(sourceBodies[i]!["valueOrder"], Names(captures[i].ValueOrder));
            Equal(sourceBound[i]!["arguments"], captures[i].Arguments);
            Equal(sourceBound[i]!["rootOrder"], Names(captures[i].RootOrder));
            Equal(sourceBound[i]!["valueOrder"], Names(captures[i].ValueOrder));
        }
        Assert.Empty(expected["reports"]!.AsArray());
        if (classification == "comparable")
        {
            Assert.Null(localCategory); Assert.Empty(specification["permittedSourceErrors"]!.AsArray());
            Assert.Equal("returned", expected["status"]!.GetValue<string>()); Assert.Equal("success", result.Status);
            Assert.Empty(result.Diagnostics); Assert.Equal(id, Assert.Single(result.Outputs).Key);
            var actual = new JsonArray(result.Outputs[id].Select(slot => (JsonNode?)new JsonArray(slot.Select(value =>
            {
                Assert.Equal(RuntimeValueKind.Json, value.Kind); var json = value.ToJson();
                Assert.IsAssignableFrom<JsonValue>(json); _ = json!.GetValue<string>(); return json;
            }).ToArray())).ToArray());
            Equal(expected["outputs"], actual);
            Assert.Empty(expected["ui"]!.AsObject()); Assert.False(expected["hasSubgraph"]!.GetValue<bool>());
            Assert.DoesNotContain(events, e => e.Type == "execution_error");
        }
        else
        {
            Assert.NotNull(localCategory); Assert.Equal("error", result.Status); Assert.Empty(result.Outputs);
            var diagnostic = Assert.Single(result.Diagnostics); Assert.Equal("execution_error", diagnostic.Code);
            Assert.Equal(id, diagnostic.NodeId); Assert.StartsWith(localCategory + " at UTF-16 position 0:", diagnostic.Message);
            var error = Assert.Single(events, e => e.Type == "execution_error");
            Assert.Equal(id, error.NodeId); Assert.Equal(diagnostic.Message, error.Message);
            if (classification == "source-error")
            {
                Assert.Equal("raised", expected["status"]!.GetValue<string>()); Assert.False(expected.ContainsKey("outputs"));
                var permitted = Assert.Single(specification["permittedSourceErrors"]!.AsArray())!;
                Assert.Equal("mapper", expected["stage"]!.GetValue<string>()); Equal(permitted["stage"], expected["stage"]);
                Equal(permitted["type"], expected["errorType"]);
                // Enforce the semantic boundary explicitly; do not equate Python exception messages.
                Assert.Equal(localCategory == "format_missing_field"
                    ? (id.Contains("position", StringComparison.Ordinal) ? "IndexError" : "KeyError") : "ValueError",
                    expected["errorType"]!.GetValue<string>());
                Assert.NotEmpty(expected["errorMessage"]!.GetValue<string>());
            }
            else
            {
                Assert.Equal("unsupported-profile", classification);
                Assert.Equal("returned", expected["status"]!.GetValue<string>());
                Assert.Empty(specification["permittedSourceErrors"]!.AsArray());
                Assert.NotEmpty(specification["unsupportedReasons"]!.AsArray());
                // Python succeeded; the local refusal is required by the prospective profile.
                // Its observed Python string is deliberately not asserted as a local output.
                Assert.NotEmpty(expected["outputs"]!.AsArray());
                Assert.StartsWith("unsupported_format_", localCategory);
            }
        }
        foreach (var (sourceId, count) in producerCalls)
            Assert.Equal(promptInputs.Any(p => IsLink(p.Value) && p.Value![0]!.GetValue<string>() == sourceId) ? 1 : 0, count);
        // Producer memo counts are local observables, not an invented source cache-read API parity.
        Assert.Equal(promptBefore, prompt.ToJsonString());
        Assert.Equal(specificationBefore, specification.ToJsonString()); Assert.Equal(expectedBefore, expected.ToJsonString());
    }

    private static void CompareMetadata(JsonObject expected, JsonObject actual, NodeSchema schema, JsonObject specification)
    {
        var source = expected["objectInfo"]!.DeepClone().AsObject(); actual = actual.DeepClone().AsObject();
        string sourceDescription = source["description"]!.GetValue<string>(), localDescription = actual["description"]!.GetValue<string>();
        Assert.NotEqual(sourceDescription, localDescription);
        Assert.Contains("Supports all of Python's format options", sourceDescription);
        Assert.Contains("partial", localDescription); Assert.Contains("python-format-text-v1", localDescription);
        source.Remove("description"); actual.Remove("description"); Equal(source, actual);
        Equal(expected["inputTypes"], actual["input"]);
        var expanded = NodeInputExpansion.Expand(schema, specification["promptInputs"]!.AsObject().Select(p => p.Key));
        var finalized = new JsonObject();
        foreach (bool required in new[] { true, false })
        {
            string partition = required ? "required" : "optional"; var inputs = new JsonObject();
            foreach (var input in expanded.Inputs.Where(i => i.Required == required))
                inputs.Add(input.Name, new JsonArray(input.Type, input.Options?.DeepClone() ?? new JsonObject()));
            finalized.Add(partition, inputs);
            Equal(expected["finalizedOrder"]![partition], Names(inputs.Select(p => p.Key)));
        }
        Equal(expected["finalized"], finalized);
    }

    private sealed record Capture(JsonObject Arguments, string[] RootOrder, string[] ValueOrder);
    private sealed class ObservedFormat(StringFormatNode inner, List<Capture> captures) : IRuntimeNode
    {
        public NodeSchema Schema => inner.Schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            var snapshot = new JsonObject(); foreach (var (name, value) in inputs) snapshot.Add(name, value.ToJson());
            captures.Add(new(snapshot, inputs.Keys.ToArray(), inputs["values"].Properties.Keys.ToArray()));
            return inner.ExecuteAsync(context, inputs, cancellationToken);
        }
    }
    private sealed class Producer(NodeSchema schema, Func<RuntimeNodeContext, NodeExecutionOutput> execute) : IRuntimeNode
    {
        public NodeSchema Schema => schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        { cancellationToken.ThrowIfCancellationRequested(); return ValueTask.FromResult(execute(context)); }
    }
    private static RuntimeValue CacheValue(RuntimeNodeContext context, JsonNode? source)
    {
        // Clone parsed JSON directly: null/bool/Int64 limits/float lexemes/containers keep their types.
        // No ToString conversion, precision-changing numeric cast or private format-marker syntax.
        var value = context.Json(source?.DeepClone());
        Assert.Equal(source?.ToJsonString(), value.ToJson()?.ToJsonString()); return value;
    }
    private static bool IsLink(JsonNode? value) => value is JsonArray { Count: 2 } a && a[0] is JsonValue id &&
        id.TryGetValue<string>(out _) && a[1] is JsonValue slot && slot.TryGetValue<int>(out _);
    private static JsonArray Names(IEnumerable<string> values) => new(values.Select(v => (JsonNode?)JsonValue.Create(v)).ToArray());
    private static void Equal(JsonNode? expected, JsonNode? actual) => Assert.True(JsonNode.DeepEquals(expected, actual),
        $"Expected {expected?.ToJsonString() ?? "null"}; actual {actual?.ToJsonString() ?? "null"}.");
    private static JsonObject Resource(string name, int bytes, string sha)
    {
        using var stream = typeof(StringFormatSourceReferenceTests).Assembly.GetManifestResourceStream("ComfySharp.Core.Tests.Fixtures." + name);
        Assert.NotNull(stream); using var copy = new MemoryStream(); stream.CopyTo(copy); var raw = copy.ToArray();
        Assert.Equal(bytes, raw.Length); Assert.Equal(sha, Convert.ToHexStringLower(SHA256.HashData(raw)));
        return JsonNode.Parse(raw)!.AsObject();
    }
}
