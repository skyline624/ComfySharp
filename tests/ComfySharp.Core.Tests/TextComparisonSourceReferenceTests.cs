using System.Security.Cryptography;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>
/// Forty published cases: 36 comparable, two typed body errors with a distinct coercing
/// engine path, and two source-only modes refused at local COMBO admission.
/// The observer delegates to actual registered nodes and snapshots only managed values.
/// </summary>
public sealed class TextComparisonSourceReferenceTests
{
    private const string ProtocolSha = "0d00fa0c7721b07b2fc12381f60ed364b286963553577fcdb6672c028e18aa92";
    private const string ReferenceSha = "12935c33d4632a79e499a7c24b052d88036aee8323e45ea92d923443b2b9e04f";

    [Theory]
    [InlineData("contains-start", "comparable")]
    [InlineData("contains-middle", "comparable")]
    [InlineData("contains-end", "comparable")]
    [InlineData("contains-absent", "comparable")]
    [InlineData("contains-case-distinct", "comparable")]
    [InlineData("contains-both-empty", "comparable")]
    [InlineData("contains-needle-empty", "comparable")]
    [InlineData("contains-text-empty", "comparable")]
    [InlineData("contains-astral-combining", "comparable")]
    [InlineData("contains-nul-newline", "comparable")]
    [InlineData("contains-ascii", "comparable")]
    [InlineData("contains-dotted-i", "comparable")]
    [InlineData("contains-sigma-isolated", "comparable")]
    [InlineData("contains-sigma-final", "comparable")]
    [InlineData("contains-sigma-medial", "comparable")]
    [InlineData("contains-sigma-0345-ignorable", "comparable")]
    [InlineData("contains-sigma-boundary", "comparable")]
    [InlineData("contains-not-casefold", "comparable")]
    [InlineData("contains-astral", "comparable")]
    [InlineData("contains-no-nfc", "comparable")]
    [InlineData("contains-repeat-last", "comparable")]
    [InlineData("contains-blocker", "comparable")]
    [InlineData("contains-null-text", "source-error")]
    [InlineData("contains-numeric-needle", "source-error")]
    [InlineData("compare-starts-with-sensitive", "comparable")]
    [InlineData("compare-starts-with-lower", "comparable")]
    [InlineData("compare-ends-with-sensitive", "comparable")]
    [InlineData("compare-ends-with-lower", "comparable")]
    [InlineData("compare-equal-sensitive", "comparable")]
    [InlineData("compare-equal-lower", "comparable")]
    [InlineData("compare-empty-starts-with", "comparable")]
    [InlineData("compare-empty-ends-with", "comparable")]
    [InlineData("compare-empty-equal", "comparable")]
    [InlineData("compare-expansion-start", "comparable")]
    [InlineData("compare-expansion-end", "comparable")]
    [InlineData("compare-context-per-argument", "comparable")]
    [InlineData("compare-multi-ignorable", "comparable")]
    [InlineData("compare-mapped-modes", "comparable")]
    [InlineData("compare-invalid-unknown", "source-only-invalid-mode")]
    [InlineData("compare-invalid-equal", "source-only-invalid-mode")]
    public async Task RealNodesMatchPublishedBoundariesAndExposeAdmissionDifferences(string id, string classification)
    {
        var protocol = Resource("protocol", 41632, ProtocolSha);
        var reference = Resource("reference", 280652, ReferenceSha);
        ValidateProvenance(protocol, reference);
        var specification = Assert.Single(protocol["cases"]!.AsArray(), c => c!["id"]!.GetValue<string>() == id)!.AsObject();
        var expected = Assert.Single(reference["referenceOutputs"]!.AsArray(), c => c!["id"]!.GetValue<string>() == id)!.AsObject();
        Assert.Equal(classification, specification["portClassification"]!.GetValue<string>());
        Equal(specification["portClassification"], expected["portClassification"]);
        string type = specification["node"]!.GetValue<string>(); Equal(specification["node"], expected["node"]);
        Assert.Equal("frozen " + type + ".execute -> CPython builtin predicates/lower", expected["bodyOrigin"]!.GetValue<string>());
        Equal(specification["promptInputs"], expected["inputsAfter"]); Equal(specification["cache"], expected["cacheAfter"]);
        string specificationBefore = specification.ToJsonString(), expectedBefore = expected.ToJsonString();
        Equal(new JsonArray("off", "on", "off"), expected["repeats"]!["sequence"]);
        Assert.True(expected["repeats"]!["bitIdentical"]!.GetValue<bool>());
        var hashes = expected["repeats"]!["sha256"]!.AsArray(); Assert.Equal(3, hashes.Count);
        Assert.All(hashes, h => Assert.Matches("^[0-9a-f]{64}$", h!.GetValue<string>()));
        Assert.Single(hashes.Select(h => h!.GetValue<string>()).Distinct(StringComparer.Ordinal));

        var builtins = BuiltInNodes.CreateRegistry(); Assert.True(builtins.TryGet(type, out var actualNode));
        if (type == "StringContains") Assert.IsType<StringContainsNode>(actualNode);
        else Assert.IsType<StringCompareNode>(actualNode);
        CompareMetadata(expected, builtins.ToObjectInfo()[type]!.AsObject(), actualNode.Schema, specification);
        var captures = new List<Capture>(); var registry = new NodeRegistry();
        registry.Register(new ObservedNode(actualNode, captures));
        var prompt = new JsonObject(); var producerCalls = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var (producerId, value) in specification["cache"]!.AsObject())
        {
            string capturedId = producerId; var slots = value!.AsArray(); producerCalls.Add(producerId, 0);
            registry.Register(new Producer(new(producerId, producerId, "reference", [],
                slots.Select(_ => new OutputSchema("*", IsList: true)).ToArray()), context =>
            {
                producerCalls[capturedId]++;
                return new(slots.Select(slot => context.List(slot!.AsArray().Select(v => Decode(context, v)))).ToArray());
            }));
            prompt[producerId] = new JsonObject { ["class_type"] = producerId, ["inputs"] = new JsonObject() };
        }
        // Only genuine links or scalar literals occur here; no wrapper/shape adaptation.
        prompt[id] = new JsonObject { ["class_type"] = type, ["inputs"] = specification["promptInputs"]!.DeepClone() };
        string promptBefore = prompt.ToJsonString(); var engine = new EngineService(registry);
        var validation = engine.Validate(prompt, [id]); Assert.Empty(captures);
        Assert.All(producerCalls.Values, n => Assert.Equal(0, n));
        var events = new List<EngineEvent>();
        using (var result = await engine.ExecuteValuesAsync(prompt, [id], e => { events.Add(e); return ValueTask.CompletedTask; }))
        {
            if (classification == "source-only-invalid-mode")
            {
                // Source bypassed COMBO validation and merged its real body's None as no outputs.
                // The assertions of this record are integrity checks, not fake local body observations.
                Assert.Equal("returned", expected["status"]!.GetValue<string>());
                Assert.Empty(expected["outputs"]!.AsArray()); Assert.Empty(expected["ui"]!.AsObject());
                Assert.False(expected["hasSubgraph"]!.GetValue<bool>()); Assert.Empty(expected["reports"]!.AsArray());
                var body = Assert.Single(expected["observation"]!["bodyCalls"]!.AsArray())!;
                var bound = Assert.Single(expected["observation"]!["nestedInputReturns"]!.AsArray())!;
                Equal(specification["promptInputs"], body["arguments"]); Equal(body["arguments"], bound["arguments"]);
                Equal(Names(specification["promptInputs"]!.AsObject().Select(p => p.Key)), body["argumentOrder"]);
                Equal(body["argumentOrder"], bound["rootOrder"]);
                // Local rejection is a separate admission assertion, not an equivalent Python failure.
                Assert.False(validation.IsValid); Assert.Equal("error", result.Status); Assert.Empty(result.Outputs);
                var diagnostic = Assert.Single(validation.Diagnostics);
                Assert.Equal("invalid_input_type", diagnostic.Code); Assert.Equal("mode", diagnostic.InputName);
                Assert.Equal(id, diagnostic.NodeId); Assert.Equal(diagnostic, Assert.Single(result.Diagnostics));
                Assert.Empty(captures);
            }
            else
            {
                Assert.True(validation.IsValid); Assert.Empty(validation.Diagnostics);
                Assert.Equal("success", result.Status); Assert.Equal(id, Assert.Single(result.Outputs).Key);
                if (classification == "source-error")
                {
                    Assert.Equal("raised", expected["status"]!.GetValue<string>());
                    Assert.Equal("mapper", expected["stage"]!.GetValue<string>());
                    Assert.Equal(id == "contains-null-text" ? "TypeError" : "AttributeError", expected["errorType"]!.GetValue<string>());
                    var permitted = Assert.Single(specification["permittedSourceErrors"]!.AsArray())!;
                    Equal(permitted["stage"], expected["stage"]); Equal(permitted["type"], expected["errorType"]);
                    Assert.False(expected.ContainsKey("outputs")); Assert.Empty(expected["reports"]!.AsArray()); Assert.Empty(result.Diagnostics);
                    // Same prompt through the engine: STRING normalization is observed, not concealed.
                    // These local false results are not compared with a source Boolean: none was returned.
                    var normalized = specification["promptInputs"]!.DeepClone().AsObject();
                    normalized[id == "contains-null-text" ? "string" : "substring"] = id == "contains-null-text" ? "None" : "7";
                    Equal(normalized, Assert.Single(captures).Arguments);
                    Equal(Names(normalized.Select(p => p.Key)), Names(captures[0].Order));
                    Assert.False(Assert.Single(result.Outputs[id][0]).ToJson()!.GetValue<bool>());
                    await CompareTypedFailure(actualNode, expected, specification);
                }
                else
                {
                    Assert.Equal("comparable", classification); Assert.Equal("returned", expected["status"]!.GetValue<string>());
                    Assert.Empty(specification["permittedSourceErrors"]!.AsArray());
                    Equal(expected["outputs"], new JsonArray(result.Outputs[id].Select(slot =>
                        (JsonNode?)new JsonArray(slot.Select(EncodeOutput).ToArray())).ToArray()));
                    var sourceCalls = expected["observation"]!["bodyCalls"]!.AsArray();
                    Assert.Equal(sourceCalls.Count, captures.Count);
                    Assert.Equal(sourceCalls.Count, expected["observation"]!["nestedInputReturns"]!.AsArray().Count);
                    for (int i = 0; i < captures.Count; i++) CompareCapture(expected, captures[i], i);
                    Assert.Empty(expected["ui"]!.AsObject()); Assert.False(expected["hasSubgraph"]!.GetValue<bool>());
                    CompareReports(expected, result.Diagnostics, events);
                    if (id == "contains-blocker") Assert.Empty(captures);
                }
            }
        }
        // Only managed snapshots remain after operation disposal; no invocation wrapper is retained.
        foreach (var capture in captures) Equal(Names(capture.Order), Names(capture.Arguments.Select(p => p.Key)));
        Assert.All(producerCalls.Values, n => Assert.Equal(1, n));
        Assert.Equal(promptBefore, prompt.ToJsonString());
        Assert.Equal(specificationBefore, specification.ToJsonString()); Assert.Equal(expectedBefore, expected.ToJsonString());
        Assert.DoesNotContain(events, e => e.Type == "executed"); // Neither node emits UI.
    }

    private static async Task CompareTypedFailure(IRuntimeNode node, JsonObject expected, JsonObject specification)
    {
        using var borrowed = new RuntimeNodeContext();
        var inputs = specification["promptInputs"]!.AsObject().ToDictionary(p => p.Key,
            p => borrowed.Json(p.Value?.DeepClone()), StringComparer.Ordinal);
        var captures = new List<Capture>();
        using (var operation = new RuntimeNodeContext())
        {
            var error = await Assert.ThrowsAsync<ArgumentException>(() =>
                new ObservedNode(node, captures).ExecuteAsync(operation, inputs, CancellationToken.None).AsTask());
            Assert.Contains("requires", error.Message);
        }
        CompareCapture(expected, Assert.Single(captures), 0);
        Assert.Single(expected["observation"]!["bodyCalls"]!.AsArray());
        Assert.Single(expected["observation"]!["nestedInputReturns"]!.AsArray());
        foreach (var (name, value) in inputs) Equal(specification["promptInputs"]![name], value.ToJson());
    }

    private static void CompareCapture(JsonObject expected, Capture actual, int index)
    {
        var body = expected["observation"]!["bodyCalls"]![index]!;
        var bound = expected["observation"]!["nestedInputReturns"]![index]!;
        Equal(body["arguments"], actual.Arguments); Equal(body["argumentOrder"], Names(actual.Order));
        Equal(bound["arguments"], actual.Arguments); Equal(bound["rootOrder"], Names(actual.Order));
        _ = actual.Arguments["case_sensitive"]!.GetValue<bool>();
    }

    private static void CompareReports(JsonObject source, IReadOnlyList<EngineDiagnostic> diagnostics, List<EngineEvent> events)
    {
        var reports = source["reports"]!.AsArray(); var actual = events.Where(e => e.Type == "execution_error").ToArray();
        Assert.Equal(reports.Count, actual.Length); Assert.Equal(reports.Count, diagnostics.Count);
        for (int i = 0; i < reports.Count; i++)
        {
            Assert.Equal(reports[i]!["event"]!.GetValue<string>(), actual[i].Type);
            Assert.Equal(reports[i]!["payload"]!["node_id"]!.GetValue<string>(), actual[i].NodeId);
            Assert.Equal(reports[i]!["payload"]!["exception_message"]!.GetValue<string>(), actual[i].Message);
            Assert.Equal("execution_blocked", diagnostics[i].Code); Assert.Equal(actual[i].Message, diagnostics[i].Message);
        }
    }

    private static void CompareMetadata(JsonObject expected, JsonObject info, NodeSchema schema, JsonObject specification)
    {
        Equal(expected["objectInfo"], info); Equal(expected["inputTypes"], info["input"]);
        var expanded = NodeInputExpansion.Expand(schema, specification["promptInputs"]!.AsObject().Select(p => p.Key));
        var finalized = new JsonObject();
        foreach (bool required in new[] { true, false })
        {
            string partition = required ? "required" : "optional"; var inputs = new JsonObject();
            foreach (var input in expanded.Inputs.Where(i => i.Required == required))
                inputs.Add(input.Name, new JsonArray(input.Type, input.Options?.DeepClone() ?? new JsonObject()));
            finalized.Add(partition, inputs); Equal(expected["finalizedOrder"]![partition], Names(inputs.Select(p => p.Key)));
        }
        Equal(expected["finalized"], finalized);
    }

    private static void ValidateProvenance(JsonObject protocol, JsonObject reference)
    {
        Assert.True(reference["executed"]!.GetValue<bool>());
        Assert.Equal("5466f97312d8dbdbe5c656a6c31061895afc9af3", reference["collectorCommit"]!.GetValue<string>());
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", reference["backendCommit"]!.GetValue<string>());
        Assert.Equal("text-comparison-python312-v1", reference["protocolId"]!.GetValue<string>());
        Assert.Equal(ProtocolSha, reference["protocolSha256"]!.GetValue<string>());
        Assert.Equal("3.12.10", reference["python"]!.GetValue<string>());
        Assert.Equal("CPython", reference["implementation"]!.GetValue<string>());
        Assert.Equal("15.0.0", reference["unicodeVersion"]!.GetValue<string>());
        Equal(protocol["sources"], reference["sourceFiles"]); Equal(protocol["helperFiles"], reference["helperFiles"]);
        Equal(protocol["profile"], reference["profile"]);
        Assert.Equal(10, reference["sourceFiles"]!.AsArray().Count);
        Assert.Equal(75, reference["sourceFiles"]!.AsArray().Sum(s => s!["declarations"]!.AsArray().Count));
        var cases = protocol["cases"]!.AsArray(); var records = reference["referenceOutputs"]!.AsArray();
        Assert.Equal(40, reference["cases"]!.GetValue<int>()); Assert.Equal(40, cases.Count); Assert.Equal(40, records.Count);
        Assert.Equal(40, cases.Select(c => c!["id"]!.GetValue<string>()).Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(cases.Select(c => c!["id"]!.GetValue<string>()), records.Select(c => c!["id"]!.GetValue<string>()));
        foreach (var (kind, count) in new[] { ("comparable", 36), ("source-error", 2), ("source-only-invalid-mode", 2) })
        {
            Assert.Equal(count, cases.Count(c => c!["portClassification"]!.GetValue<string>() == kind));
            Assert.Equal(count, records.Count(c => c!["portClassification"]!.GetValue<string>() == kind));
        }
        foreach (var (name, size, sha) in new[] {
            ("protocol.json", 41632, ProtocolSha),
            ("reference.py", 16102, "df8297425891b959a3a657663ab972700a87792b189bc4644cfae5af6c098584"),
            ("README.md", 6066, "8ab8beffaf0103e94be4c5128bd2c4c81bb23c19a575f78a3d84fd359a429462") })
        {
            var pin = reference["laboratoryFiles"]!["labs/text-comparison-source/" + name]!;
            Assert.Equal(size, pin["bytes"]!.GetValue<int>()); Assert.Equal(sha, pin["sha256"]!.GetValue<string>());
        }
        // The 68 lower digests remain verbatim for separate tests, not counted as node cases.
    }

    private sealed record Capture(JsonObject Arguments, string[] Order);
    private sealed class ObservedNode(IRuntimeNode inner, List<Capture> captures) : IRuntimeNode
    {
        public NodeSchema Schema => inner.Schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            var snapshot = new JsonObject(); foreach (var (name, value) in inputs) snapshot.Add(name, Encode(value));
            captures.Add(new(snapshot, inputs.Keys.ToArray()));
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
    private static RuntimeValue Decode(RuntimeNodeContext context, JsonNode? value) => value is JsonObject o && o.Count == 1 && o.ContainsKey("$blocker")
        ? context.Blocker(o["$blocker"]?.GetValue<string>()) : context.Json(value?.DeepClone());
    private static JsonNode? Encode(RuntimeValue value) => value.Kind == RuntimeValueKind.Blocker
        ? new JsonObject { ["$blocker"] = value.Blocker.Message } : value.ToJson();
    private static JsonNode? EncodeOutput(RuntimeValue value)
    {
        if (value.Kind != RuntimeValueKind.Blocker)
        { Assert.Equal(RuntimeValueKind.Json, value.Kind); _ = value.ToJson()!.GetValue<bool>(); }
        return Encode(value);
    }
    private static JsonArray Names(IEnumerable<string> names) => new(names.Select(n => (JsonNode?)JsonValue.Create(n)).ToArray());
    private static void Equal(JsonNode? expected, JsonNode? actual) => Assert.True(JsonNode.DeepEquals(expected, actual),
        $"Expected {expected?.ToJsonString() ?? "null"}; actual {actual?.ToJsonString() ?? "null"}.");
    private static JsonObject Resource(string name, int size, string sha)
    {
        using var stream = typeof(TextComparisonSourceReferenceTests).Assembly.GetManifestResourceStream("ComfySharp.Core.Tests.Fixtures.text-comparison." + name + ".json");
        Assert.NotNull(stream); using var copy = new MemoryStream(); stream.CopyTo(copy); var bytes = copy.ToArray();
        Assert.Equal(size, bytes.Length); Assert.Equal(sha, Convert.ToHexStringLower(SHA256.HashData(bytes)));
        return JsonNode.Parse(bytes)!.AsObject();
    }
}
