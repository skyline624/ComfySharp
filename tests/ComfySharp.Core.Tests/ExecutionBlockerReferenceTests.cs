using System.Security.Cryptography;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>Published V1 mapper observations projected onto real EngineService operations.
/// Fixture tags are decoded only by source nodes; they are never product prompt literals.</summary>
public sealed class ExecutionBlockerReferenceTests
{
    private const string ReferenceSha = "062a71742134c9fdc93c78b48348446c077ae5d20c1309772ec3c9b43cba2a72";
    private const string ProtocolSha = "861118bdd79dc584185ad6f01fd63d89e5fde8345dd071913469b99ce1e40916";

    [Theory]
    [InlineData("silent-middle")]
    [InlineData("message-middle")]
    [InlineData("empty-message")]
    [InlineData("prompt-order")]
    [InlineData("silent-first")]
    [InlineData("repeat-last-blocker")]
    [InlineData("input-is-list")]
    [InlineData("nested-list")]
    [InlineData("nested-map")]
    [InlineData("output-is-list")]
    [InlineData("partial-slot")]
    [InlineData("whole-producer")]
    [InlineData("list-producer-block")]
    [InlineData("async-middle")]
    [InlineData("input-list-nested")]
    [InlineData("all-empty")]
    [InlineData("mixed-empty")]
    [InlineData("input-list-mixed-empty")]
    [InlineData("ui-survivors")]
    [InlineData("producer-block-ui")]
    [InlineData("ordinary-error")]
    public async Task MapperCallsOutputsUiAndBlockMessagesMatchPublishedSource(string id)
    {
        var expected = Case(id);
        var fixtureInputs = expected["inputsAfter"]!.AsObject();
        string inputBefore = fixtureInputs.ToJsonString();
        var calls = new JsonArray();
        var events = new List<EngineEvent>();
        var registry = new NodeRegistry();
        var prompt = Sources(registry, fixtureInputs, out var arguments);
        bool inputIsList = id is "input-is-list" or "input-list-nested" or "input-list-mixed-empty";
        OutputSchema[] outputs = id switch
        {
            "whole-producer" or "partial-slot" => [new("*"), new("*")],
            "output-is-list" or "list-producer-block" => [new("*", IsList: true)],
            _ => [new("*")]
        };
        // Deliberately reverse schema order: blocker priority belongs to the prompt.
        var schema = new NodeSchema("Consumer", "Consumer", "reference",
            fixtureInputs.Select(p => new InputSchema(p.Key, "*")).Reverse().ToArray(), outputs, InputIsList: inputIsList);
        registry.Register(new Node(schema, async (context, inputs, token) =>
        {
            if (id == "async-middle") await Task.Yield();
            token.ThrowIfCancellationRequested();
            calls.Add(EncodeInputs(inputs));
            return id switch
            {
                "output-is-list" => new([context.List([inputs["x"]])]),
                "partial-slot" => new([context.Blocker("slot"), inputs["x"]]),
                "whole-producer" => NodeExecutionOutput.Blocked("producer"),
                "list-producer-block" => NodeExecutionOutput.Blocked(),
                "all-empty" => new([context.Json(JsonValue.Create(1))]),
                "ui-survivors" => new([inputs["x"]], new() { ["values"] = new JsonArray(Encode(inputs["x"])) }),
                "producer-block-ui" => NodeExecutionOutput.Blocked("producer", new() { ["values"] = new JsonArray("producer") }),
                "ordinary-error" => throw new InvalidOperationException("fixture ordinary error"),
                _ => new([inputs["x"]])
            };
        }));
        prompt[id] = new JsonObject { ["class_type"] = "Consumer", ["inputs"] = arguments };
        using var result = await new EngineService(registry).ExecuteValuesAsync(prompt, [id], e =>
        { events.Add(e); return ValueTask.CompletedTask; });
        Equal(expected["calls"], calls);
        Assert.Equal(inputBefore, fixtureInputs.ToJsonString());

        bool raised = expected["status"]!.GetValue<string>() == "raised";
        Assert.Equal(raised ? "error" : "success", result.Status);
        if (raised)
        {
            // EngineService translates node exceptions into diagnostics. Python exception
            // classes and mapper interrupt/context counters have no public engine analogue.
            Assert.Empty(result.Outputs);
            var diagnostic = Assert.Single(result.Diagnostics);
            Assert.Equal("execution_error", diagnostic.Code);
            Assert.Equal(id, diagnostic.NodeId);
            Assert.Equal(id == "mixed-empty" ? "IndexError" : "FixtureNodeError", expected["errorType"]!.GetValue<string>());
            Assert.Equal(id == "mixed-empty" ? "Cannot repeat the last item of an empty execution list." : "fixture ordinary error", diagnostic.Message);
            Assert.Empty(expected["reports"]!.AsArray());
        }
        else
        {
            Equal(expected["outputs"], new JsonArray(result.Outputs[id].Select(slot =>
                (JsonNode?)new JsonArray(slot.Select(Encode).ToArray())).ToArray()));
            var actualUi = events.Where(e => e.Type == "executed" && e.NodeId == id).ToArray();
            var expectedUi = expected["ui"]!.AsObject();
            if (expectedUi.Count == 0) Assert.Empty(actualUi);
            else Equal(expectedUi, Assert.Single(actualUi).Output);
            var reports = expected["reports"]!.AsArray();
            var actualReports = events.Where(e => e.Type == "execution_error").ToArray();
            Assert.Equal(reports.Count, actualReports.Length);
            Assert.Equal(reports.Count, result.Diagnostics.Count);
            for (int i = 0; i < reports.Count; i++)
            {
                Assert.Equal(reports[i]!["type"]!.GetValue<string>(), actualReports[i].Type);
                Assert.Equal(reports[i]!["payload"]!["node_id"]!.GetValue<string>(), actualReports[i].NodeId);
                Assert.Equal(reports[i]!["payload"]!["exception_message"]!.GetValue<string>(), actualReports[i].Message);
                Assert.Equal("execution_blocked", result.Diagnostics[i].Code);
                Assert.Equal(actualReports[i].Message, result.Diagnostics[i].Message);
            }
        }
    }

    [Theory]
    [InlineData("lazy-mixed-rows")]
    [InlineData("lazy-all-blocked")]
    public async Task AggregateLazySnapshotPreservesSourceActiveRowsAndRequestedNames(string id)
    {
        var expected = Case(id);
        var registry = new NodeRegistry();
        var inputs = expected["inputsAfter"]!.AsObject();
        var prompt = Sources(registry, inputs, out var arguments);
        var rows = new JsonArray();
        var requested = new List<string>();
        var branches = new List<string>();
        int lazyCalls = 0;
        foreach (string name in new[] { "when_false", "when_true" })
        {
            string captured = name;
            registry.Register(new Node(new(name, name, "reference", [], [new("*")]), (c, _, _) =>
            { branches.Add(captured); return ValueTask.FromResult(new NodeExecutionOutput([c.Json(JsonValue.Create(captured))])); }));
            prompt[name] = new JsonObject { ["class_type"] = name, ["inputs"] = new JsonObject() };
            arguments[name] = new JsonArray(name, 0);
        }
        registry.Register(new Node(new("Lazy", "Lazy", "reference",
            inputs.Select(p => new InputSchema(p.Key, "*")).Concat(new[]
                { new InputSchema("when_false", "*", Lazy: true), new InputSchema("when_true", "*", Lazy: true) }).ToArray(), [new("*")]),
            (c, _, _) => ValueTask.FromResult(new NodeExecutionOutput([c.Json(null)])),
            snapshot =>
            {
                lazyCalls++;
                for (int i = 0; i < snapshot["select"].Count; i++)
                {
                    rows.Add(new JsonObject(snapshot.Select(p => KeyValuePair.Create(p.Key, Encode(p.Value[i])))));
                    requested.Add(snapshot["select"][i].ToJson()!.GetValue<bool>() ? "when_true" : "when_false");
                }
                return requested.Distinct(StringComparer.Ordinal).ToArray();
            }));
        prompt[id] = new JsonObject { ["class_type"] = "Lazy", ["inputs"] = arguments };
        using var result = await new EngineService(registry).ExecuteValuesAsync(prompt, [id]);
        Assert.Equal("success", result.Status);
        Equal(expected["calls"], rows);
        var sourceNames = expected["mappedReturns"]!.AsArray().OfType<JsonArray>()
            .SelectMany(a => a.Select(n => n!.GetValue<string>())).ToArray();
        Assert.Equal(sourceNames, requested);
        Assert.Equal(sourceNames.Distinct(StringComparer.Ordinal), branches);
        Assert.Equal(expected["calls"]!.AsArray().Count == 0 ? 0 : 1, lazyCalls);
        // No comparison of final job outputs/reports: source captured the lazy mapper
        // without execution_block_cb, while the engine subsequently executes the node.
    }

    private static JsonObject Case(string id)
    {
        using var stream = typeof(ExecutionBlockerReferenceTests).Assembly.GetManifestResourceStream(
            "ComfySharp.Core.Tests.Fixtures.execution-blocker.reference.json")!;
        Assert.NotNull(stream);
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        byte[] raw = buffer.ToArray();
        Assert.Equal(25302, raw.Length);
        Assert.Equal(ReferenceSha, Convert.ToHexString(SHA256.HashData(raw)).ToLowerInvariant());
        var reference = JsonNode.Parse(raw)!.AsObject();
        Assert.True(reference["executed"]!.GetValue<bool>());
        Assert.Equal("fffa6b06bbde2a76e2fe40e9097f03c7b169cee8", reference["collectorCommit"]!.GetValue<string>());
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", reference["backendCommit"]!.GetValue<string>());
        Assert.Equal(ProtocolSha, reference["protocolSha256"]!.GetValue<string>());
        Assert.Equal(6, reference["sourceDeclarations"]!.AsArray().Count);
        var cases = reference["referenceOutputs"]!.AsArray();
        Assert.Equal(24, cases.Count);
        return Assert.Single(cases, c => c!["id"]!.GetValue<string>() == id)!.AsObject();
    }

    private static JsonObject Sources(NodeRegistry registry, JsonObject inputs, out JsonObject arguments)
    {
        var prompt = new JsonObject();
        arguments = new JsonObject();
        foreach (var (name, data) in inputs)
        {
            string source = "source_" + name;
            var values = data!.AsArray();
            registry.Register(new Node(new(source, source, "reference", [], [new("*", IsList: true)]),
                (c, _, _) => ValueTask.FromResult(new NodeExecutionOutput([c.List(values.Select(v => Decode(c, v)))]))));
            prompt[source] = new JsonObject { ["class_type"] = source, ["inputs"] = new JsonObject() };
            arguments[name] = new JsonArray(source, 0);
        }
        return prompt;
    }

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
    private static void Equal(JsonNode? expected, JsonNode? actual) =>
        Assert.True(JsonNode.DeepEquals(expected, actual), $"Expected {expected}; actual {actual}");

    private sealed class Node(NodeSchema schema,
        Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, CancellationToken, ValueTask<NodeExecutionOutput>> run,
        Func<IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>>, IReadOnlyCollection<string>>? lazy = null) : IRuntimeNode
    {
        public NodeSchema Schema => schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) => run(context, inputs, cancellationToken);
        public IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> resolvedInputs) => lazy?.Invoke(resolvedInputs) ?? [];
    }
}
