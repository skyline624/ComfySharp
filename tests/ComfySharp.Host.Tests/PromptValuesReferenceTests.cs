using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;

namespace ComfySharp.Host.Tests;

/// <summary>Phase-specific comparison with a source validation/acquisition collection.
/// The source executed no node bodies. These tests execute real C# bodies but compare their
/// incoming arguments, not manufactured source outputs. PreviewAny is only a validation target.
/// Direct source acquisition and is_link have no public product equivalent and remain evidence.</summary>
public sealed class PromptValuesReferenceTests
{
    private const string SourceSha = "04af1fcf3c83c91d70cfd7a1d0b97f51a6eca5bdcf9c612ac9ea29fa420bee3f";
    private const string ProtocolSha = "dbd4038623e3313005cbb29393da978c0d8c9eb49a96860fb765ff217b6fe2ee";
    private static readonly string[] ValidationDifferences = ["slot-boolean-false", "slot-negative"];
    private static readonly string[] ArgumentMatches =
    [
        "link-string-id", "literal-null", "literal-empty-map", "literal-nested-map",
        "wrapped-empty-list", "wrapped-numeric-pair", "reserved-key-extra", "double-wrapper",
        "optional-null", "int-from-bool", "float-from-string", "string-from-null", "boolean-from-wrapped-empty"
    ];

    [Theory]
    [InlineData("raw-empty-list")]
    [InlineData("raw-numeric-pair")]
    [InlineData("link-string-id")]
    [InlineData("malformed-link-length")]
    [InlineData("slot-string")]
    [InlineData("slot-null")]
    [InlineData("slot-boolean-false")]
    [InlineData("slot-float-integral")]
    [InlineData("slot-float-fraction")]
    [InlineData("slot-negative")]
    [InlineData("slot-out-of-range")]
    [InlineData("numeric-node-id")]
    [InlineData("boolean-node-id")]
    [InlineData("missing-node")]
    [InlineData("literal-null")]
    [InlineData("literal-empty-map")]
    [InlineData("literal-nested-map")]
    [InlineData("wrapped-empty-list")]
    [InlineData("wrapped-numeric-pair")]
    [InlineData("wrapped-link-shaped")]
    [InlineData("reserved-key-extra")]
    [InlineData("double-wrapper")]
    [InlineData("required-absent")]
    [InlineData("optional-null")]
    [InlineData("int-from-bool")]
    [InlineData("float-from-string")]
    [InlineData("string-from-null")]
    [InlineData("boolean-from-wrapped-empty")]
    public async Task ValidationAndRealNodeArgumentsRespectTheExplicitSourcePartition(string id)
    {
        var source = Resource("prompt-values.reference.json", 677389, SourceSha);
        var protocol = Resource("prompt-values.protocol.json", 17911, ProtocolSha);
        Assert.True(source["executed"]!.GetValue<bool>());
        Assert.False(source["sourceNodeExecution"]!.GetValue<bool>());
        Assert.Equal("8dad64f0cf9b04548ee722b7533776b25d3b4c46", source["collectorCommit"]!.GetValue<string>());
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", source["backendCommit"]!.GetValue<string>());
        Assert.Equal(ProtocolSha, source["provenance"]!["protocol"]!["sha256"]!.GetValue<string>());
        Assert.Equal("prompt-values-boundaries-v1", source["protocolId"]!.GetValue<string>());
        var cases = source["referenceOutputs"]!.AsArray();
        Assert.Equal(28, cases.Count);
        Assert.Equal(16, cases.Count(c => c!["validation"]!["accepted"]!.GetValue<bool>()));
        Assert.Equal(protocol["cases"]!.AsArray().Select(c => c!["id"]!.GetValue<string>()),
            cases.Select(c => c!["id"]!.GetValue<string>()));
        Assert.Equal(ArgumentMatches, cases.Where(c => c!["validation"]!["accepted"]!.GetValue<bool>() &&
            !ValidationDifferences.Contains(c["id"]!.GetValue<string>(), StringComparer.Ordinal) &&
            c["id"]!.GetValue<string>() != "wrapped-link-shaped").Select(c => c!["id"]!.GetValue<string>()));
        var specification = Assert.Single(protocol["cases"]!.AsArray(), c => c!["id"]!.GetValue<string>() == id)!.AsObject();
        var expected = Assert.Single(cases, c => c!["id"]!.GetValue<string>() == id)!.AsObject();
        string expectedBefore = expected.ToJsonString();
        var prompt = Prompt(protocol, specification);
        string promptBefore = prompt.ToJsonString();
        string type = specification["classType"]!.GetValue<string>();
        Equal(expected["validation"]!["promptBefore"], Encode(prompt));
        Assert.True(expected["originalPromptUnchanged"]!.GetValue<bool>());
        Assert.Equal(3, expected["repeats"]!["count"]!.GetValue<int>());
        Assert.True(expected["repeats"]!["freshSourceNamespaces"]!.GetValue<bool>());
        Assert.True(expected["repeats"]!["bitIdentical"]!.GetValue<bool>());
        var hashes = expected["repeats"]!["sha256"]!.AsArray();
        Assert.Equal(3, hashes.Count);
        Assert.All(hashes, h => Assert.Matches("^[0-9a-f]{64}$", h!.GetValue<string>()));
        Assert.Single(hashes.Select(h => h!.GetValue<string>()).Distinct(StringComparer.Ordinal));

        // Preserve original absence and numeric token types without simulating source is_link.
        bool present = specification["present"]!.GetValue<bool>();
        Assert.Equal(present, expected["recognitionOriginal"]!["inputPresent"]!.GetValue<bool>());
        Assert.Equal(present, expected["recognitionOriginal"]!["predicateInvoked"]!.GetValue<bool>());
        if (present) Equal(expected["recognitionOriginal"]!["value"], Encode(specification["value"]));
        else Assert.False(expected["recognitionOriginal"]!.AsObject().ContainsKey("value"));
        if (id == "slot-float-integral")
            Assert.Equal("0.0", prompt["subject"]!["inputs"]!["inputs.input0"]![1]!.ToJsonString());
        AssertAcquisitionEvidence(expected["acquisitionOriginal"]!.AsObject(), expected["validation"]!["promptBefore"]!, protocol);
        AssertSourceValidation(expected["validation"]!.AsObject());

        var (engine, calls) = ObservedEngine();
        var validation = engine.Validate(prompt, ["preview"]);
        bool sourceAccepted = expected["validation"]!["accepted"]!.GetValue<bool>();
        bool localDifference = ValidationDifferences.Contains(id, StringComparer.Ordinal);
        if (localDifference)
        {
            Assert.True(sourceAccepted);
            Assert.False(validation.IsValid); // Do not repair bool or negative slots to match Python.
        }
        else Assert.Equal(sourceAccepted, validation.IsValid);
        Assert.Empty(calls);
        Assert.Equal(promptBefore, prompt.ToJsonString());
        var repeated = engine.Validate(prompt, ["preview"]);
        Assert.Equal(validation.IsValid, repeated.IsValid);
        Assert.Equal(promptBefore, prompt.ToJsonString());
        Assert.Empty(calls);

        if (!sourceAccepted)
        {
            Assert.Equal("not-run", expected["acquisitionAfterValidation"]!["status"]!.GetValue<string>());
            Assert.Equal("source-validation-rejected", expected["acquisitionAfterValidation"]!["reason"]!.GetValue<string>());
        }
        else AssertAcquisitionEvidence(expected["acquisitionAfterValidation"]!.AsObject(), expected["validation"]!["promptAfter"]!, protocol);

        if (!validation.IsValid)
        {
            Assert.Empty(validation.ValidTargets);
            Assert.Contains(validation.Diagnostics, d => d.Code == LocalRejection(id));
            using var rejected = await engine.ExecuteValuesAsync(prompt, ["subject"]);
            Assert.Equal("error", rejected.Status);
            Assert.Empty(rejected.Outputs);
            Assert.Empty(calls);
        }
        else
        {
            Assert.Equal(new[] { "preview" }, validation.ValidTargets);
            Assert.Empty(validation.Diagnostics);
            await ExecuteAndCompareArguments(engine, calls, prompt, type,
                expected["acquisitionAfterValidation"]!.AsObject(), id == "wrapped-link-shaped");
        }
        Assert.Equal(promptBefore, prompt.ToJsonString());

        if (specification["secondValidationPass"]!.GetValue<bool>())
        {
            var second = expected["secondValidation"]!.AsObject();
            AssertSourceValidation(second);
            Equal(expected["validation"]!["promptAfter"], second["promptBefore"]);
            if (id == "wrapped-empty-list")
            {
                Assert.True(repeated.IsValid); // Same original C# input, still wrapped and immutable.
                Assert.False(second["accepted"]!.GetValue<bool>()); // Source re-used its unwrapped [].
            }
            // Explicit new input: this is the source's mutated first-pass prompt, never a
            // replacement for the original-case test (especially wrapped-link-shaped).
            var adapted = Decode(expected["validation"]!["promptAfter"]!)!.AsObject();
            string adaptedBefore = adapted.ToJsonString();
            var (adaptedEngine, adaptedCalls) = ObservedEngine();
            var adaptedValidation = adaptedEngine.Validate(adapted, ["preview"]);
            Assert.Equal(second["accepted"]!.GetValue<bool>(), adaptedValidation.IsValid);
            Assert.Empty(adaptedCalls);
            if (adaptedValidation.IsValid)
            {
                AssertAcquisitionEvidence(expected["acquisitionAfterSecondValidation"]!.AsObject(), second["promptAfter"]!, protocol);
                await ExecuteAndCompareArguments(adaptedEngine, adaptedCalls, adapted, type,
                    expected["acquisitionAfterSecondValidation"]!.AsObject(), false);
            }
            else
            {
                Assert.Equal("not-run", expected["acquisitionAfterSecondValidation"]!["status"]!.GetValue<string>());
                Assert.Equal("source-second-validation-rejected", expected["acquisitionAfterSecondValidation"]!["reason"]!.GetValue<string>());
                Assert.Contains(adaptedValidation.Diagnostics, d => d.Code == "invalid_link");
            }
            Assert.Equal(adaptedBefore, adapted.ToJsonString());
        }
        else Assert.False(expected.ContainsKey("secondValidation"));
        Assert.Equal(expectedBefore, expected.ToJsonString());
    }

    private static async Task ExecuteAndCompareArguments(EngineService engine, List<Call> calls,
        JsonObject prompt, string subjectType, JsonObject acquisition, bool envelopeDifference)
    {
        using var result = await engine.ExecuteValuesAsync(prompt, ["subject"]);
        Assert.Equal("success", result.Status);
        Assert.Empty(result.Diagnostics);
        Assert.Equal("subject", Assert.Single(result.Outputs).Key);
        Assert.Single(result.Outputs["subject"]); // Structure only: source executed no body/output.
        Assert.DoesNotContain(calls, c => c.Type == "PreviewAny");
        var subject = Assert.Single(calls, c => c.Type == subjectType);
        var flat = new JsonObject();
        if (subjectType == "CreateList")
        {
            var group = subject.Arguments["inputs"]!.AsObject();
            foreach (var (name, value) in group) flat.Add("inputs." + name, value?.DeepClone());
        }
        else foreach (var (name, value) in subject.Arguments) flat.Add(name, new JsonArray(value?.DeepClone()));
        if (envelopeDifference)
        {
            Equal(Encode(new JsonObject { ["inputs.input0"] = new JsonArray(new JsonArray("id", 0)) }), Encode(flat));
            Assert.False(JsonNode.DeepEquals(acquisition["flat"], Encode(flat)));
            Assert.DoesNotContain(calls, c => c.Type == "PrimitiveString");
        }
        else Equal(acquisition["flat"], Encode(flat));
        Equal(acquisition["flatOrder"], new JsonArray(flat.Select(p => (JsonNode?)JsonValue.Create(p.Key)).ToArray()));
        bool linked = prompt["subject"]!["inputs"]!.AsObject().Any(p => p.Value is JsonArray);
        Assert.Equal(linked ? 2 : 1, calls.Count); // Subject plus the real reachable producer, if any.
    }

    private static void AssertSourceValidation(JsonObject phase)
    {
        Assert.Equal("returned", phase["status"]!.GetValue<string>());
        Assert.Equal("tuple", phase["result"]!["kind"]!.GetValue<string>());
        var tuple = phase["result"]!["items"]!.AsArray();
        Assert.Equal(4, tuple.Count);
        Assert.Equal("bool", tuple[0]!["kind"]!.GetValue<string>());
        Assert.Equal(phase["accepted"]!.GetValue<bool>(), tuple[0]!["value"]!.GetValue<bool>());
    }

    private static void AssertAcquisitionEvidence(JsonObject acquisition, JsonNode typedPrompt, JsonObject protocol)
    {
        Assert.Contains(acquisition["status"]!.GetValue<string>(), new[] { "returned", "raised" });
        Equal(Encode(Decode(typedPrompt)!["subject"]!["inputs"]), acquisition["inputsAfter"]);
        Equal(Encode(protocol["infrastructure"]!["cache"]), acquisition["cacheAfter"]);
        // These are source-only phases. Reading their evidence is not a fabricated product API.
        if (acquisition["status"]!.GetValue<string>() == "raised")
            Assert.Equal("TypeError", acquisition["errorType"]!.GetValue<string>());
        else Assert.Equal("dict", acquisition["flat"]!["kind"]!.GetValue<string>());
    }

    private static string LocalRejection(string id) => id switch
    {
        "slot-out-of-range" => "invalid_output_index",
        "missing-node" => "missing_node",
        "required-absent" => "required_input_missing",
        _ => "invalid_link"
    };

    private static JsonObject Prompt(JsonObject protocol, JsonObject specification)
    {
        var prompt = protocol["infrastructure"]!["producerPrompt"]!.DeepClone().AsObject();
        var inputs = new JsonObject();
        if (specification["optionalAnchor"]!.GetValue<bool>()) inputs["inputs.input0"] = "anchor";
        if (specification["present"]!.GetValue<bool>())
            inputs[specification["inputName"]!.GetValue<string>()] = specification["value"]?.DeepClone();
        prompt["subject"] = new JsonObject { ["class_type"] = specification["classType"]!.DeepClone(), ["inputs"] = inputs };
        prompt["preview"] = new JsonObject { ["class_type"] = protocol["infrastructure"]!["outputClass"]!.DeepClone(),
            ["inputs"] = protocol["infrastructure"]!["outputInput"]!.DeepClone() };
        return prompt;
    }

    private static (EngineService Engine, List<Call> Calls) ObservedEngine()
    {
        var calls = new List<Call>(); var observed = new NodeRegistry();
        foreach (var real in TensorNodes.CreateRegistry().Nodes) observed.Register(new ObservedNode(real, calls));
        return (new EngineService(observed), calls);
    }
    private sealed record Call(string Type, JsonObject Arguments);
    private sealed class ObservedNode(IRuntimeNode real, List<Call> calls) : IRuntimeNode
    {
        public NodeSchema Schema => real.Schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            calls.Add(new(Schema.ClassType, new JsonObject(inputs.Select(p => KeyValuePair.Create(p.Key, p.Value.ToJson())))));
            return real.ExecuteAsync(context, inputs, cancellationToken);
        }
        public IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> inputs) =>
            real.GetRequiredLazyInputs(inputs);
    }

    // Typed codec preserves dictionary order, scalar kinds and exact finite float64 bits.
    // It never turns Python tuples into prompt arrays: tuples only describe source API returns.
    private static JsonObject Encode(JsonNode? value)
    {
        if (value is null) return new() { ["kind"] = "null" };
        if (value is JsonObject o) return new() { ["kind"] = "dict", ["items"] = new JsonArray(o.Select(p =>
            (JsonNode?)new JsonArray(JsonValue.Create(p.Key), Encode(p.Value))).ToArray()) };
        if (value is JsonArray a) return new() { ["kind"] = "list", ["items"] = new JsonArray(a.Select(v => (JsonNode?)Encode(v)).ToArray()) };
        var scalar = value.AsValue();
        if (scalar.TryGetValue<bool>(out var b)) return new() { ["kind"] = "bool", ["value"] = b };
        if (scalar.TryGetValue<string>(out var s)) return new() { ["kind"] = "string", ["value"] = s };
        if (scalar.TryGetValue<int>(out var i32)) return new() { ["kind"] = "int", ["decimal"] = i32.ToString(CultureInfo.InvariantCulture) };
        if (scalar.TryGetValue<long>(out var i)) return new() { ["kind"] = "int", ["decimal"] = i.ToString(CultureInfo.InvariantCulture) };
        double f = scalar.GetValue<double>();
        Assert.True(double.IsFinite(f));
        Span<byte> bytes = stackalloc byte[8]; BinaryPrimitives.WriteInt64LittleEndian(bytes, BitConverter.DoubleToInt64Bits(f));
        // Python repr is not recreated. Strip it only from typed float metadata on both sides.
        return new() { ["kind"] = "float64", ["bitsLE"] = Convert.ToHexString(bytes).ToLowerInvariant() };
    }
    private static JsonNode? Decode(JsonNode encoded)
    {
        string kind = encoded["kind"]!.GetValue<string>();
        return kind switch
        {
            "null" => null,
            "bool" => JsonValue.Create(encoded["value"]!.GetValue<bool>()),
            "string" => JsonValue.Create(encoded["value"]!.GetValue<string>()),
            // Parse the integer token as JSON, as at the real prompt boundary. A JsonValue<long>
            // constructed in managed code would not expose TryGetValue<int> for a link's slot.
            "int" => JsonNode.Parse(encoded["decimal"]!.GetValue<string>()),
            "float64" => JsonValue.Create(BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(
                Convert.FromHexString(encoded["bitsLE"]!.GetValue<string>())))),
            "list" => new JsonArray(encoded["items"]!.AsArray().Select(v => Decode(v!)).ToArray()),
            "dict" => new JsonObject(encoded["items"]!.AsArray().Select(p => KeyValuePair.Create(p![0]!.GetValue<string>(), Decode(p[1]!)))),
            _ => throw new InvalidDataException("Source API tuple or unknown kind cannot become a prompt value: " + kind)
        };
    }
    private static JsonNode? Comparable(JsonNode? value) => value switch
    {
        JsonObject o => new JsonObject(o.Where(p => !(p.Key == "repr" && o["kind"]?.GetValue<string>() == "float64"))
            .Select(p => KeyValuePair.Create(p.Key, Comparable(p.Value)))),
        JsonArray a => new JsonArray(a.Select(Comparable).ToArray()),
        _ => value?.DeepClone()
    };
    private static void Equal(JsonNode? expected, JsonNode? actual) =>
        Assert.True(JsonNode.DeepEquals(Comparable(expected), Comparable(actual)), $"Expected {expected}; actual {actual}");
    private static JsonObject Resource(string name, int size, string hash)
    {
        using var stream = typeof(PromptValuesReferenceTests).Assembly.GetManifestResourceStream("ComfySharp.Host.Tests.Fixtures." + name);
        Assert.NotNull(stream); using var buffer = new MemoryStream(); stream.CopyTo(buffer);
        byte[] raw = buffer.ToArray();
        Assert.Equal(size, raw.Length);
        Assert.Equal(hash, Convert.ToHexString(SHA256.HashData(raw)).ToLowerInvariant());
        return JsonNode.Parse(raw)!.AsObject();
    }
}
