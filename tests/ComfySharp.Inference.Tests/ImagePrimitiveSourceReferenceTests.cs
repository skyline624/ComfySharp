using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Twenty-eight real node calls against the three retained Windows CPU/F32 source records.
/// Exact payloads/layouts and a separate post-body copy observation; no mapper or HTTP parity claim.</summary>
[Collection("Classical VAE")]
public sealed class ImagePrimitiveSourceReferenceTests
{
    private const string ProtocolSha = "26ce418d969ee583ab077880530d20b6c11530094d7bbb97956bc5feaf56c0b4";
    private const string ReferenceSha = "6b31ffb26843506bed8ee0e41cc97e91087e5f46ce01e39cdedf27fa3e218a62";
    private static readonly JsonSerializerOptions Compact = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
    private static readonly Lazy<Corpus> Source = new(LoadCorpus);
    private sealed record Corpus(JsonObject Protocol, JsonObject Reference);

    [Theory]
    [InlineData("empty-black")]
    [InlineData("empty-white")]
    [InlineData("empty-low-blue")]
    [InlineData("empty-asymmetric")]
    [InlineData("empty-red-orange")]
    [InlineData("empty-green")]
    [InlineData("empty-rgb")]
    [InlineData("invert-rgb")]
    [InlineData("invert-rgba-alpha")]
    [InlineData("invert-batch")]
    [InlineData("invert-rgba-strided")]
    [InlineData("invert-rgb-nchw-view")]
    [InlineData("invert-rgb-offset")]
    [InlineData("invert-rgba-alpha-offset")]
    [InlineData("repeat-identity-copy")]
    [InlineData("repeat-whole-sequence")]
    [InlineData("repeat-rgba")]
    [InlineData("repeat-strided")]
    [InlineData("repeat-nchw-view")]
    [InlineData("repeat-offset")]
    [InlineData("repeat-triple-batch")]
    [InlineData("extract-first")]
    [InlineData("extract-middle")]
    [InlineData("extract-last")]
    [InlineData("extract-negative-last")]
    [InlineData("extract-negative-full")]
    [InlineData("extract-negative-clamped")]
    [InlineData("extract-positive-clamped")]
    public async Task RegisteredNodeMatchesExactSourceAndRetainsIndependentImage(string id)
    {
        var corpus = Source.Value;
        var specification = Assert.Single(corpus.Protocol["cases"]!.AsArray(), c => Text(c!, "id") == id)!;
        var expected = Assert.Single(corpus.Reference["cases"]!.AsArray(), c => Text(c!, "id") == id)!;
        string nodeId = Text(specification, "node");
        var repeats = expected["repeats"]!.AsArray();
        var registry = TensorNodes.CreateRegistry();
        Assert.True(registry.TryGet(nodeId, out var node));
        foreach (var repeat in repeats)
            CompareSchema(node.Schema, registry.ToObjectInfo()[nodeId]!.AsObject(), repeat!["schema"]!);

        NativeRuntimeBootstrap.Initialize();
        Assert.True(BitConverter.IsLittleEndian, "The admitted fixture layout is little-endian Float32.");
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        using var inputOwner = new RuntimeNodeContext();
        var inputs = new Dictionary<string, RuntimeValue>(StringComparer.Ordinal);
        Tensor? image = null;
        if (specification["image"] is JsonObject descriptor)
        {
            var raw = Convert.FromBase64String(Text(descriptor, "storageLittleEndianBase64"));
            Assert.Equal(descriptor["storageElements"]!.GetValue<int>() * sizeof(float), raw.Length);
            Assert.Equal(Text(descriptor, "storageSha256"), Hash(raw));
            // Native allocation first, then exact input bytes; no managed-array backing or layout normalization.
            var storage = empty(new long[] { raw.Length / sizeof(float) }, dtype: ScalarType.Float32, device: CPU);
            raw.AsSpan().CopyTo(storage.bytes);
            Assert.Equal(Hash(raw), Hash(storage.bytes));
            Assert.True(Aligned64(storage));
            image = storage.as_strided(Longs(descriptor["shape"]!), Longs(descriptor["stride"]!), descriptor["storageOffset"]!.GetValue<long>());
            inputs.Add("image", inputOwner.Own(image));
            image.DetachFromDisposeScope();
            foreach (var repeat in repeats) CompareTensor(image, repeat!["inputBefore"]!);
        }
        foreach (var pair in specification["arguments"]!.AsObject()) inputs.Add(pair.Key, inputOwner.Json(pair.Value));

        RuntimeValue retained;
        using (var invocation = new RuntimeNodeContext())
        {
            // Actual registered node, directly as in the source laboratory. No synthetic implementation or engine coercion.
            var pending = node.ExecuteAsync(invocation, inputs, default);
            Assert.True(pending.IsCompletedSuccessfully, "These image nodes have synchronous bodies.");
            var returned = await pending; // Already completed above: native dispose scopes stay on this thread.
            Assert.Null(returned.Ui);
            retained = Assert.Single(returned.Result).Retain();
        }
        using (retained)
        {
            Assert.Equal(RuntimeValueKind.Native, retained.Kind);
            var result = retained.GetNative<Tensor>(); // Borrowed from retained, never disposed separately.
            foreach (var repeat in repeats)
            {
                Equal(specification["arguments"], repeat!["arguments"]);
                CompareTensor(result, repeat["output"]!);
                if (image is not null) CompareTensor(image, repeat["inputAfterBody"]!);
            }
            if (image is null)
            {
                Assert.All(repeats, r => Assert.Equal("not_applicable_no_image_input", Text(r!["copyObservation"]!, "status")));
                return;
            }

            var mutationIndex = Longs(specification["mutationIndex"]!);
            Tensor scalar = image;
            for (int dimension = 0; dimension < 4; dimension++) scalar = scalar.select(0, mutationIndex[dimension]);
            foreach (var repeat in repeats)
            {
                var observation = repeat!["copyObservation"]!;
                Equal(specification["mutationIndex"], observation["mutationIndex"]);
                Assert.Equal(Text(observation, "scalarBeforeF32Hex"), ScalarHex(scalar));
            }
            // Separate copy check after the single source-equivalent call, with the published overlapping mutation index.
            scalar.add_(0.5);
            foreach (var repeat in repeats)
            {
                var observation = repeat!["copyObservation"]!;
                Assert.Equal(Text(observation, "scalarAfterF32Hex"), ScalarHex(scalar));
                CompareTensor(image, observation["inputAfterMutation"]!);
                CompareTensor(result, observation["outputAfterMutation"]!);
                CompareTensor(result, repeat["output"]!);
            }
        }
    }

    private static void CompareSchema(NodeSchema schema, JsonObject actualInfo, JsonNode expected)
    {
        var required = expected["inputTypes"]!["required"]!.AsObject();
        Assert.Equal(required.Select(p => p.Key), schema.Inputs.Select(i => i.Name));
        Assert.All(schema.Inputs, input => Assert.True(input.Required && !input.Lazy));
        foreach (var input in schema.Inputs)
        {
            var sourceInput = required[input.Name]!.AsArray();
            Assert.Equal(input.Type, sourceInput[0]!.GetValue<string>());
            if (sourceInput.Count == 1) Assert.Null(input.Options);
            else { Assert.Equal(2, sourceInput.Count); Equal(sourceInput[1], input.Options ?? new JsonObject()); }
        }
        if (Text(expected, "kind") == "source-v3-object-info")
        {
            Equal(expected["objectInfo"], actualInfo);
            Equal(expected["inputTypes"], actualInfo["input"]);
        }
        else
        {
            // Legacy collection deliberately observed declarations, not a complete HTTP object_info response.
            Assert.Equal("source-legacy-declarations-not-complete-object-info", Text(expected, "kind"));
            Assert.Equal(Text(expected, "category"), schema.Category);
            Assert.Equal(expected["returnTypes"]!.AsArray().Select(n => n!.GetValue<string>()), schema.Outputs.Select(o => o.Type));
            Assert.False(schema.InputIsList || schema.OutputNode);
            Assert.All(schema.Outputs, output => Assert.False(output.IsList));
        }
    }

    private static void CompareTensor(Tensor actual, JsonNode expected)
    {
        Assert.Equal(ScalarType.Float32, actual.dtype);
        Assert.Equal(DeviceType.CPU, actual.device_type);
        Assert.False(actual.requires_grad);
        Assert.Equal(Longs(expected["shape"]!), actual.shape);
        Assert.Equal(Longs(expected["stride"]!), actual.stride());
        Assert.Equal(expected["storageOffset"]!.GetValue<long>(), actual.storage_offset());
        Assert.Equal(expected["aligned64"]!.GetValue<bool>(), Aligned64(actual));
        byte[] expectedBytes = Payload(expected);
        using var capture = NewDisposeScope();
        var logical = actual.contiguous(); // Original metadata checked above; the copy is only for capture.
        Assert.Equal(expectedBytes, logical.bytes.ToArray());
    }

    private static byte[] Payload(JsonNode record)
    {
        Assert.Equal("float32", Text(record, "dtype")); Assert.Equal("cpu", Text(record, "device"));
        Assert.False(record["requiresGrad"]!.GetValue<bool>());
        var shape = Longs(record["shape"]!);
        Assert.Equal(4, shape.Length); Assert.All(shape, n => Assert.True(n > 0));
        Assert.Contains(shape[3], new long[] { 3, 4 });
        long count = shape.Aggregate(1L, (a, b) => checked(a * b));
        Assert.InRange(count, 1, 4096);
        var raw = Convert.FromBase64String(Text(record, "littleEndianBase64"));
        Assert.Equal(count * sizeof(float), raw.LongLength);
        Assert.Equal(raw.Length, record["bytes"]!.GetValue<int>());
        Assert.Equal(Text(record, "sha256"), Hash(raw));
        for (int offset = 0; offset < raw.Length; offset += 4)
            Assert.True(float.IsFinite(BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(raw.AsSpan(offset, 4)))));
        return raw;
    }

    private static unsafe bool Aligned64(Tensor value)
    {
        using var scope = NewDisposeScope();
        // TorchSharp bytes requires contiguity. Selecting index zero on each dimension creates
        // a contiguous scalar view, with the SAME storage and original storage offset. No clone,
        // contiguous conversion or nonzero index is used, so this is the original first address.
        var first = value;
        while (first.dim() > 0) first = first.select(0, 0);
        Assert.Equal(value.storage_offset(), first.storage_offset());
        fixed (byte* address = first.bytes) return ((nuint)address & 63) == 0;
    }

    private static string ScalarHex(Tensor scalar)
    {
        Span<byte> raw = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(raw, BitConverter.SingleToInt32Bits(scalar.item<float>()));
        return Convert.ToHexStringLower(raw);
    }

    private static Corpus LoadCorpus()
    {
        var protocol = Resource("protocol", 40343, ProtocolSha);
        var reference = Resource("reference", 460188, ReferenceSha);
        Assert.Equal("image-primitives-cpu-f32-bits-v1", Text(protocol, "id"));
        Assert.Equal(Text(protocol, "id"), Text(reference, "protocolId"));
        Assert.Equal(ProtocolSha, Text(reference, "protocolSha256"));
        Assert.Equal("829a6b87f48ef7b1022da512524c5966e998a369", Text(reference, "collectorCommit"));
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", Text(reference, "backendCommit"));
        Assert.True(reference["executed"]!.GetValue<bool>());
        Assert.Equal("completed", Text(reference, "status"));
        Assert.Equal(28, reference["caseCount"]!.GetValue<int>());
        Assert.Equal(84, reference["sourceBodyInvocations"]!.GetValue<int>());
        Equal(protocol["sources"], reference["sourceFiles"]); Equal(protocol["helperFiles"], reference["helperOrigins"]);
        Assert.Equal(6, reference["sourceFiles"]!.AsArray().Count);
        Assert.Equal(59, reference["sourceFiles"]!.AsArray().Sum(s => s!["declarations"]!.AsArray().Count));
        Equal(protocol["profile"], reference["profile"]); Equal(protocol["environment"], reference["environment"]);
        Equal(reference["provenanceBefore"], reference["provenanceAfter"]);
        var before = reference["provenanceBefore"]!;
        Assert.Equal(ProtocolSha, Text(before["laboratoryFiles"]!["labs/image-primitives-source/protocol.json"]!, "sha256"));
        Assert.Equal("bbc62d6369496bc5fd3150826d3f7d28d3f123c75e341cfb78bb531aae219515",
            Text(before["laboratoryFiles"]!["labs/image-primitives-source/reference.py"]!, "sha256"));
        var lockInfo = before["dependencyLock"]!;
        var lockSpec = Assert.Single(protocol["dependencyLocks"]!.AsArray(), l => Text(l!, "target") == Text(reference, "target"))!;
        Equal(lockSpec["canonicalSha256"], lockInfo["canonicalSha256"]);
        Assert.Equal("replace_CRLF_with_LF_only_no_other_byte_changes", Text(lockInfo, "normalizationRule"));
        Assert.Equal("db3ba7b6998392497b69501a1d72c74499562503b5a41d98d72d9e12ae7794e3", Text(lockInfo, "rawSha256"));
        var native = reference["native"]!;
        Assert.Equal("3.12.10", Text(native, "python")); Assert.Equal("2.10.0+cpu", Text(native, "torch"));
        Assert.Equal("win-x64", Text(reference, "target"));
        Assert.Equal(1, native["threads"]!.GetValue<int>()); Assert.Equal(1, native["interopThreads"]!.GetValue<int>());
        Assert.False(native["gradEnabled"]!.GetValue<bool>());
        Assert.Equal("unset", Text(native, "requestedCapability"));
        Assert.Equal("after_first_image_body_once_per_process", Text(native["nativeLibraries"]!, "capturePoint"));
        Assert.Equal("available", Text(native["nativeLibraries"]!, "status"));
        Assert.NotEmpty(native["nativeLibraries"]!["libraries"]!.AsArray());
        // This records the Python source process, not the libraries loaded by the current .NET test process.
        foreach (var library in native["nativeLibraries"]!["libraries"]!.AsArray())
        {
            Assert.Equal("available", Text(library!, "status"));
            Assert.Matches("^[0-9a-f]{64}$", Text(library!, "sha256"));
        }
        var specifications = protocol["cases"]!.AsArray(); var cases = reference["cases"]!.AsArray();
        Assert.Equal(28, specifications.Count); Assert.Equal(28, cases.Count);
        Assert.Equal(specifications.Select(c => Text(c!, "id")), cases.Select(c => Text(c!, "id")));
        Assert.Equal(28, cases.Select(c => Text(c!, "id")).Distinct(StringComparer.Ordinal).Count());
        foreach (var pair in specifications.Zip(cases))
        {
            var item = pair.Second!; var spec = pair.First!;
            Equal(spec["node"], item["node"]);
            Assert.True(item["bitIdentical"]!.GetValue<bool>());
            var repeats = item["repeats"]!.AsArray(); var hashes = item["repeatSha256"]!.AsArray();
            Assert.Equal(3, repeats.Count); Assert.Equal(3, hashes.Count);
            for (int i = 0; i < 3; i++)
            {
                var repeat = repeats[i]!;
                Assert.Equal(1, repeat["sourceBodyInvocations"]!.GetValue<int>());
                Equal(spec["arguments"], repeat["arguments"]);
                Assert.Equal(hashes[i]!.GetValue<string>(), Hash(Encoding.UTF8.GetBytes(repeat.ToJsonString(Compact))));
                Equal(repeats[0], repeat);
                Payload(repeat["output"]!);
                Equal(repeat["inputBefore"], repeat["inputAfterBody"]);
                if (spec["image"] is null) continue;
                Payload(repeat["inputBefore"]!);
                var mutation = repeat["copyObservation"]!;
                Assert.Equal("observed_after_source_return", Text(mutation, "status"));
                Equal(spec["mutationIndex"], mutation["mutationIndex"]);
                Assert.True(mutation["verification"]!["inputChanged"]!.GetValue<bool>());
                Assert.True(mutation["verification"]!["outputBytesUnchanged"]!.GetValue<bool>());
                Assert.NotEqual(Text(repeat["inputBefore"]!, "sha256"), Text(mutation["inputAfterMutation"]!, "sha256"));
                Equal(repeat["output"], mutation["outputAfterMutation"]);
                Payload(mutation["inputAfterMutation"]!);
            }
        }
        return new(protocol, reference);
    }

    private static JsonObject Resource(string kind, int size, string sha)
    {
        using var stream = typeof(ImagePrimitiveSourceReferenceTests).Assembly.GetManifestResourceStream(
            "ComfySharp.Inference.Tests.Fixtures.image-primitives." + kind + ".json");
        Assert.NotNull(stream);
        using var bytes = new MemoryStream(); stream.CopyTo(bytes);
        byte[] raw = bytes.ToArray(); Assert.Equal(size, raw.Length); Assert.Equal(sha, Hash(raw));
        return JsonNode.Parse(raw)!.AsObject();
    }

    private static string Text(JsonNode node, string key) => node[key]!.GetValue<string>();
    private static long[] Longs(JsonNode node) => node.AsArray().Select(v => v!.GetValue<long>()).ToArray();
    private static string Hash(ReadOnlySpan<byte> raw) => Convert.ToHexStringLower(SHA256.HashData(raw));
    private static void Equal(JsonNode? expected, JsonNode? actual) => Assert.True(JsonNode.DeepEquals(expected, actual),
        $"Expected {expected?.ToJsonString()} but received {actual?.ToJsonString()}");
}
