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
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>One registered node call per case against three independently collected source records.
/// The prospective bilinear tolerance applies only to output values; inputs and metadata remain exact.</summary>
[Collection("Classical VAE")]
public sealed class ImageBatchSourceReferenceTests(ITestOutputHelper output)
{
    private const string ProtocolSha = "6d3f1e09482713beb02497063dd2e3721032fd8a24ff486e5deed9eb23461962";
    private const string ReferenceSha = "879044ad5c86b0a31c96129bd680736b63e55b3142819b01a120db8b7863d2e0";
    private static readonly JsonSerializerOptions Compact = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
    private static readonly Lazy<(JsonObject Protocol, JsonObject Reference)> Source = new(LoadCorpus);

    [Theory]
    [InlineData("same-rgb-batches")]
    [InlineData("same-rgba-views")]
    [InlineData("pad-first-rgb")]
    [InlineData("pad-second-rgb")]
    [InlineData("upsample-half-pixel")]
    [InlineData("downsample-no-antialias")]
    [InlineData("crop-wide")]
    [InlineData("crop-tall")]
    [InlineData("crop-half-x-even-zero")]
    [InlineData("crop-one-half-x-even-two")]
    [InlineData("crop-half-y-even-zero")]
    [InlineData("crop-one-half-y-even-two")]
    [InlineData("asymmetric-tall-crop-views")]
    [InlineData("asymmetric-wide-crop-views")]
    [InlineData("rgba-alpha-downsample")]
    [InlineData("pad-second-before-resize")]
    [InlineData("pad-first-before-resize")]
    [InlineData("resize-batches-offsets")]
    [InlineData("empty-center-crop")]
    public async Task RegisteredBatchMatchesFrozenSource(string id)
    {
        var corpus = Source.Value;
        var spec = Assert.Single(corpus.Protocol["cases"]!.AsArray(), c => Text(c!, "id") == id)!;
        var expected = Assert.Single(corpus.Reference["cases"]!.AsArray(), c => Text(c!, "id") == id)!;
        var repeats = expected["repeats"]!.AsArray();
        bool bilinear = Text(spec, "classification") == "bilinear";
        bool sourceError = Text(spec, "classification") == "source-error";
        var registry = TensorNodes.CreateRegistry();
        Assert.True(registry.TryGet("ImageBatch", out var node));
        var info = registry.ToObjectInfo()["ImageBatch"]!;
        foreach (var repeat in repeats)
        {
            var schema = repeat!["schema"]!;
            Assert.Equal("source-legacy-declarations-not-complete-object-info", Text(schema, "kind"));
            Assert.Equal("batch", Text(schema, "function"));
            Assert.Equal(schema["inputTypes"]!["required"]!.AsObject().Select(p => p.Key), node.Schema.Inputs.Select(i => i.Name));
            Assert.All(node.Schema.Inputs, input =>
            {
                Assert.True(input.Required); Assert.False(input.Lazy); Assert.Null(input.Options);
                Assert.Equal(input.Type, schema["inputTypes"]!["required"]![input.Name]![0]!.GetValue<string>());
            });
            Equal(schema["returnTypes"], info["output"]); Equal(schema["searchAliases"], info["search_aliases"]);
            Assert.Equal(Text(schema, "category"), node.Schema.Category);
            Assert.True(schema["deprecated"]!.GetValue<bool>() && node.Schema.Deprecated && info["deprecated"]!.GetValue<bool>());
            Assert.False(node.Schema.OutputNode || node.Schema.InputIsList);
        }

        NativeRuntimeBootstrap.Initialize();
        Assert.True(BitConverter.IsLittleEndian);
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        using var owner = new RuntimeNodeContext();
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        var inputs = new Dictionary<string, RuntimeValue>(StringComparer.Ordinal);
        foreach (string key in new[] { "image1", "image2" })
        {
            var descriptor = spec[key]!;
            byte[] raw = Convert.FromBase64String(Text(descriptor, "storageLittleEndianBase64"));
            Assert.Equal(Text(descriptor, "storageSha256"), Hash(raw));
            Assert.Equal(descriptor["storageElements"]!.GetValue<int>() * 4, raw.Length);
            var storage = empty(new long[] { raw.Length / 4 }, dtype: ScalarType.Float32, device: CPU);
            raw.AsSpan().CopyTo(storage.bytes);
            Assert.True(Aligned64(storage)); Assert.Equal(raw, storage.bytes.ToArray());
            var tensor = storage.as_strided(Longs(descriptor["shape"]!), Longs(descriptor["stride"]!), descriptor["storageOffset"]!.GetValue<long>());
            tensors.Add(key, tensor); inputs.Add(key, owner.Own(tensor)); tensor.DetachFromDisposeScope();
            foreach (var repeat in repeats) CompareTensor(tensor, repeat!["inputBefore"]![key]!);
        }

        RuntimeValue? retained = null;
        using (var invocation = new RuntimeNodeContext())
        {
            if (sourceError)
            {
                long before = Tensor.TotalCount;
                // A real native interpolation failure, not a failure in input validation or the observer.
                var error = await Record.ExceptionAsync(async () =>
                {
                    var pending = node.ExecuteAsync(invocation, inputs, default);
                    Assert.True(pending.IsCompletedSuccessfully);
                    await pending;
                });
                Assert.NotNull(error);
                Assert.Contains("interpolate", error.StackTrace ?? "", StringComparison.Ordinal);
                Assert.Equal(before, Tensor.TotalCount);
                foreach (var repeat in repeats)
                {
                    Assert.Equal("raised", Text(repeat!, "status"));
                    foreach (var pair in tensors) CompareTensor(pair.Value, repeat!["inputAfterBody"]![pair.Key]!);
                }
                return;
            }
            var pending = node.ExecuteAsync(invocation, inputs, default);
            Assert.True(pending.IsCompletedSuccessfully); // Native scopes must not cross a thread suspension.
            var returned = await pending;
            Assert.Null(returned.Ui);
            retained = Assert.Single(returned.Result).Retain();
        }
        using (retained)
        {
            var result = retained.GetNative<Tensor>();
            (double Absolute, double Ratio) metric = default;
            foreach (var repeat in repeats)
            {
                metric = CompareTensor(result, repeat!["output"]!, bilinear);
                foreach (var pair in tensors) CompareTensor(pair.Value, repeat["inputAfterBody"]![pair.Key]!);
            }
            byte[] unchangedOutput = LogicalBytes(result);
            int step = 0;
            foreach (string key in new[] { "image1", "image2" })
            {
                var scalar = tensors[key];
                foreach (long index in Longs(spec["mutationIndices"]![key]!)) scalar = scalar.select(0, index);
                foreach (var repeat in repeats)
                    Assert.Equal(Text(repeat!["copyObservation"]!["sequence"]![step]!, "scalarBeforeF32Hex"), ScalarHex(scalar));
                scalar.add_(0.5);
                foreach (var repeat in repeats)
                {
                    var mutation = repeat!["copyObservation"]!["sequence"]![step]!;
                    Assert.Equal(key, Text(mutation, "input"));
                    Assert.Equal(Text(mutation, "scalarAfterF32Hex"), ScalarHex(scalar));
                    foreach (var pair in tensors) CompareTensor(pair.Value, mutation["inputsAfterMutation"]![pair.Key]!);
                    CompareTensor(result, mutation["outputAfterMutation"]!, bilinear);
                }
                Assert.Equal(unchangedOutput, LogicalBytes(result));
                step++;
            }
            // The published corner mutations can lie outside the crop. This separate whole-input
            // mutation checks managed output ownership; it is not another source-equivalent body call.
            foreach (var tensor in tensors.Values) tensor.fill_(99);
            Assert.Equal(unchangedOutput, LogicalBytes(result));
            output.WriteLine("case={0}; maxAbsolute={1:R}; maxBoundRatio={2:R}; profile={3}",
                id, metric.Absolute, metric.Ratio, Text(spec, "classification"));
        }
    }

    private static (double Absolute, double Ratio) CompareTensor(Tensor actual, JsonNode expected, bool bilinear = false)
    {
        Assert.Equal(ScalarType.Float32, actual.dtype); Assert.Equal(DeviceType.CPU, actual.device_type);
        Assert.False(actual.requires_grad);
        Assert.Equal(Longs(expected["shape"]!), actual.shape); Assert.Equal(Longs(expected["stride"]!), actual.stride());
        Assert.Equal(expected["storageOffset"]!.GetValue<long>(), actual.storage_offset());
        Assert.Equal(expected["aligned64"]!.GetValue<bool>(), Aligned64(actual));
        byte[] source = Payload(expected), received = LogicalBytes(actual);
        Assert.Equal(source.Length, received.Length);
        if (!bilinear) { Assert.Equal(source, received); return default; }
        double maximum = 0, ratio = 0;
        for (int i = 0; i < source.Length; i += 4)
        {
            float reference = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(source.AsSpan(i, 4)));
            float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(received.AsSpan(i, 4)));
            Assert.True(float.IsFinite(value));
            double difference = Math.Abs((double)value - reference), bound = 1e-6 + 1e-6 * Math.Abs(reference);
            Assert.True(difference <= bound, $"Float32[{i / 4}]: actual={value:R}, source={reference:R}, abs={difference:R}, bound={bound:R}");
            maximum = Math.Max(maximum, difference); ratio = Math.Max(ratio, difference / bound);
        }
        return (maximum, ratio);
    }

    private static byte[] LogicalBytes(Tensor value)
    {
        using var scope = NewDisposeScope();
        return value.contiguous().bytes.ToArray();
    }

    private static byte[] Payload(JsonNode record)
    {
        Assert.Equal("float32", Text(record, "dtype")); Assert.Equal("cpu", Text(record, "device"));
        Assert.False(record["requiresGrad"]!.GetValue<bool>());
        long[] shape = Longs(record["shape"]!);
        Assert.Equal(4, shape.Length); Assert.All(shape, n => Assert.True(n > 0));
        Assert.Contains(shape[3], new long[] { 3, 4 });
        long count = shape.Aggregate(1L, (a, b) => checked(a * b)); Assert.InRange(count, 1, 4096);
        byte[] raw = Convert.FromBase64String(Text(record, "littleEndianBase64"));
        Assert.Equal(count * 4, raw.LongLength); Assert.Equal(raw.Length, record["bytes"]!.GetValue<int>());
        Assert.Equal(Text(record, "sha256"), Hash(raw));
        for (int i = 0; i < raw.Length; i += 4)
            Assert.True(float.IsFinite(BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(raw.AsSpan(i, 4)))));
        return raw;
    }

    private static unsafe bool Aligned64(Tensor value)
    {
        using var scope = NewDisposeScope();
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

    private static (JsonObject, JsonObject) LoadCorpus()
    {
        var protocol = Resource("protocol", 41708, ProtocolSha);
        var reference = Resource("reference", 693068, ReferenceSha);
        Assert.Equal("image-batch-cpu-f32-v1", Text(protocol, "id"));
        Assert.Equal(Text(protocol, "id"), Text(reference, "protocolId"));
        Assert.Equal(ProtocolSha, Text(reference, "protocolSha256"));
        Assert.Equal("5fd5f1e5af171d2dc43d7d37366d7f4f50e976de", Text(reference, "collectorCommit"));
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", Text(reference, "backendCommit"));
        Assert.True(reference["executed"]!.GetValue<bool>()); Assert.Equal("completed", Text(reference, "status"));
        Assert.Equal(19, reference["caseCount"]!.GetValue<int>()); Assert.Equal(57, reference["sourceBodyInvocations"]!.GetValue<int>());
        Equal(protocol["profile"], reference["profile"]); Equal(protocol["environment"], reference["environment"]);
        Equal(protocol["helperFiles"], reference["helperOrigins"]);
        Equal(reference["provenanceBefore"], reference["provenanceAfter"]);
        var before = reference["provenanceBefore"]!;
        Equal(protocol["sources"], before["sourceFiles"]);
        Assert.Equal(2, before["sourceFiles"]!.AsArray().Count);
        Assert.Equal(2, before["sourceFiles"]!.AsArray().Sum(f => f!["declarations"]!.AsArray().Count));
        Assert.Equal("9cae9158b8da5a54860e3fbdc3f50a5b886c2d6f3bfd3720cd3fe5b4a797c333",
            Text(before["laboratoryFiles"]!["labs/image-batch-source/reference.py"]!, "sha256"));
        Assert.Equal(ProtocolSha, Text(before["laboratoryFiles"]!["labs/image-batch-source/protocol.json"]!, "sha256"));
        var lockSpec = Assert.Single(protocol["dependencyLocks"]!.AsArray(), l => Text(l!, "target") == "win-x64")!;
        Equal(lockSpec["canonicalSha256"], before["dependencyLock"]!["canonicalSha256"]);
        Assert.Equal("db3ba7b6998392497b69501a1d72c74499562503b5a41d98d72d9e12ae7794e3", Text(before["dependencyLock"]!, "rawSha256"));
        var native = reference["native"]!;
        Assert.Equal("win-x64", Text(reference, "target")); Assert.Equal("3.12.10", Text(native, "python"));
        Assert.Equal("2.10.0+cpu", Text(native, "torch")); Assert.Equal("unset", Text(native, "requestedCapability"));
        Assert.Equal(1, native["threads"]!.GetValue<int>()); Assert.Equal(1, native["interopThreads"]!.GetValue<int>());
        Assert.False(native["gradEnabled"]!.GetValue<bool>());
        Assert.Equal("after_first_image_batch_body_once_per_process", Text(native["nativeLibraries"]!, "capturePoint"));
        Assert.NotEmpty(native["nativeLibraries"]!["libraries"]!.AsArray());
        var specs = protocol["cases"]!.AsArray(); var cases = reference["cases"]!.AsArray();
        Assert.Equal(19, specs.Count); Assert.Equal(19, cases.Count);
        Assert.Equal(specs.Select(c => Text(c!, "id")), cases.Select(c => Text(c!, "id")));
        Assert.Equal(19, cases.Select(c => Text(c!, "id")).Distinct(StringComparer.Ordinal).Count());
        foreach (var pair in specs.Zip(cases))
        {
            var spec = pair.First!; var item = pair.Second!;
            Equal(spec["classification"], item["classification"]); Assert.True(item["bitIdentical"]!.GetValue<bool>());
            bool error = Text(spec, "classification") == "source-error";
            var repeats = item["repeats"]!.AsArray(); var hashes = item["neutralRepeatSha256"]!.AsArray();
            Assert.Equal(3, repeats.Count); Assert.Equal(3, hashes.Count);
            for (int i = 0; i < 3; i++)
            {
                var repeat = repeats[i]!;
                var neutral = repeat.DeepClone().AsObject(); neutral.Remove("observation");
                Assert.Equal(hashes[i]!.GetValue<string>(), Hash(Encoding.UTF8.GetBytes(neutral.ToJsonString(Compact))));
                Equal(repeats[0], neutral); Assert.Equal(1, repeat["sourceBodyInvocations"]!.GetValue<int>());
                Equal(repeat["inputBefore"], repeat["inputAfterBody"]);
                foreach (string key in new[] { "image1", "image2" }) Payload(repeat["inputBefore"]![key]!);
                Assert.Equal(error ? "raised" : "returned", Text(repeat, "status"));
                if (error)
                {
                    Assert.Equal("body", Text(repeat, "stage")); Assert.Equal("RuntimeError", Text(repeat, "errorType"));
                    Assert.Null(repeat["output"]);
                    Assert.Equal("not_applicable_source_error_no_output", Text(repeat["copyObservation"]!, "status"));
                }
                else
                {
                    Payload(repeat["output"]!);
                    Assert.Equal("separate_post_body_mutations", Text(repeat["copyObservation"]!, "status"));
                    var mutations = repeat["copyObservation"]!["sequence"]!.AsArray(); Assert.Equal(2, mutations.Count);
                    for (int step = 0; step < 2; step++)
                    {
                        var mutation = mutations[step]!; string key = step == 0 ? "image1" : "image2";
                        Assert.Equal(key, Text(mutation, "input")); Equal(spec["mutationIndices"]![key], mutation["index"]);
                        Equal(repeat["output"], mutation["outputAfterMutation"]);
                        Assert.True(mutation["verification"]!["inputChanged"]!.GetValue<bool>());
                        Assert.True(mutation["verification"]!["outputUnchanged"]!.GetValue<bool>());
                        foreach (string name in new[] { "image1", "image2" }) Payload(mutation["inputsAfterMutation"]![name]!);
                    }
                }
            }
            var observation = repeats[1]!["observation"]!.AsArray();
            Assert.Equal(Text(spec, "classification") == "exact-no-resize" ? 0 : 1, observation.Count);
            if (observation.Count > 0)
            {
                var observed = observation[0]!;
                Assert.Equal("actual_common_upscale_return", Text(observed, "capture"));
                Assert.Equal("center", Text(observed, "crop")); Assert.Equal("bilinear", Text(observed, "method"));
                Assert.Equal(error, observed["exceptionalReturn"]!.GetValue<bool>());
                var crop = Longs(observed["cropped"]!["shape"]!);
                Assert.Equal(error, crop[^2] == 0 || crop[^1] == 0);
            }
        }
        return (protocol, reference);
    }

    private static JsonObject Resource(string kind, int size, string sha)
    {
        using var stream = typeof(ImageBatchSourceReferenceTests).Assembly.GetManifestResourceStream(
            "ComfySharp.Inference.Tests.Fixtures.image-batch." + kind + ".json");
        Assert.NotNull(stream);
        using var bytes = new MemoryStream(); stream.CopyTo(bytes);
        byte[] raw = bytes.ToArray(); Assert.Equal(size, raw.Length); Assert.Equal(sha, Hash(raw));
        return JsonNode.Parse(raw)!.AsObject();
    }

    private static string Text(JsonNode node, string key) => node[key]!.GetValue<string>();
    private static long[] Longs(JsonNode node) => node.AsArray().Select(v => v!.GetValue<long>()).ToArray();
    private static string Hash(ReadOnlySpan<byte> raw) => Convert.ToHexStringLower(SHA256.HashData(raw));
    private static void Equal(JsonNode? expected, JsonNode? actual) => Assert.True(JsonNode.DeepEquals(expected, actual));
}
