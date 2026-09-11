using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.RuntimeProbe;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Consumes the independently audited source collection; never computes or replaces an oracle.
/// Sixty-four distinct source captures and thirty-six repeated final comparisons across four cases.</summary>
[Collection("Classical VAE")]
public sealed class Sd15PipelineReferenceTests : IDisposable
{
    private const string Collector = "6ad8207ffe46d3d1e57780b9fc4c7288b686638d";
    private const string Backend = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a";
    private const string Profile = "sd15-pipeline-native210-cpu-f32-v1";
    private static readonly string[] CaseIds = ["empty-one-step", "weighted-three-step", "two-chunks-separate", "maximum-start"];
    private static readonly string[] BoundaryNames = ["positive.hidden", "positive.pooled", "negative.hidden", "negative.pooled",
        "initialDiffusionLatent", "finalDiffusionLatent", "rawVaeLatent", "image"];
    private readonly int previousThreads;

    private sealed record Snapshot(long[] Shape, long[] Stride, long StorageOffset, string Dtype, string Sha256,
        float[] Values, bool RequiresGrad);
    private sealed record Parameter(string Name, long[] Shape, long[] Stride, string Sha256, bool Aligned64, bool RequiresGrad);
    private sealed record Step(long Index, Snapshot X, Snapshot Denoised, Snapshot Sigma, bool GradEnabled)
    {
        public Snapshot SigmaHat => Sigma; // The same observed scalar under the fixed no-churn contract.
    }

    public Sd15PipelineReferenceTests()
    {
        NativeRuntimeBootstrap.Initialize();
        previousThreads = get_num_threads();
        set_num_threads(1);
        SdReferenceRuntime.Verify();
    }

    [Theory]
    [InlineData("empty-one-step")]
    [InlineData("weighted-three-step")]
    [InlineData("two-chunks-separate")]
    [InlineData("maximum-start")]
    public void PipelineMatchesItsSamePlatformFrozenSource(string identifier)
    {
        using var corpus = ReadCorpus();
        using var protocol = JsonDocument.Parse(ClipReferenceTests.Resource("sd15-pipeline.protocol.json"));
        var root = corpus.RootElement;
        var definition = Assert.Single(protocol.RootElement.GetProperty("cases").EnumerateArray(), c => Text(c, "id") == identifier);
        var reference = Assert.Single(root.GetProperty("cases").EnumerateArray(), c => Text(c, "id") == identifier);
        var selected = Sd15PipelineCases.Get(identifier);
        var conditioning = selected.TokenizeAndVerify();
        CheckOptions(definition, selected);
        Assert.Equal(new ClipTextConfig(16, 32, 2, 4, ClipActivation.QuickGelu), Sd15PipelineDiagnostic.ClipConfig);
        Assert.Equal(new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false), Sd15PipelineExecution.UnetConfig);
        Assert.Equal(new ClassicalVaeConfig(32), Sd15PipelineExecution.VaeConfig);

        var borrowed = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        T Transfer<T>(string component, IReadOnlyDictionary<string, Tensor> values,
            Func<IReadOnlyDictionary<string, Tensor>, T> make) where T : IDisposable
        {
            foreach (var (name, value) in values) borrowed.Add(component + "/" + name, value);
            return make(values);
        }
        var clipPlan = SdSyntheticWeightBuilder.Describe(ClipWeightSchema.Describe(Sd15PipelineDiagnostic.ClipConfig));
        var unetPlan = SdSyntheticWeightBuilder.DescribeUnet(Sd15PipelineExecution.UnetConfig);
        var vaePlan = SdSyntheticWeightBuilder.Describe(ClassicalVaeWeightSchema.Describe(Sd15PipelineExecution.VaeConfig));
        Assert.Equal(971, clipPlan.TensorCount + unetPlan.TensorCount + vaePlan.TensorCount);
        Assert.Equal(58_137_836L, clipPlan.ResidentBytes + unetPlan.ResidentBytes + vaePlan.ResidentBytes);
        using var clipBank = SdSyntheticWeightBuilder.Create(clipPlan, new(clipPlan.ResidentBytes),
            v => Transfer("clip", v, values => ClipWeightSet.FromOwnedTensors(Sd15PipelineDiagnostic.ClipConfig, values)));
        using var unetBank = SdSyntheticWeightBuilder.Create(unetPlan, new(unetPlan.ResidentBytes),
            v => Transfer("unet", v, values => UnetWeightSet.FromOwnedTensors(Sd15PipelineExecution.UnetConfig, values)));
        using var vaeBank = SdSyntheticWeightBuilder.Create(vaePlan, new(vaePlan.ResidentBytes),
            v => Transfer("vae", v, values => ClassicalVaeWeightSet.FromOwnedTensors(Sd15PipelineExecution.VaeConfig, values)));
        var parametersBefore = CaptureParameters(borrowed);
        CheckParameters(parametersBefore, root.GetProperty("parameters"));
        CheckParameters(parametersBefore, reference.GetProperty("parameterHashesBefore"));

        var captures = new Dictionary<string, Snapshot>(StringComparer.Ordinal);
        var steps = new List<Step>();
        var finals = new List<Dictionary<string, Snapshot>>(3);
        var restoredGrad = new List<bool>(3);
        var captureGrad = new List<bool>(8);
        Dictionary<string, Snapshot> inputsBefore;
        Dictionary<string, Snapshot> inputsAfter;
        using (var clipGraph = new ClipTextEncoder(clipBank.Weights))
        using (var clip = new ComfyClipEncoder(clipGraph, ComfySharp.Tokenization.ClipProfile.Sd1L))
        using (var unet = new SdUnet(unetBank.Weights))
        using (var vaeGraph = new ClassicalVae(vaeBank.Weights))
        using (var vae = new ComfyImageVae(vaeGraph))
        using (var callerGrad = set_grad_enabled(true))
        using (var noise = Input(selected.NoiseValues(), [1, 4, 4, 5]))
        using (var sigmas = Input(selected.SigmaValues(), [selected.Steps + 1]))
        {
            inputsBefore = new() { ["noise"] = Capture(noise), ["sigmas"] = Capture(sigmas) };
            Assert.True(SdSyntheticWeightBuilder.IsAligned(noise));
            Assert.True(SdSyntheticWeightBuilder.IsAligned(sigmas));
            Assert.Equal(selected.NoiseSha256, inputsBefore["noise"].Sha256);
            Assert.Equal(selected.SigmaSha256, inputsBefore["sigmas"].Sha256);
            CheckInputs(inputsBefore, reference.GetProperty("nativeInputBefore"));
            for (int repeat = 0; repeat < 3; repeat++)
            {
                // Only managed snapshots survive callbacks. The same observation schedule applies
                // whether or not file tracing is requested. No wrapper is retained or mutated.
                using var actual = Sd15PipelineExecution.Run(clip, unet, vae, conditioning, noise, sigmas,
                    capture: repeat == 1 ? (boundary, value) =>
                    {
                        captures.Add(BoundaryName(boundary), Capture(value));
                        captureGrad.Add(is_grad_enabled());
                    } : null,
                    configureSampler: repeat == 1 ? sampler => sampler.DiagnosticObserver = (index, x, denoised, sigma) =>
                        steps.Add(new(index, Capture(x), Capture(denoised), Capture(sigma), is_grad_enabled())) : null);
                restoredGrad.Add(is_grad_enabled());
                finals.Add(new() { ["finalDiffusionLatent"] = Capture(actual.DiffusionLatent),
                    ["rawVaeLatent"] = Capture(actual.RawVaeLatent), ["image"] = Capture(actual.Image) });
            }
            inputsAfter = new() { ["noise"] = Capture(noise), ["sigmas"] = Capture(sigmas) };
        }
        var parametersAfter = CaptureParameters(borrowed);
        WriteTrace(identifier, root, definition, reference, inputsBefore, inputsAfter, parametersBefore, parametersAfter,
            captures, steps, finals, restoredGrad, captureGrad);

        // All captures are written above, before numeric/repeat/neutrality assertions can fail.
        Assert.All(restoredGrad, value => Assert.True(value));
        Assert.All(captureGrad, value => Assert.False(value));
        CheckInputs(inputsAfter, reference.GetProperty("nativeInputAfter"));
        foreach (string name in inputsBefore.Keys) EqualSnapshot(inputsBefore[name], inputsAfter[name]);
        CheckParameters(parametersAfter, root.GetProperty("parameters"));
        CheckParameters(parametersAfter, reference.GetProperty("parameterHashesAfter"));
        Assert.Equal(parametersBefore.Select(p => p.Sha256), parametersAfter.Select(p => p.Sha256));
        Assert.Equal(BoundaryNames.Order(StringComparer.Ordinal), captures.Keys.Order(StringComparer.Ordinal));
        var expectedCaptures = reference.GetProperty("captures");
        Assert.Equal(BoundaryNames.Order(StringComparer.Ordinal), expectedCaptures.EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal));
        foreach (string name in BoundaryNames) Compare(captures[name], expectedCaptures.GetProperty(name), identifier + "/" + name);
        Assert.Equal(selected.Steps, steps.Count);
        var expectedSteps = reference.GetProperty("steps").EnumerateArray().ToArray();
        Assert.Equal(steps.Count, expectedSteps.Length);
        for (int index = 0; index < steps.Count; index++)
        {
            var actual = steps[index];
            var expected = expectedSteps[index];
            Assert.Equal((long)index, actual.Index);
            Assert.Equal(index, expected.GetProperty("index").GetInt32());
            Assert.False(actual.GradEnabled);
            Compare(actual.X, expected.GetProperty("x"), $"{identifier}/step-{index}/x");
            Compare(actual.Denoised, expected.GetProperty("denoised"), $"{identifier}/step-{index}/denoised");
            Assert.Empty(actual.Sigma.Shape);
            Assert.False(actual.Sigma.RequiresGrad);
            Assert.Equal(BitConverter.SingleToInt32Bits(selected.SigmaValues()[index]), BitConverter.SingleToInt32Bits(Assert.Single(actual.Sigma.Values)));
            // No churn: the actual observed scalar is also sigmaHat. No extra operation is reconstructed.
            foreach (string key in new[] { "sigma", "sigmaHat" })
            {
                Assert.Empty(Shape(expected.GetProperty(key)));
                var scalar = ReadValues(expected.GetProperty(key));
                Assert.Equal(Hash(MemoryMarshal.AsBytes(scalar.AsSpan())), actual.Sigma.Sha256);
            }
        }
        Assert.Equal(3, finals.Count);
        var sourceRepeats = reference.GetProperty("repeatHashes").EnumerateArray().ToArray();
        Assert.Equal(3, sourceRepeats.Length);
        foreach (var (name, first) in finals[0])
        {
            Assert.All(finals, run => Assert.Equal(first.Sha256, run[name].Sha256));
            Assert.All(sourceRepeats, run => Assert.Equal(Text(sourceRepeats[0], name), Text(run, name)));
            Assert.Equal(Text(expectedCaptures.GetProperty(name), "sha256"), Text(sourceRepeats[0], name));
            for (int repeat = 0; repeat < 3; repeat++) Compare(finals[repeat][name], expectedCaptures.GetProperty(name), $"{identifier}/repeat-{repeat}/{name}");
        }
    }

    private static JsonDocument ReadCorpus()
    {
        Assert.True(BitConverter.IsLittleEndian);
        string target = SdSamplingReferenceTests.Target;
        var pins = target switch
        {
            "win-x64" => ("41f3b8324ef686ea8d01499083405291cd83bcac6a2b4856b4cc18b159817860", "976ce52fc5979480d8d069190a57e5f46cb32f14ac37883c7444b19a373563b2", 3_983_944),
            "linux-x64" => ("61d83b606620ee68ebcf928564d60fa75359d4d8c6cace2fd31ad7408f97be36", "8c06321dac0cbb0fd771770f7d20465fae188b3316c04b10dc1a34888a1d8969", 3_984_788),
            "osx-arm64" => ("7592859e31d9312eaee34043142fd83f8987bfcb0cb8f64fd5238018171c3326", "ce50568eef84cfa3bbcd5430ed667ab6f0929aa9aa8e3641018b95650fb57899", 3_983_715),
            _ => throw new PlatformNotSupportedException()
        };
        byte[] manifestBytes = ClipReferenceTests.Resource($"sd15-pipeline.{target}.manifest.json");
        Assert.Equal(pins.Item1, Hash(manifestBytes));
        using var manifest = JsonDocument.Parse(manifestBytes);
        var metadata = manifest.RootElement;
        Assert.Equal("completed", Text(metadata, "status"));
        Assert.Equal(4, metadata.GetProperty("cases").GetInt32());
        Assert.Equal(64, metadata.GetProperty("captureRecords").GetInt32());
        Assert.Equal(971, metadata.GetProperty("selectedParameters").GetInt32());
        Assert.Equal(3e-5, metadata.GetProperty("comparison").GetProperty("absoluteTolerance").GetDouble());
        Assert.Equal(3e-5, metadata.GetProperty("comparison").GetProperty("relativeTolerance").GetDouble());
        Assert.True(JsonElement.DeepEquals(metadata.GetProperty("inputIntegrityBefore"), metadata.GetProperty("inputIntegrityAfter")));
        byte[] bytes = ClipReferenceTests.Resource($"sd15-pipeline.{target}.json");
        Assert.Equal(pins.Item3, bytes.Length);
        Assert.Equal(pins.Item2, Hash(bytes));
        var component = metadata.GetProperty("component");
        Assert.Equal("pipeline.json", Text(component, "file"));
        Assert.Equal(bytes.Length, component.GetProperty("bytes").GetInt32());
        Assert.Equal(pins.Item2, Text(component, "sha256"));
        var document = JsonDocument.Parse(bytes);
        try
        {
            var root = document.RootElement;
            foreach (var record in new[] { metadata, root })
            {
                Assert.Equal(Profile, Text(record, "profile"));
                Assert.Equal(Backend, Text(record, "backendCommit"));
                Assert.Equal(Collector, Text(record, "collectorCommit"));
                Assert.Equal(Sd15PipelineCases.ProtocolSha256, Text(record, "protocolSha256"));
                Assert.Equal(target, Text(record, "target"));
            }
            var runtime = root.GetProperty("runtime");
            Assert.Equal("3.12.10", Text(runtime, "python"));
            Assert.Equal(target == "osx-arm64" ? "2.10.0" : "2.10.0+cpu", Text(runtime, "torch"));
            Assert.Equal("449b1768410104d3ed79d3bcfe4ba1d65c7f22c0", Text(runtime, "torchGit"));
            Assert.Equal(1, runtime.GetProperty("threads").GetInt32());
            Assert.Equal(1, runtime.GetProperty("interopThreads").GetInt32());
            Assert.True(root.GetProperty("syntheticWeights").GetBoolean());
            Assert.False(root.GetProperty("pretrainedWeightsUsed").GetBoolean());
            Assert.Equal(new[] { "off", "on", "off" }, root.GetProperty("observerSequence").EnumerateArray().Select(v => v.GetString()));
            Assert.Equal(CaseIds.Order(StringComparer.Ordinal), root.GetProperty("cases").EnumerateArray().Select(c => Text(c, "id")).Order(StringComparer.Ordinal));
            var protocolBytes = ClipReferenceTests.Resource("sd15-pipeline.protocol.json");
            Assert.Equal(Sd15PipelineCases.ProtocolSha256, Hash(protocolBytes));
            using var protocol = JsonDocument.Parse(protocolBytes);
            Assert.True(JsonElement.DeepEquals(root.GetProperty("inputRecords"), protocol.RootElement.GetProperty("inputRecords")));
            Assert.True(JsonElement.DeepEquals(root.GetProperty("texts"), protocol.RootElement.GetProperty("texts")));
            Assert.Equal(20, root.GetProperty("inputRecords").GetArrayLength());
            // The hash-validated case loader verifies exact source configurations and option bits.
            Assert.Equal(4, Sd15PipelineCases.All.Count);
            return document;
        }
        catch { document.Dispose(); throw; }
    }

    private static void CheckOptions(JsonElement definition, Sd15PipelineCase actual)
    {
        Assert.Equal(actual.Id, Text(definition, "id"));
        Assert.Equal("epsilon", Text(definition, "predictionKind"));
        Assert.Equal("separate", Text(definition, "policy"));
        Assert.False(definition.GetProperty("negativeContextIsNull").GetBoolean());
        Assert.False(definition.GetProperty("disableScaleOneOptimization").GetBoolean());
        Assert.Equal(actual.GuidanceScale, definition.GetProperty("scale").GetDouble());
        Assert.Equal(actual.MaximumDenoise, definition.GetProperty("maximumDenoise").GetBoolean());
        Assert.Equal(actual.Steps, definition.GetProperty("steps").GetInt32());
    }

    private static Tensor Input(float[] values, long[] shape)
    {
        var tensor = empty(shape, dtype: ScalarType.Float32, device: CPU);
        try { values.AsSpan().CopyTo(MemoryMarshal.Cast<byte, float>(tensor.bytes)); return tensor; }
        catch { tensor.Dispose(); throw; }
    }

    private static Snapshot Capture(Tensor tensor)
    {
        long[] shape = tensor.shape, stride = tensor.stride();
        long offset = tensor.storage_offset();
        // Only serialization is contiguous. Never substitute this snapshot into the graph.
        using var copy = tensor.contiguous();
        var values = copy.data<float>().ToArray();
        return new(shape, stride, offset, tensor.dtype == ScalarType.Float32 ? "float32" : tensor.dtype.ToString(),
            Hash(MemoryMarshal.AsBytes(values.AsSpan())), values, tensor.requires_grad);
    }

    private static Parameter[] CaptureParameters(Dictionary<string, Tensor> tensors) => tensors.OrderBy(p => p.Key, StringComparer.Ordinal)
        .Select(p => new Parameter(p.Key, p.Value.shape, p.Value.stride(), Hash(p.Value.bytes),
            SdSyntheticWeightBuilder.IsAligned(p.Value), p.Value.requires_grad)).ToArray();

    private static void CheckParameters(Parameter[] actual, JsonElement source)
    {
        var expected = source.EnumerateObject().SelectMany(component => component.Value.EnumerateArray()
            .Select(p => (Name: component.Name + "/" + Text(p, "name"), Value: p))).OrderBy(p => p.Name, StringComparer.Ordinal).ToArray();
        Assert.Equal(971, actual.Length);
        Assert.Equal(971, expected.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.Equal(expected[i].Name, actual[i].Name);
            Assert.Equal(Shape(expected[i].Value), actual[i].Shape);
            Assert.Equal(Longs(expected[i].Value.GetProperty("stride")), actual[i].Stride);
            Assert.Equal(Text(expected[i].Value, "sha256"), actual[i].Sha256);
            Assert.True(actual[i].Aligned64);
            Assert.True(expected[i].Value.GetProperty("aligned64").GetBoolean());
            Assert.False(actual[i].RequiresGrad);
        }
    }

    private static void CheckInputs(Dictionary<string, Snapshot> inputs, JsonElement source)
    {
        foreach (var (name, actual) in inputs)
        {
            var expected = source.GetProperty(name);
            Assert.Equal(Text(expected, "dtype"), actual.Dtype);
            Assert.Equal("cpu", Text(expected, "device"));
            Assert.Equal(Shape(expected), actual.Shape);
            Assert.Equal(Longs(expected.GetProperty("stride")), actual.Stride);
            Assert.Equal(expected.GetProperty("storageOffset").GetInt64(), actual.StorageOffset);
            Assert.Equal(Text(expected, "sha256"), actual.Sha256);
        }
    }

    private static void EqualSnapshot(Snapshot before, Snapshot after)
    {
        Assert.Equal(before.Shape, after.Shape); Assert.Equal(before.Stride, after.Stride);
        Assert.Equal(before.StorageOffset, after.StorageOffset); Assert.Equal(before.Dtype, after.Dtype);
        Assert.Equal(before.Sha256, after.Sha256); Assert.Equal(before.RequiresGrad, after.RequiresGrad);
    }

    private static float[] ReadValues(JsonElement record)
    {
        Assert.Equal("float32", Text(record, "dtype"));
        var values = record.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        Assert.Equal(Shape(record).Aggregate(1L, (a, b) => checked(a * b)), values.LongLength);
        Assert.All(values, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(Text(record, "sha256"), Hash(MemoryMarshal.AsBytes(values.AsSpan())));
        return values;
    }

    private static void Compare(Snapshot actual, JsonElement expected, string label)
    {
        Assert.Equal("float32", actual.Dtype);
        Assert.False(actual.RequiresGrad);
        Assert.Equal(Shape(expected), actual.Shape);
        float[] wanted = ReadValues(expected);
        Assert.Equal(wanted.Length, actual.Values.Length);
        for (int index = 0; index < wanted.Length; index++)
        {
            double delta = Math.Abs((double)actual.Values[index] - wanted[index]);
            double bound = 3e-5 + 3e-5 * Math.Abs(wanted[index]);
            Assert.True(float.IsFinite(actual.Values[index]) && delta <= bound,
                $"{label}[{index}]: actual={actual.Values[index]:R}, source={wanted[index]:R}, delta={delta:R}, bound={bound:R}.");
        }
    }

    private static void WriteTrace(string id, JsonElement root, JsonElement definition, JsonElement reference,
        Dictionary<string, Snapshot> inputs, Dictionary<string, Snapshot> inputsAfter, Parameter[] parameters,
        Parameter[] parametersAfter, Dictionary<string, Snapshot> captures, List<Step> steps,
        List<Dictionary<string, Snapshot>> finals, List<bool> restoredGrad, List<bool> captureGrad)
    {
        string? directory = Environment.GetEnvironmentVariable("COMFYSHARP_SD15_PIPELINE_TRACE_DIR");
        if (string.IsNullOrWhiteSpace(directory)) return;
        Directory.CreateDirectory(directory);
        var runtime = JsonSerializer.SerializeToNode(SdRuntimeIdentity.CaptureAfterForward())!.AsObject();
        runtime["capturePoint"] = "after_three_complete_pipeline_runs";
        runtime["nativeLibraries"]!["capturePoint"] = "process_cache_first_access_after_native_work_may_precede_this_case";
        var trace = new { id, profile = Profile, backendCommit = Backend, collectorCommit = Collector,
            target = SdSamplingReferenceTests.Target, protocolSha256 = Sd15PipelineCases.ProtocolSha256,
            synthetic = true, diagnosticOnly = true, qualification = "none", definition,
            inputRecords = root.GetProperty("inputRecords"), texts = root.GetProperty("texts"),
            inputs, inputsAfter, parameters, parametersAfter, captures, steps, finals, restoredGrad, captureGrad,
            observerSequence = new[] { "off", "on", "off" },
            sigmaHatMeaning = "No churn: the observed sigma is also sigmaHat, without a reconstructed primitive.",
            sourceAuxiliaryProjectionCalls = reference.GetProperty("sourceAuxiliaryProjectionCalls"),
            productProjectionPolicy = "SD1-L requests unprojected pooling; auxiliary projection calls are not instrumented here.",
            runtimeIdentity = runtime };
        var settings = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals };
        string path = Path.Combine(directory, id + ".json"), temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(trace, settings));
        File.Move(temporary, path, overwrite: false);
    }

    private static string BoundaryName(Sd15PipelineBoundary value) => value switch
    {
        Sd15PipelineBoundary.PositiveHidden => "positive.hidden", Sd15PipelineBoundary.PositivePooled => "positive.pooled",
        Sd15PipelineBoundary.NegativeHidden => "negative.hidden", Sd15PipelineBoundary.NegativePooled => "negative.pooled",
        Sd15PipelineBoundary.InitialDiffusionLatent => "initialDiffusionLatent", Sd15PipelineBoundary.FinalDiffusionLatent => "finalDiffusionLatent",
        Sd15PipelineBoundary.RawVaeLatent => "rawVaeLatent", Sd15PipelineBoundary.Image => "image", _ => throw new ArgumentOutOfRangeException(nameof(value))
    };
    private static string Text(JsonElement value, string name) => value.GetProperty(name).GetString()!;
    private static long[] Longs(JsonElement value) => value.EnumerateArray().Select(v => v.GetInt64()).ToArray();
    private static long[] Shape(JsonElement value) => Longs(value.GetProperty("shape"));
    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    public void Dispose() => set_num_threads(previousThreads);
}
