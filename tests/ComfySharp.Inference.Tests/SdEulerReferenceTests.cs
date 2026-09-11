using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Consumes independently collected frozen-source trajectories, never executes Python.
/// Source collection and manifest identities were independently audited before fixture import.
/// Numerical failures do not authorize changing the source outputs or the fixed comparison bound.</summary>
[Collection("Classical VAE")]
public sealed class SdEulerReferenceTests : IDisposable
{
    private const string Backend = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a";
    private const string Profile = "sd-euler-native210-cpu-f32-v1";
    private const string ProtocolHash = "7868368a38fdf895fbe8698d4ab5bbbea1352a11df6dd7b3bd6c1e776c55cd05";
    // Independent source CI 34650726593, collector commit
    // 96fec72dd928276fc3f0ddc93a17406ea8249073; imported without changing bytes.
    private const string WindowsManifestHash = "b528a5f0055ddbae5dad409dffc1bac7a74d08295b638920165791441eaf17a8";
    private const string LinuxManifestHash = "b1aaf6e2e1a6c8d51fe261012cd20257980d8e4a4561f5bbb016c915be72fa1e";
    private const string MacManifestHash = "002da5ae9cb823cdf88523c9e8a22328cd2305bbafd82620ab708e8eb323292f";
    private static readonly string[] CaseIds = ["one-step", "three-step-batch", "unequal-separate",
        "unequal-concatenated", "near-one-null", "near-one-present", "near-one-disabled", "velocity-three-step"];
    private readonly int previousThreads;

    public SdEulerReferenceTests()
    {
        NativeRuntimeBootstrap.Initialize();
        previousThreads = get_num_threads();
        set_num_threads(1);
    }

    [Theory]
    [InlineData("one-step")]
    [InlineData("three-step-batch")]
    [InlineData("unequal-separate")]
    [InlineData("unequal-concatenated")]
    [InlineData("near-one-null")]
    [InlineData("near-one-present")]
    [InlineData("near-one-disabled")]
    [InlineData("velocity-three-step")]
    public void TrajectoryMatchesItsSamePlatformFrozenSource(string identifier)
    {
        using var document = ReadCorpus();
        var root = document.RootElement;
        var cases = root.GetProperty("cases").EnumerateArray().ToArray();
        Assert.Equal(CaseIds.Order(StringComparer.Ordinal),
            cases.Select(c => c.GetProperty("id").GetString()!).Order(StringComparer.Ordinal));
        var reference = Assert.Single(cases, c => c.GetProperty("id").GetString() == identifier);
        var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
        CheckConfiguration(root.GetProperty("config"));
        var (kind, options) = CheckCase(reference, identifier);
        using var bank = SdSyntheticInputs.CreateUnet(config);
        var parametersBefore = CheckParameters(bank, root.GetProperty("parameters"));
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, kind);
        using var sampler = new SdEulerSampler(denoiser);
        using var scope = NewDisposeScope();
        using var callerGrad = set_grad_enabled(true);
        var latent = SdSamplingReferenceTests.Input(reference.GetProperty("latent"));
        var sigmas = SdSamplingReferenceTests.Input(reference.GetProperty("sigmas"));
        var positive = SdSamplingReferenceTests.Input(reference.GetProperty("positive"));
        var negativeReference = reference.GetProperty("negative");
        Tensor? negative = negativeReference.ValueKind == JsonValueKind.Null ? null : SdSamplingReferenceTests.Input(negativeReference);
        var inputs = new Dictionary<string, Tensor> { ["latent"] = latent, ["sigmas"] = sigmas, ["positive"] = positive };
        if (negative is not null) inputs.Add("negative", negative);
        var before = inputs.ToDictionary(p => p.Key, p => Capture(p.Value));
        var states = new List<StepSnapshot>();
        var outputs = new List<TensorSnapshot>();
        var restoredGrad = new List<bool>();
        var outputRequiresGrad = new List<bool>();
        // Always collect managed snapshots, so opting into file output adds no observer calls.
        // The callback never retains a Tensor wrapper, creates a native view or mutates its input.
        try
        {
            for (int repeat = 0; repeat < 3; repeat++)
            {
                sampler.DiagnosticObserver = repeat == 1 ? (index, current, denoised, sigma) =>
                    states.Add(new(index, Capture(current), Capture(denoised), Capture(sigma), is_grad_enabled())) : null;
                using var actual = sampler.Sample(latent, sigmas, positive, negative, options);
                outputs.Add(Capture(actual));
                restoredGrad.Add(is_grad_enabled());
                outputRequiresGrad.Add(actual.requires_grad);
            }
        }
        finally { sampler.DiagnosticObserver = null; }
        var after = inputs.ToDictionary(p => p.Key, p => Capture(p.Value).Sha256);
        var parametersAfter = CaptureParameters(bank);
        WriteTraceIfRequested(identifier, root, reference, before, after, parametersBefore, parametersAfter,
            states, outputs, restoredGrad, outputRequiresGrad);

        // Evidence above is complete before any numerical comparison or repeat/neutrality assertion.
        Assert.All(restoredGrad, value => Assert.True(value));
        Assert.All(outputRequiresGrad, value => Assert.False(value));
        foreach (var input in before) Assert.Equal(input.Value.Sha256, after[input.Key]);
        AssertParameters(parametersAfter, root.GetProperty("parameters"));
        Assert.All(outputs, value => Assert.Equal(outputs[0].Sha256, value.Sha256));
        var expectedSteps = reference.GetProperty("steps").EnumerateArray().ToArray();
        Assert.Equal(sigmas.shape[0] - 1, states.Count);
        Assert.Equal(expectedSteps.Length, states.Count);
        for (int index = 0; index < states.Count; index++)
        {
            var actual = states[index];
            var expected = expectedSteps[index];
            Assert.Equal((long)index, actual.Index);
            Assert.Equal((long)index, expected.GetProperty("index").GetInt64());
            Assert.False(actual.GradEnabled);
            Compare(actual.X, expected.GetProperty("x"), $"{identifier}/step-{index}/x");
            Compare(actual.Denoised, expected.GetProperty("denoised"), $"{identifier}/step-{index}/denoised");
            Assert.Equal(Shape(expected.GetProperty("sigma")), actual.Sigma.Shape);
            Assert.Equal(ReadValues(expected.GetProperty("sigma")), actual.Sigma.Values);
            // No churn: the source sigmaHat is the same actual scalar sigma, not a new C# operation.
            Assert.Equal(Shape(expected.GetProperty("sigmaHat")), actual.Sigma.Shape);
            Assert.Equal(ReadValues(expected.GetProperty("sigmaHat")), actual.Sigma.Values);
        }
        for (int repeat = 0; repeat < outputs.Count; repeat++)
            Compare(outputs[repeat], reference.GetProperty("output"), $"{identifier}/repeat-{repeat}/output");
    }

    private static JsonDocument ReadCorpus()
    {
        SdReferenceRuntime.Verify();
        Assert.True(BitConverter.IsLittleEndian, "Frozen tensor records use little-endian Float32 bytes.");
        string target = SdSamplingReferenceTests.Target;
        string hash = target switch { "win-x64" => WindowsManifestHash, "linux-x64" => LinuxManifestHash,
            "osx-arm64" => MacManifestHash, _ => throw new PlatformNotSupportedException() };
        Assert.Equal(64, hash.Length);
        var manifestBytes = ClipReferenceTests.Resource($"sd-euler.{target}.manifest.json");
        Assert.Equal(hash, Hash(manifestBytes));
        using var manifest = JsonDocument.Parse(manifestBytes);
        var metadata = manifest.RootElement;
        Assert.Equal(Backend, metadata.GetProperty("backendCommit").GetString());
        Assert.Equal(Profile, metadata.GetProperty("profile").GetString());
        Assert.Equal(target, metadata.GetProperty("target").GetString());
        Assert.Equal(ProtocolHash, metadata.GetProperty("protocolSha256").GetString());
        Assert.True(metadata.GetProperty("syntheticWeights").GetBoolean());
        Assert.False(metadata.GetProperty("pretrainedWeightsUsed").GetBoolean());
        Assert.Equal("sha256-name-lcg-high16-power2-v1", metadata.GetProperty("parameterRecipe").GetString());
        Assert.Equal(3e-5, metadata.GetProperty("comparison").GetProperty("absoluteTolerance").GetDouble());
        Assert.Equal(3e-5, metadata.GetProperty("comparison").GetProperty("relativeTolerance").GetDouble());
        var runtime = metadata.GetProperty("runtime");
        Assert.Equal("3.12.10", runtime.GetProperty("python").GetString());
        Assert.Equal(target == "osx-arm64" ? "2.10.0" : "2.10.0+cpu", runtime.GetProperty("torch").GetString());
        Assert.Equal("449b1768410104d3ed79d3bcfe4ba1d65c7f22c0", runtime.GetProperty("torchGit").GetString());
        Assert.Equal(1, runtime.GetProperty("threads").GetInt32());
        Assert.Equal(1, runtime.GetProperty("interopThreads").GetInt32());
        var component = metadata.GetProperty("component");
        Assert.Equal("euler", component.GetProperty("name").GetString());
        Assert.Equal("euler.json", component.GetProperty("file").GetString());
        byte[] bytes = ClipReferenceTests.Resource($"sd-euler.{target}.json");
        Assert.Equal(component.GetProperty("bytes").GetInt32(), bytes.Length);
        Assert.Equal(component.GetProperty("sha256").GetString(), Hash(bytes));
        var document = JsonDocument.Parse(bytes);
        try
        {
            var root = document.RootElement;
            Assert.Equal(Backend, root.GetProperty("backendCommit").GetString());
            Assert.Equal(Profile, root.GetProperty("profile").GetString());
            Assert.Equal(target, root.GetProperty("target").GetString());
            Assert.True(JsonElement.DeepEquals(metadata.GetProperty("sources"), root.GetProperty("sources")));
            return document;
        }
        catch { document.Dispose(); throw; }
    }

    private static void CheckConfiguration(JsonElement config)
    {
        Assert.Equal(32, config.GetProperty("baseChannels").GetInt32());
        Assert.Equal(16, config.GetProperty("contextSize").GetInt32());
        Assert.Equal("fixedCount", config.GetProperty("headMode").GetString());
        Assert.Equal(4, config.GetProperty("headParameter").GetInt32());
        Assert.False(config.GetProperty("useLinearProjection").GetBoolean());
    }

    private static (SdPredictionKind, SdGuidanceOptions) CheckCase(JsonElement reference, string identifier)
    {
        bool nearOne = identifier.StartsWith("near-one-", StringComparison.Ordinal);
        bool concatenated = identifier == "unequal-concatenated";
        bool velocity = identifier == "velocity-three-step";
        bool disabled = identifier == "near-one-disabled";
        Assert.Equal(velocity ? "velocity" : "epsilon", reference.GetProperty("predictionKind").GetString());
        Assert.Equal(concatenated ? "concatenateCompatible" : "separate", reference.GetProperty("policy").GetString());
        Assert.Equal(nearOne ? 1.0000000005 : 3.5, reference.GetProperty("scale").GetDouble());
        Assert.Equal(disabled, reference.GetProperty("disableScaleOneOptimization").GetBoolean());
        Assert.Equal(identifier == "near-one-null", reference.GetProperty("negative").ValueKind == JsonValueKind.Null);
        float[] sigmas = identifier == "one-step" ? [1.5f, 0f] : nearOne ? [0.5f, 0.125f, 0f] : [1.5f, 0.5f, 0.125f, 0f];
        Assert.Equal(sigmas, ReadValues(reference.GetProperty("sigmas")));
        Assert.Equal(new long[] { sigmas.Length }, Shape(reference.GetProperty("sigmas")));
        long batch = nearOne || identifier == "one-step" ? 1 : 2;
        long positiveLength = identifier is "one-step" or "three-step-batch" ? 3 : 2;
        Assert.Equal(new long[] { batch, 4, 4, 5 }, Shape(reference.GetProperty("latent")));
        Assert.Equal(new long[] { batch, positiveLength, 16 }, Shape(reference.GetProperty("positive")));
        if (identifier != "near-one-null") Assert.Equal(new long[] { batch, 3, 16 }, Shape(reference.GetProperty("negative")));
        foreach (string name in new[] { "latent", "positive", "negative" })
            if (reference.GetProperty(name).ValueKind != JsonValueKind.Null) _ = ReadValues(reference.GetProperty(name));
        Assert.True(reference.GetProperty("inputsUnchanged").GetBoolean());
        Assert.True(reference.GetProperty("parametersUnchanged").GetBoolean());
        Assert.Equal(new[] { "off", "on", "off" }, reference.GetProperty("observerSequence").EnumerateArray().Select(v => v.GetString()));
        var repeats = reference.GetProperty("repeatHashes").EnumerateArray().ToArray();
        Assert.Equal(3, repeats.Length);
        Assert.All(repeats, value => Assert.Equal(reference.GetProperty("output").GetProperty("sha256").GetString(), value.GetString()));
        Assert.Equal(sigmas.Length - 1, reference.GetProperty("modelCallsPerRun").GetInt32());
        Assert.Equal(new[] { batch }, reference.GetProperty("modelSigmaShape").EnumerateArray().Select(v => v.GetInt64()));
        return (velocity ? SdPredictionKind.Velocity : SdPredictionKind.Epsilon, new()
        {
            Scale = nearOne ? 1.0000000005 : 3.5,
            BatchMode = concatenated ? SdGuidanceBatchMode.ConcatenateCompatible : SdGuidanceBatchMode.Separate,
            DisableScaleOneOptimization = disabled
        });
    }

    private sealed record TensorSnapshot(long[] Shape, long[] Stride, string Dtype, float[] Values, string Sha256);
    private sealed record StepSnapshot(long Index, TensorSnapshot X, TensorSnapshot Denoised, TensorSnapshot Sigma, bool GradEnabled);
    private sealed record ParameterSnapshot(string Name, long[] Shape, string Sha256);

    private static TensorSnapshot Capture(Tensor value)
    {
        Assert.Equal(ScalarType.Float32, value.dtype);
        Assert.Equal(TorchSharp.DeviceType.CPU, value.device_type);
        Assert.True(value.is_contiguous(), "Euler diagnostic captures require contiguous tensors; no native copy is inserted.");
        float[] values = value.data<float>().ToArray();
        return new(value.shape, value.stride(), "float32", values, Hash(MemoryMarshal.AsBytes(values.AsSpan())));
    }

    private static ParameterSnapshot[] CaptureParameters(UnetWeightSet bank) =>
        UnetWeightSchema.Describe(bank.Config).Keys.Order(StringComparer.Ordinal).Select(name =>
        {
            var value = Capture(bank.GetTensor(name));
            return new ParameterSnapshot(name, value.Shape, value.Sha256);
        }).ToArray();

    private static ParameterSnapshot[] CheckParameters(UnetWeightSet bank, JsonElement expected)
    {
        var actual = CaptureParameters(bank);
        AssertParameters(actual, expected);
        return actual;
    }

    private static void AssertParameters(ParameterSnapshot[] actual, JsonElement expected)
    {
        var records = expected.EnumerateArray().OrderBy(r => r.GetProperty("name").GetString(), StringComparer.Ordinal).ToArray();
        Assert.Equal(686, actual.Length);
        Assert.Equal(686, records.Length);
        for (int index = 0; index < actual.Length; index++)
        {
            Assert.Equal(records[index].GetProperty("name").GetString(), actual[index].Name);
            Assert.Equal(Shape(records[index]), actual[index].Shape);
            Assert.True(records[index].GetProperty("sha256").GetString() == actual[index].Sha256,
                $"Source parameter bytes differ: {actual[index].Name}.");
        }
    }

    private static long[] Shape(JsonElement record) => record.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray();
    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    private static float[] ReadValues(JsonElement record)
    {
        Assert.Equal("float32", record.GetProperty("dtype").GetString());
        float[] values = record.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        Assert.Equal(Shape(record).Aggregate(1L, (a, b) => checked(a * b)), values.LongLength);
        Assert.All(values, v => Assert.True(float.IsFinite(v), "Euler reference corpus must remain finite."));
        Assert.Equal(record.GetProperty("sha256").GetString(), Hash(MemoryMarshal.AsBytes(values.AsSpan())));
        return values;
    }

    private static void Compare(TensorSnapshot actual, JsonElement expected, string label)
    {
        Assert.Equal(Shape(expected), actual.Shape);
        float[] wanted = ReadValues(expected);
        Assert.Equal(wanted.Length, actual.Values.Length);
        for (int index = 0; index < wanted.Length; index++)
        {
            float value = actual.Values[index];
            double delta = Math.Abs((double)value - wanted[index]);
            double bound = 3e-5 + 3e-5 * Math.Abs(wanted[index]);
            Assert.True(float.IsFinite(value) && delta <= bound,
                $"{label}[{index}]: actual={value:R}, source={wanted[index]:R}, delta={delta:R}, bound={bound:R}.");
        }
    }

    private static void WriteTraceIfRequested(string identifier, JsonElement root, JsonElement reference,
        Dictionary<string, TensorSnapshot> before, Dictionary<string, string> after,
        ParameterSnapshot[] parametersBefore, ParameterSnapshot[] parametersAfter,
        List<StepSnapshot> states, List<TensorSnapshot> outputs, List<bool> restoredGrad, List<bool> outputRequiresGrad)
    {
        string? directory = Environment.GetEnvironmentVariable("COMFYSHARP_SD_EULER_TRACE_DIR");
        if (string.IsNullOrWhiteSpace(directory)) return;
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, identifier + ".json");
        var record = new
        {
            id = identifier, profile = Profile, target = SdSamplingReferenceTests.Target, backendCommit = Backend,
            synthetic = true, diagnosticOnly = true, qualification = "none", config = root.GetProperty("config"),
            predictionKind = reference.GetProperty("predictionKind"), policy = reference.GetProperty("policy"),
            scale = reference.GetProperty("scale"), disableScaleOneOptimization = reference.GetProperty("disableScaleOneOptimization"),
            inputs = before, inputHashesAfter = after, parameters = parametersBefore, parametersAfter,
            steps = states, outputs, observerSequence = new[] { "off", "on", "off" },
            repeatHashes = outputs.Select(v => v.Sha256), restoredGrad, outputRequiresGrad,
            sigmaHatMeaning = "No churn; the observed sigma is also sigmaHat. No second primitive is reconstructed.",
            runtimeIdentity = CaptureRuntimeAfterTrajectory()
        };
        // Named literals preserve nonfinite diagnostic values; assertions below still reject them.
        var settings = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals };
        string temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(record, settings));
        File.Move(temporary, path, overwrite: false);
    }

    private static System.Text.Json.Nodes.JsonObject CaptureRuntimeAfterTrajectory()
    {
        // Reuse the managed identity/cache machinery but describe this caller's actual boundary.
        // A preceding traced test may already have populated the process-wide library cache.
        var identity = JsonSerializer.SerializeToNode(SdRuntimeIdentity.CaptureAfterForward())!.AsObject();
        identity["capturePoint"] = "after_three_complete_euler_trajectories";
        identity["nativeLibraries"]!["capturePoint"] = "process_cache_first_access_after_completed_native_work_may_precede_this_case";
        return identity;
    }

    public void Dispose() => set_num_threads(previousThreads);
}
