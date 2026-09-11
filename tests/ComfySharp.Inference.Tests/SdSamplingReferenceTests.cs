using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Expected values are produced by verified frozen Python source in a separate laboratory.
/// Tests consume checked-in JSON only and never invoke that laboratory.</summary>
[Collection("Classical VAE")]
public sealed class SdSamplingReferenceTests : IDisposable
{
    private readonly int previousThreads;

    public SdSamplingReferenceTests()
    {
        NativeRuntimeBootstrap.Initialize();
        previousThreads = get_num_threads();
        set_num_threads(1);
    }

    public void Dispose() => set_num_threads(previousThreads);

    internal static string Target => (OperatingSystem.IsWindows(), OperatingSystem.IsLinux(), RuntimeInformation.ProcessArchitecture) switch
    {
        (true, _, Architecture.X64) => "win-x64",
        (_, true, Architecture.X64) => "linux-x64",
        (false, false, Architecture.Arm64) when OperatingSystem.IsMacOS() => "osx-arm64",
        _ => throw new PlatformNotSupportedException("No source reference has been qualified for this target.")
    };

    internal static JsonDocument Corpus(string component)
    {
        SdReferenceRuntime.Verify();
        var bytes = ClipReferenceTests.Resource($"sd-components.{Target}.{component}.json");
        using var manifest = ReadPinnedManifest();
        var identity = manifest.RootElement.GetProperty("components").EnumerateArray().Single(c => c.GetProperty("name").GetString() == component);
        Assert.Equal(identity.GetProperty("bytes").GetInt32(), bytes.Length);
        Assert.Equal(identity.GetProperty("sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(bytes)));
        var document = JsonDocument.Parse(bytes);
        Assert.Equal("sd-components-native210-cpu-f32-v1", document.RootElement.GetProperty("profile").GetString());
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", document.RootElement.GetProperty("backendCommit").GetString());
        Assert.Equal(Target, document.RootElement.GetProperty("target").GetString());
        return document;
    }

    internal static JsonDocument ReadPinnedManifest()
    {
        // Independent source CI 34632139846 at ComfySharp commit 86812a9dae975a6a59586de87d9d7d3e0bd1ef99.
        string hash = Target switch
        {
            "win-x64" => "e40f65335711a33de88acbb787c7eb8e4c3ce04df020b1a3b62c14dda996bab2",
            "linux-x64" => "53ccb79667d7c5f191d91d4bc62dc70fec7612809836f77fa7e388f334716d9a",
            "osx-arm64" => "a6d97ffdb955fb0da7ed4c7319751143bacf3ba74c0dcaac36bf94e4186edde6",
            _ => throw new PlatformNotSupportedException()
        };
        var bytes = ClipReferenceTests.Resource($"sd-components.{Target}.manifest.json");
        Assert.Equal(hash, Convert.ToHexStringLower(SHA256.HashData(bytes)));
        return JsonDocument.Parse(bytes);
    }

    internal static Tensor Input(JsonElement record)
    {
        long[] shape = record.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray();
        Assert.Equal("float32", record.GetProperty("dtype").GetString());
        var values = record.GetProperty("values").EnumerateArray().Select(ReadFloat).ToArray();
        Assert.Equal(record.GetProperty("sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan()))));
        return tensor(values, shape, dtype: ScalarType.Float32, device: CPU);
    }

    internal static void Compare(Tensor actual, JsonElement expected)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()), actual.shape);
        if (expected.GetProperty("dtype").GetString() == "int64")
        {
            Assert.Equal(ScalarType.Int64, actual.dtype);
            long[] values = expected.GetProperty("values").EnumerateArray().Select(v => v.GetInt64()).ToArray();
            Assert.Equal(expected.GetProperty("sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan()))));
            Assert.Equal(values, actual.data<long>().ToArray());
            return;
        }
        Assert.Equal(ScalarType.Float32, actual.dtype);
        float[] reference = expected.GetProperty("values").EnumerateArray().Select(ReadFloat).ToArray();
        Assert.Equal(expected.GetProperty("sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(reference.AsSpan()))));
        using var contiguous = actual.contiguous();
        var observed = contiguous.data<float>().ToArray();
        Assert.Equal(reference.Length, observed.Length);
        for (int i = 0; i < reference.Length; i++)
        {
            float value = observed[i], wanted = reference[i];
            if (float.IsNaN(wanted)) Assert.True(float.IsNaN(value), $"Element {i}: expected NaN, actual {value}.");
            else if (float.IsInfinity(wanted)) Assert.Equal(wanted, value);
            else Assert.True(float.IsFinite(value) && Math.Abs((double)value - wanted) <= 3e-5 + 3e-5 * Math.Abs(wanted),
                $"Element {i}: actual={value:R}, source={wanted:R}, delta={Math.Abs((double)value - wanted):R}.");
        }
    }

    private static float ReadFloat(JsonElement value) => value.ValueKind == JsonValueKind.Number ? value.GetSingle() : value.GetString() switch
    {
        "NaN" => float.NaN, "Infinity" => float.PositiveInfinity, "-Infinity" => float.NegativeInfinity,
        _ => throw new InvalidDataException("Unrecognized floating-point representation.")
    };

    [Fact]
    public void ScheduleTablesAndIndicesMatchSamePlatformSource()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var doc = Corpus("sampling");
        var reference = doc.RootElement.GetProperty("schedule");
        var sampling = SdDiscreteSampling.Default;
        Compare(tensor(sampling.Sigmas.ToArray()), reference.GetProperty("sigmas"));
        Compare(tensor(sampling.LogSigmas.ToArray()), reference.GetProperty("logSigmas"));
        using var indices = sampling.Timestep(Input(reference.GetProperty("timestepInputs")));
        Compare(indices, reference.GetProperty("timesteps"));
        using var interpolated = sampling.Sigma(Input(reference.GetProperty("sigmaInputs")));
        Compare(interpolated, reference.GetProperty("interpolatedSigmas"));
        var percents = reference.GetProperty("percentInputs").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var expected = reference.GetProperty("percentSigmas").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        for (int i = 0; i < percents.Length; i++)
            Assert.InRange(Math.Abs(sampling.PercentToSigma(percents[i]) - expected[i]), 0, 3e-5 + 3e-5 * Math.Abs(expected[i]));
    }

    [Fact]
    public void EpsilonVelocityInputAndNoiseScalingMatchSource()
    {
        NativeRuntimeBootstrap.Initialize();
        using var doc = Corpus("sampling");
        foreach (var reference in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            using var scope = NewDisposeScope();
            var sigma = Input(reference.GetProperty("sigma"));
            var latent = Input(reference.GetProperty("latent"));
            var prediction = Input(reference.GetProperty("prediction"));
            var kind = reference.GetProperty("predictionKind").GetString() == "epsilon" ? SdPredictionKind.Epsilon : SdPredictionKind.Velocity;
            using var input = SdSamplingMath.ScaleInput(latent, sigma);
            using var denoised = SdSamplingMath.Denoised(latent, prediction, sigma, kind);
            using var noise = SdSamplingMath.NoiseScaling(prediction, latent, sigma);
            using var maximum = SdSamplingMath.NoiseScaling(prediction, latent, sigma, maximumDenoise: true);
            Compare(input, reference.GetProperty("scaledInput"));
            Compare(denoised, reference.GetProperty("denoised"));
            Compare(noise, reference.GetProperty("noiseScaled"));
            Compare(maximum, reference.GetProperty("maximumNoiseScaled"));
        }
    }

    [Fact]
    public void GuidanceAndLatentFormatBoundariesMatchSource()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var doc = Corpus("sampling");
        var guidance = doc.RootElement.GetProperty("guidance");
        var positive = Input(guidance.GetProperty("conditional"));
        var negative = Input(guidance.GetProperty("unconditional"));
        foreach (var reference in guidance.GetProperty("cases").EnumerateArray())
        {
            double scale = reference.GetProperty("scale").GetDouble();
            bool omit = SdSamplingMath.CanOmitUnconditional(scale, reference.GetProperty("disableOptimization").GetBoolean());
            Assert.Equal(reference.GetProperty("omitted").GetBoolean(), omit);
            using var actual = SdSamplingMath.Guide(positive, omit ? null : negative, scale);
            Compare(actual, reference.GetProperty("output"));
        }
        var formats = doc.RootElement.GetProperty("latentFormats");
        var raw = Input(formats.GetProperty("input"));
        foreach (var reference in formats.GetProperty("cases").EnumerateArray())
        {
            double scale = reference.GetProperty("kind").GetString() == "SD15" ? SdSamplingMath.Sd15LatentScale : SdSamplingMath.SdxlLatentScale;
            Assert.Equal(reference.GetProperty("scale").GetDouble(), scale);
            using var diffusion = SdSamplingMath.ProcessLatentIn(raw, scale);
            using var roundTrip = SdSamplingMath.ProcessLatentOut(diffusion, scale);
            Compare(diffusion, reference.GetProperty("diffusion"));
            Compare(roundTrip, reference.GetProperty("roundTrip"));
        }
    }
}
