using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Each batching policy has its own same-platform frozen-source reference.
/// The source itself does not satisfy a universal numerical equality between policies.</summary>
[Collection("Classical VAE")]
public sealed class SdDenoiserReferenceTests : IDisposable
{
    private readonly int previousThreads;

    public SdDenoiserReferenceTests()
    {
        NativeRuntimeBootstrap.Initialize();
        previousThreads = get_num_threads();
        set_num_threads(1);
    }

    [Theory]
    [InlineData("guidance-2-3", "separate")]
    [InlineData("guidance-2-3", "concatenateCompatible")]
    [InlineData("guidance-3-3", "separate")]
    [InlineData("guidance-3-3", "concatenateCompatible")]
    [InlineData("guidance-2-10", "separate")]
    [InlineData("guidance-2-10", "concatenateCompatible")]
    public void PolicyMatchesItsSamePlatformSource(string identifier, string policy)
    {
        using var document = SdSamplingReferenceTests.Corpus("guidance");
        Assert.Equal(1, get_num_threads());
        Assert.Equal(1, get_num_interop_threads());
        var root = document.RootElement;
        var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
        var referenceConfig = root.GetProperty("config");
        Assert.Equal(config.BaseChannels, referenceConfig.GetProperty("baseChannels").GetInt32());
        Assert.Equal(config.ContextSize, referenceConfig.GetProperty("contextSize").GetInt32());
        Assert.Equal("fixedCount", referenceConfig.GetProperty("headMode").GetString());
        Assert.Equal(config.HeadParameter, referenceConfig.GetProperty("headParameter").GetInt32());
        Assert.False(referenceConfig.GetProperty("useLinearProjection").GetBoolean());
        Assert.Equal("epsilon", root.GetProperty("predictionKind").GetString());
        using var bank = SdSyntheticInputs.CreateUnet(config);
        VerifyParameters(bank, root.GetProperty("parameters"));
        using var model = new SdUnet(bank);
        using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        var cases = root.GetProperty("cases").EnumerateArray().ToArray();
        Assert.Equal(3, cases.Length);
        var reference = Assert.Single(cases, item => item.GetProperty("id").GetString() == identifier);
        var expected = reference.GetProperty("outputs").GetProperty(policy);
        Assert.Equal(3, reference.GetProperty("repeatHashes").GetArrayLength());
        foreach (var repeat in reference.GetProperty("repeatHashes").EnumerateArray())
            Assert.Equal(expected.GetProperty("sha256").GetString(), repeat.GetProperty(policy).GetString());

        using var scope = NewDisposeScope();
        // Input() verifies each serialized F32 payload against its SHA256.
        var latent = SdSamplingReferenceTests.Input(reference.GetProperty("latent"));
        var sigma = SdSamplingReferenceTests.Input(reference.GetProperty("sigma"));
        var positive = SdSamplingReferenceTests.Input(reference.GetProperty("positive"));
        var negative = SdSamplingReferenceTests.Input(reference.GetProperty("negative"));
        bool compatible = SdDenoiser.TryCommonContextLength(positive.shape[1], negative.shape[1], out long common);
        Assert.Equal(reference.GetProperty("concatEligible").GetBoolean(), compatible);
        Assert.Equal(reference.GetProperty("commonLength").GetInt64(), common);
        var options = new SdGuidanceOptions
        {
            Scale = reference.GetProperty("scale").GetDouble(),
            BatchMode = policy == "separate" ? SdGuidanceBatchMode.Separate : SdGuidanceBatchMode.ConcatenateCompatible
        };
        string? firstHash = null;
        for (int repeat = 0; repeat < 3; repeat++)
        {
            using var actual = denoiser.DenoiseGuided(latent, sigma, positive, negative, options);
            SdSamplingReferenceTests.Compare(actual, expected);
            string hash = Hash(actual);
            if (firstHash is null) firstHash = hash;
            else Assert.Equal(firstHash, hash);
        }
    }

    private static void VerifyParameters(UnetWeightSet bank, JsonElement records)
    {
        var schema = UnetWeightSchema.Describe(bank.Config);
        var parameters = records.EnumerateArray().ToArray();
        Assert.Equal(686, schema.Count);
        Assert.Equal(schema.Count, parameters.Length);
        Assert.Equal(schema.Keys.Order(StringComparer.Ordinal),
            parameters.Select(item => item.GetProperty("name").GetString()).Order(StringComparer.Ordinal));
        foreach (var reference in parameters)
        {
            string name = reference.GetProperty("name").GetString()!;
            long[] shape = reference.GetProperty("shape").EnumerateArray().Select(item => item.GetInt64()).ToArray();
            Assert.Equal(shape, schema[name]);
            var parameter = bank.GetTensor(name);
            Assert.Equal(shape, parameter.shape);
            Assert.True(Hash(parameter) == reference.GetProperty("sha256").GetString(), $"Source parameter bytes differ: {name}.");
        }
    }

    private static string Hash(Tensor value)
    {
        Assert.Equal(ScalarType.Float32, value.dtype);
        using var contiguous = value.contiguous();
        float[] values = contiguous.data<float>().ToArray();
        return Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    }

    public void Dispose() => set_num_threads(previousThreads);
}
