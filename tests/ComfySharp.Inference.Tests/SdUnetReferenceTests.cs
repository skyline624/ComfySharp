using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Full reduced topology against separately executed frozen source, using the
/// reference for the current platform. The C# recipe supplies inputs, never expected outputs.</summary>
[Collection("Classical VAE")]
public sealed class SdUnetReferenceTests : IDisposable
{
    private readonly int previousThreads;

    public SdUnetReferenceTests()
    {
        NativeRuntimeBootstrap.Initialize();
        previousThreads = get_num_threads();
        set_num_threads(1);
    }

    public void Dispose() => set_num_threads(previousThreads);

    [Theory]
    [InlineData("sd15-reduced", "square")]
    [InlineData("sd15-reduced", "odd-rectangle")]
    [InlineData("sd15-reduced", "batch-distinct-time")]
    [InlineData("sd15-reduced", "batch-shared-time-odd")]
    [InlineData("sd2-reduced", "square")]
    [InlineData("sd2-reduced", "odd-rectangle")]
    [InlineData("sd2-reduced", "batch-distinct-time")]
    [InlineData("sd2-reduced", "batch-shared-time-odd")]
    public void FullGraphMatchesSamePlatformFrozenSource(string modelId, string caseName)
    {
        using var document = SdSamplingReferenceTests.Corpus("unet");
        var root = document.RootElement;
        var modelReference = root.GetProperty("models").GetProperty(modelId);
        bool linear = modelId == "sd2-reduced";
        var config = new SdUnetConfig(32, 16,
            linear ? SdAttentionHeadMode.FixedSize : SdAttentionHeadMode.FixedCount,
            linear ? 8 : 4, linear);
        CheckConfiguration(config, modelReference.GetProperty("config"));

        using var bank = SdSyntheticInputs.CreateUnet(config);
        CheckParameters(bank, modelReference.GetProperty("parameters"));
        using var model = new SdUnet(bank);
        string? traceDirectory = Environment.GetEnvironmentVariable("COMFYSHARP_SD_UNET_TRACE_DIR");
        Dictionary<string, object>? traces = null;
        if (!string.IsNullOrWhiteSpace(traceDirectory))
        {
            traces = new Dictionary<string, object>(StringComparer.Ordinal);
            model.DiagnosticObserver = (name, tensor) => traces[name] = TraceRecord(tensor);
        }
        var reference = root.GetProperty("cases").EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == modelId + "/" + caseName);
        Assert.Equal(modelId, reference.GetProperty("model").GetString());
        var sourceHashes = reference.GetProperty("repeatHashes").EnumerateArray().Select(item => item.GetString()).ToArray();
        Assert.Equal(3, sourceHashes.Length);
        Assert.All(sourceHashes, hash => Assert.Equal(reference.GetProperty("output").GetProperty("sha256").GetString(), hash));

        using var scope = NewDisposeScope();
        var latent = SdSamplingReferenceTests.Input(reference.GetProperty("latent"));
        var timesteps = SdSamplingReferenceTests.Input(reference.GetProperty("timesteps"));
        var context = SdSamplingReferenceTests.Input(reference.GetProperty("context"));
        string? firstHash = null;
        for (int repetition = 0; repetition < 3; repetition++)
        {
            using var output = model.Forward(latent, timesteps, context);
            if (repetition == 0 && traces is not null)
            {
                traces["output"] = TraceRecord(output);
                Directory.CreateDirectory(traceDirectory!);
                File.WriteAllText(Path.Combine(traceDirectory!, modelId + "--" + caseName + ".json"),
                    JsonSerializer.Serialize(new { id = modelId + "/" + caseName, synthetic = true, tensors = traces }));
                model.DiagnosticObserver = null;
            }
            SdSamplingReferenceTests.Compare(output, reference.GetProperty("output"));
            string hash = TensorHash(output);
            if (firstHash is null) firstHash = hash;
            else Assert.Equal(firstHash, hash);
        }
    }

    private static void CheckConfiguration(SdUnetConfig config, JsonElement reference)
    {
        Assert.Equal(config.BaseChannels, reference.GetProperty("baseChannels").GetInt32());
        Assert.Equal(config.ContextSize, reference.GetProperty("contextSize").GetInt32());
        Assert.Equal(config.HeadMode == SdAttentionHeadMode.FixedCount ? "fixedCount" : "fixedSize",
            reference.GetProperty("headMode").GetString());
        Assert.Equal(config.HeadParameter, reference.GetProperty("headParameter").GetInt32());
        Assert.Equal(config.UseLinearProjection, reference.GetProperty("useLinearProjection").GetBoolean());
    }

    private static void CheckParameters(UnetWeightSet bank, JsonElement parameters)
    {
        var schema = UnetWeightSchema.Describe(bank.Config);
        var records = parameters.EnumerateArray().ToArray();
        Assert.Equal(686, schema.Count);
        Assert.Equal(686, records.Length);
        Assert.Equal(schema.Keys.Order(StringComparer.Ordinal),
            records.Select(item => item.GetProperty("name").GetString()!).Order(StringComparer.Ordinal));
        foreach (var reference in records)
        {
            string name = reference.GetProperty("name").GetString()!;
            long[] expectedShape = reference.GetProperty("shape").EnumerateArray().Select(item => item.GetInt64()).ToArray();
            Assert.Equal(expectedShape, schema[name]);
            var tensor = bank.GetTensor(name);
            Assert.Equal(expectedShape, tensor.shape);
            Assert.True(reference.GetProperty("sha256").GetString() == TensorHash(tensor),
                $"Synthetic parameter bytes differ from the independent source input: {name}.");
        }
    }

    private static string TensorHash(Tensor tensor)
    {
        Assert.Equal(ScalarType.Float32, tensor.dtype);
        using var contiguous = tensor.contiguous();
        var values = contiguous.data<float>().ToArray();
        return Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    }

    private static object TraceRecord(Tensor tensor)
    {
        using var contiguous = tensor.contiguous();
        var values = contiguous.data<float>().ToArray();
        return new { shape = tensor.shape, dtype = "float32", values, stride = tensor.stride(),
            sha256 = Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan()))) };
    }

}
