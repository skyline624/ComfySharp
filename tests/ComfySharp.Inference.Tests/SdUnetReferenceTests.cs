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
        string? offsetText = Environment.GetEnvironmentVariable("COMFYSHARP_SD_LATENT_OFFSET");
        int? latentOffset = null;
        if (offsetText is not null)
        {
            Assert.True(int.TryParse(offsetText, System.Globalization.NumberStyles.None,
                System.Globalization.CultureInfo.InvariantCulture, out int parsedOffset) && parsedOffset is >= 0 and <= 15,
                "COMFYSHARP_SD_LATENT_OFFSET must be an integer from 0 through 15.");
            latentOffset = parsedOffset;
        }
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
        if (latentOffset is int offset)
        {
            // Diagnostic allocation control only. Keep input values, shape and strides;
            // the ordinary fixtureTensor path is unchanged when the option is absent.
            string inputHash = TensorHash(latent);
            Assert.Equal(reference.GetProperty("latent").GetProperty("sha256").GetString(), inputHash);
            long elements = latent.numel();
            var backing = empty(new long[] { checked(elements + offset) }, dtype: ScalarType.Float32, device: CPU);
            using (var firstBackingValue = backing[0])
                Assert.True(CpuModelWeightBank.IsAligned(firstBackingValue), "Native diagnostic backing must be aligned to 64 bytes.");
            var copied = backing.narrow(0, offset, elements).reshape(latent.shape);
            copied.copy_(latent);
            Assert.Equal(latent.stride(), copied.stride());
            Assert.Equal(inputHash, TensorHash(copied));
            Assert.Equal(inputHash, TensorHash(latent));
            latent = copied;
        }
        string? fineTraceDirectory = Environment.GetEnvironmentVariable("COMFYSHARP_SD_UNET_FINE_TRACE_DIR");
        if (!string.IsNullOrWhiteSpace(fineTraceDirectory) && modelId == "sd15-reduced" && caseName == "square")
        {
            SdUnetFineDiagnostic.Run(fineTraceDirectory, model, bank, latent, timesteps, context,
                reference, modelReference.GetProperty("parameters"));
            return;
        }
        string? firstHash = null;
        for (int repetition = 0; repetition < 3; repetition++)
        {
            using var output = model.Forward(latent, timesteps, context);
            if (repetition == 0 && traces is not null)
            {
                traces["output"] = TraceRecord(output);
                // Observe after the first forward so the baseline's allocation order is
                // not changed by an extra scalar-view/hash capture before graph execution.
                using var firstLatentValue = latent[0, 0, 0, 0];
                string latentHash = TensorHash(latent);
                string? fixtureHash = reference.GetProperty("latent").GetProperty("sha256").GetString();
                Assert.Equal(fixtureHash, latentHash);
                var latentInput = new
                {
                    allocation = latentOffset.HasValue ? "nativeOffset" : "fixtureTensor",
                    offsetElements = latentOffset,
                    fixtureSha256 = fixtureHash,
                    sha256 = latentHash,
                    shape = latent.shape,
                    stride = latent.stride(),
                    aligned64 = CpuModelWeightBank.IsAligned(firstLatentValue)
                };
                Directory.CreateDirectory(traceDirectory!);
                File.WriteAllText(Path.Combine(traceDirectory!, modelId + "--" + caseName + ".json"),
                    JsonSerializer.Serialize(new { id = modelId + "/" + caseName, synthetic = true, latentInput, tensors = traces }));
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
