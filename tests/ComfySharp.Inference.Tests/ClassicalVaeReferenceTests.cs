using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Reduced-width complete graphs compared with frozen same-platform CPU210 source.
/// These tests consume JSON evidence and never execute Python or use pretrained weights.</summary>
[Collection("Classical VAE")]
public sealed class ClassicalVaeReferenceTests : IDisposable
{
    private readonly int previousThreads;

    public ClassicalVaeReferenceTests()
    {
        NativeRuntimeBootstrap.Initialize();
        previousThreads = get_num_threads();
        set_num_threads(1);
    }

    [Theory]
    [InlineData("vae-encode-odd-b1", "encode")]
    [InlineData("vae-encode-odd-b2", "encode")]
    [InlineData("vae-decode-b1", "decode")]
    [InlineData("vae-decode-b2", "decode")]
    [InlineData("vae-image-encode-rgba", "imageEncode")]
    [InlineData("vae-image-encode-extra-channels", "imageEncode")]
    [InlineData("vae-image-decode-b1", "imageDecode")]
    [InlineData("vae-image-decode-b2", "imageDecode")]
    public void CompleteGraphMatchesSamePlatformSource(string identifier, string kind)
    {
        using var document = SdSamplingReferenceTests.Corpus("vae");
        Assert.Equal(1, get_num_threads());
        Assert.Equal(1, get_num_interop_threads());
        var corpus = document.RootElement;
        var config = new ClassicalVaeConfig(corpus.GetProperty("config").GetProperty("baseChannels").GetInt32());
        Assert.Equal(32, config.BaseChannels);
        using var weights = SdSyntheticInputs.CreateVae(config);
        VerifyParameters(weights, corpus.GetProperty("parameters"));

        var cases = corpus.GetProperty("cases").EnumerateArray().ToArray();
        Assert.Equal(8, cases.Length);
        Assert.Equal(cases.Length, cases.Select(c => c.GetProperty("id").GetString()).Distinct(StringComparer.Ordinal).Count());
        var reference = Assert.Single(cases, c => c.GetProperty("id").GetString() == identifier);
        Assert.Equal(kind, reference.GetProperty("kind").GetString());
        var expected = reference.GetProperty("outputs");
        Assert.Equal(3, reference.GetProperty("repeatHashes").GetArrayLength());
        foreach (var repeat in reference.GetProperty("repeatHashes").EnumerateArray())
        {
            Assert.Equal(expected.EnumerateObject().Count(), repeat.EnumerateObject().Count());
            foreach (var output in expected.EnumerateObject())
                Assert.Equal(output.Value.GetProperty("sha256").GetString(), repeat.GetProperty(output.Name).GetString());
        }

        using var scope = NewDisposeScope();
        using var model = new ClassicalVae(weights);
        using var imageVae = new ComfyImageVae(model);
        using var input = SdSamplingReferenceTests.Input(reference.GetProperty("input"));
        switch (kind)
        {
            case "encode":
                using (var moments = model.EncodeMoments(input))
                    SdSamplingReferenceTests.Compare(moments, expected.GetProperty("moments"));
                using (var latent = model.Encode(input))
                    SdSamplingReferenceTests.Compare(latent, expected.GetProperty("latent"));
                break;
            case "decode":
                using (var image = model.Decode(input))
                    SdSamplingReferenceTests.Compare(image, expected.GetProperty("image"));
                break;
            case "imageEncode":
                using (var latent = imageVae.Encode(input))
                    SdSamplingReferenceTests.Compare(latent, expected.GetProperty("latent"));
                break;
            case "imageDecode":
                using (var image = imageVae.Decode(input))
                    SdSamplingReferenceTests.Compare(image, expected.GetProperty("image"));
                break;
            default:
                throw new InvalidDataException("Unknown VAE reference operation.");
        }
    }

    private static void VerifyParameters(ClassicalVaeWeightSet weights, JsonElement references)
    {
        var schema = ClassicalVaeWeightSchema.Describe(weights.Config);
        var entries = references.EnumerateArray().ToArray();
        Assert.Equal(248, schema.Count);
        Assert.Equal(schema.Count, entries.Length);
        Assert.Equal(schema.Keys.Order(StringComparer.Ordinal),
            entries.Select(p => p.GetProperty("name").GetString()).Order(StringComparer.Ordinal));
        foreach (var entry in entries)
        {
            string name = entry.GetProperty("name").GetString()!;
            long[] shape = entry.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray();
            Assert.Equal(shape, schema[name]);
            // Inspect the actual owned bank, whose values were constructed from
            // the input-only recipe. No expected model output is computed here.
            var parameter = weights.GetTensor(name);
            Assert.Equal(shape, parameter.shape);
            Assert.Equal(ScalarType.Float32, parameter.dtype);
            float[] values = parameter.data<float>().ToArray();
            string hash = Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
            Assert.True(hash == entry.GetProperty("sha256").GetString(), $"Source parameter bytes differ: {name}.");
        }
    }

    public void Dispose() => set_num_threads(previousThreads);
}
