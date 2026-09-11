using System.Security.Cryptography;
using ComfySharp.Inference;
using Xunit;

namespace ComfySharp.Inference.Tests;

public sealed class SdSourceProvenanceTests
{
    [Fact]
    public void SamePlatformSourceDocumentsArePinnedToTheIndependentLaboratory()
    {
        using var document = SdSamplingReferenceTests.ReadPinnedManifest();
        var manifest = document.RootElement;
        Assert.Equal(SdSamplingReferenceTests.Target, manifest.GetProperty("target").GetString());
        Assert.Equal("sd-components-native210-cpu-f32-v1", manifest.GetProperty("profile").GetString());
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", manifest.GetProperty("backendCommit").GetString());
        Assert.Equal(3e-5, manifest.GetProperty("comparison").GetProperty("absoluteTolerance").GetDouble());
        Assert.Equal(3e-5, manifest.GetProperty("comparison").GetProperty("relativeTolerance").GetDouble());
        Assert.True(manifest.GetProperty("syntheticWeights").GetBoolean());
        Assert.False(manifest.GetProperty("pretrainedWeightsUsed").GetBoolean());
        var runtime = manifest.GetProperty("runtime");
        Assert.Equal("3.12.10", runtime.GetProperty("python").GetString());
        Assert.Contains(runtime.GetProperty("torch").GetString(), new[] { "2.10.0", "2.10.0+cpu" });
        Assert.Equal(1, runtime.GetProperty("threads").GetInt32());
        Assert.Equal(1, runtime.GetProperty("interopThreads").GetInt32());
        Assert.NotEmpty(runtime.GetProperty("nativeLibraries").EnumerateArray());
        Assert.Equal(6, manifest.GetProperty("scripts").EnumerateObject().Count());
        Assert.Equal(new[] { "guidance", "sampling", "unet", "vae" },
            manifest.GetProperty("components").EnumerateArray().Select(c => c.GetProperty("name").GetString()).Order(StringComparer.Ordinal));
        foreach (var component in manifest.GetProperty("components").EnumerateArray())
        {
            var bytes = ClipReferenceTests.Resource($"sd-components.{SdSamplingReferenceTests.Target}.{component.GetProperty("name").GetString()}.json");
            Assert.Equal(component.GetProperty("bytes").GetInt32(), bytes.Length);
            Assert.Equal(component.GetProperty("sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(bytes)));
        }
    }
}
