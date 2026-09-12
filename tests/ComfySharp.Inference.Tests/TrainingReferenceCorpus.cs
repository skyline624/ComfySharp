using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace ComfySharp.Inference.Tests;

/// <summary>Immutable same-platform source oracles collected independently in CI 34711337834.
/// No Python or runtime reference generation is used by these distributed .NET tests.</summary>
internal static class TrainingReferenceCorpus
{
    internal static JsonDocument Load(string component, string? target = null)
    {
        target ??= SdSamplingReferenceTests.Target;
        string original = component switch
        {
            "batches" => "training-batches.reference.json",
            "denoising" => "lora-denoising.reference.json",
            "adapters" => "training-adapters.reference.json",
            _ => throw new ArgumentOutOfRangeException(nameof(component))
        };
        string manifestHash = target switch
        {
            "win-x64" => "9361d3f035e1b4767b35ad6ae84e6de906497c5a5df13f226783f1f8d9c86660",
            "linux-x64" => "c5a8d10ae3a6dbb89cdd53f269077112bbca71393ad73e824cb061105951a98a",
            "osx-arm64" => "1af19a5d28a758b974466432086bfa5a6595e109b1eafa3eee52326d87eb3709",
            _ => throw new PlatformNotSupportedException("No training source oracle was collected for this platform.")
        };
        byte[] manifestBytes = ClipReferenceTests.Resource($"training-source.{target}.manifest.json");
        Assert.Equal(manifestHash, Hash(manifestBytes));
        using var manifest = JsonDocument.Parse(manifestBytes); var identity = manifest.RootElement;
        Assert.Equal("training-platform-source-native210-cpu-f32-v1", identity.GetProperty("profile").GetString());
        Assert.Equal(target, identity.GetProperty("target").GetString());
        Assert.Equal("cd4d0348642984393aed211cc1f8d781eea7b5be", identity.GetProperty("collectorCommit").GetString());
        Assert.Equal("34711337834", identity.GetProperty("runId").GetString());
        byte[] bytes;
        if (target == "win-x64") bytes = ClipReferenceTests.Resource(original);
        else
        {
            using var compressed = new MemoryStream(ClipReferenceTests.Resource($"training-source.{target}.{component}.json.gz"));
            using var gzip = new GZipStream(compressed, CompressionMode.Decompress);
            using var decoded = new MemoryStream(); gzip.CopyTo(decoded); bytes = decoded.ToArray();
        }
        var artifact = identity.GetProperty("artifacts").GetProperty(component);
        Assert.Equal(artifact.GetProperty("bytes").GetInt32(), bytes.Length);
        Assert.Equal(artifact.GetProperty("sha256").GetString(), Hash(bytes));
        var document = JsonDocument.Parse(bytes);
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", document.RootElement.GetProperty("sourceCommit").GetString());
        return document;
    }
    private static string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
}
