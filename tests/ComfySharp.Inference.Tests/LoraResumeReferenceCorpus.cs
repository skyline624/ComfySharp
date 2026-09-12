using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace ComfySharp.Inference.Tests;

/// <summary>Independent frozen source outputs from run 34725638721. No reference generation at test time.</summary>
internal static class LoraResumeReferenceCorpus
{
    internal static JsonDocument Load(string? target = null)
    {
        target ??= SdSamplingReferenceTests.Target;
        string manifestHash = target switch
        {
            "win-x64" => "2dd7b23459c06e054ec57b3e60cf27bc697a0a20afce17f062c68973304acfe9",
            "linux-x64" => "a848e980915ef398b4f90bee2503818fafe94a4b851e7eebb5fd818de53f6b08",
            "osx-arm64" => "3e031537d22f2b66ce810c6d5025187eb59a3d4788bb28ee40761a59bc116215",
            _ => throw new PlatformNotSupportedException("No independent resume source reference exists for this platform.")
        };
        byte[] manifestBytes = ClipReferenceTests.Resource($"lora-resume-source.{target}.manifest.json");
        Assert.Equal(manifestHash, Hash(manifestBytes));
        using var manifest = JsonDocument.Parse(manifestBytes); var identity = manifest.RootElement;
        Assert.Equal("lora-resume-source-native210-cpu-f32-v1", identity.GetProperty("profile").GetString());
        Assert.Equal("8305bc1ffbb6541478789572e2313035d8c0faf5", identity.GetProperty("collectorCommit").GetString());
        Assert.Equal("34725638721", identity.GetProperty("runId").GetString());
        Assert.Equal(target, identity.GetProperty("target").GetString());
        byte[] raw;
        if (target == "win-x64") raw = ClipReferenceTests.Resource("lora-resume.reference.json");
        else
        {
            using var compressed = new MemoryStream(ClipReferenceTests.Resource($"lora-resume-source.{target}.json.gz"));
            using var gzip = new GZipStream(compressed, CompressionMode.Decompress);
            using var decoded = new MemoryStream(); gzip.CopyTo(decoded); raw = decoded.ToArray();
        }
        var artifact = identity.GetProperty("artifact");
        Assert.Equal(artifact.GetProperty("sha256").GetString(), Hash(raw));
        Assert.Equal(artifact.GetProperty("bytes").GetInt32(), raw.Length);
        var document = JsonDocument.Parse(raw);
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", document.RootElement.GetProperty("sourceCommit").GetString());
        Assert.Equal(4, document.RootElement.GetProperty("cases").GetArrayLength());
        return document;
    }
    private static string Hash(byte[] raw) => Convert.ToHexStringLower(SHA256.HashData(raw));
}
