using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.RuntimeProbe;
using Xunit;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdDiagnosticTests
{
    [Theory]
    [InlineData("sd15", 3438083856L)]
    [InlineData("sd2", 3463642896L)]
    public void DefaultModeIsOnlyAStockMetadataPlan(string model, long residentBytes)
    {
        using var report = Run(["--model", model], out int code);
        Assert.Equal(0, code);
        var root = report.RootElement;
        Assert.Equal("plan", root.GetProperty("operation").GetString());
        Assert.False(root.GetProperty("nativeInitialized").GetBoolean());
        Assert.False(root.GetProperty("weightsGenerated").GetBoolean());
        Assert.Equal("not_assessed", root.GetProperty("modelCompatibility").GetString());
        Assert.Equal("stock", root.GetProperty("configurationKind").GetString());
        Assert.Equal(686, root.GetProperty("weightPlan").GetProperty("tensorCount").GetInt32());
        Assert.Equal(residentBytes, root.GetProperty("weightPlan").GetProperty("residentBytes").GetInt64());
        Assert.Equal(0, root.GetProperty("weightPlan").GetProperty("temporaryWeightPayloadBytes").GetInt64());
        Assert.True(root.GetProperty("memoryPlan").GetProperty("estimatedProcessBytes").GetInt64() > residentBytes);
    }

    [Fact]
    public void InsufficientExplicitBudgetRefusesBeforeNativeInitialization()
    {
        using var report = Run(["--model", "sd15", "--memory-budget-mib", "4096"], out int code);
        Assert.Equal(3, code);
        Assert.Equal("budget_exceeded", report.RootElement.GetProperty("status").GetString());
        Assert.False(report.RootElement.GetProperty("nativeInitialized").GetBoolean());
    }

    [Theory]
    [InlineData("--execute")]
    [InlineData("--repeat", "0")]
    [InlineData("--repeat", "4")]
    [InlineData("--chunk-elements", "0")]
    [InlineData("--chunk-elements", "262145")]
    [InlineData("--memory-budget-mib", "9223372036854775807")]
    [InlineData("--model", "sd2")]
    [InlineData("--weights", "Z:/private-model.safetensors")]
    public void MalformedOrImplicitExecutionArgumentsAreRejected(params string[] extra)
    {
        using var report = Run(new[] { "--model", "sd15" }.Concat(extra).ToArray(), out int code);
        Assert.Equal(2, code);
        Assert.False(report.RootElement.GetProperty("nativeInitialized").GetBoolean());
        Assert.DoesNotContain("private-model", report.RootElement.GetRawText());
    }

    [Fact]
    public void HelpAndPreCancelledOperationAreNativeFree()
    {
        using var help = Run(["--help"], out int code);
        Assert.Equal(0, code);
        Assert.False(help.RootElement.GetProperty("nativeInitialized").GetBoolean());
        using var cancelled = Run(["--model", "sd15"], out code, token: new(true));
        Assert.Equal(130, code);
        Assert.False(cancelled.RootElement.GetProperty("nativeInitialized").GetBoolean());
    }

    [Fact]
    public void ExplicitExecutionUsesActualReducedGraphAndWritesVerifiableArtifacts()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ComfySharp-sd-synthetic-" + Guid.NewGuid().ToString("N"));
        try
        {
            // Other native tests can leave allocator caches in this shared testhost.
            // Admit its observed baseline without changing the reduced model or the
            // production budget checks. This budget does not cause an allocation.
            using var process = Process.GetCurrentProcess();
            process.Refresh();
            long budgetMiB = checked((process.WorkingSet64 + 1024 * 1024 - 1) / (1024 * 1024) + 4096);
            string[] args = ["--model", "sd15", "--synthetic", "--execute", "--memory-budget-mib", budgetMiB.ToString(CultureInfo.InvariantCulture),
                "--output", directory, "--repeat", "2", "--chunk-elements", "4096"];
            using var report = Run(args, out int code,
                config: _ => new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false));
            Assert.True(code == 0, report.RootElement.GetRawText());
            var root = report.RootElement;
            Assert.Equal("execute", root.GetProperty("operation").GetString());
            Assert.Equal("reduced_diagnostic", root.GetProperty("configurationKind").GetString());
            Assert.Equal("not_assessed", root.GetProperty("modelCompatibility").GetString());
            Assert.True(root.GetProperty("weightsGenerated").GetBoolean());
            Assert.True(root.GetProperty("outputHashesRepeat").GetBoolean());
            Assert.DoesNotContain(directory, root.GetRawText());
            Assert.Equal(2, root.GetProperty("executions").GetArrayLength());
            using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "manifest.json")));
            Assert.Equal(686, manifest.RootElement.GetProperty("parameters").GetArrayLength());
            var records = manifest.RootElement.GetProperty("inputs").EnumerateArray()
                .Append(manifest.RootElement.GetProperty("output"));
            foreach (var record in records)
            {
                byte[] bytes = File.ReadAllBytes(Path.Combine(directory, record.GetProperty("file").GetString()!));
                Assert.Equal(record.GetProperty("bytes").GetInt32(), bytes.Length);
                Assert.Equal(record.GetProperty("sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(bytes)));
            }
            using var repeat = Run(args, out code);
            Assert.Equal(2, code); // Existing evidence is never overwritten, before stock allocation.
            Assert.False(repeat.RootElement.GetProperty("nativeInitialized").GetBoolean());
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true); }
    }

    private static JsonDocument Run(string[] args, out int code, Func<string, SdUnetConfig>? config = null,
        CancellationToken token = default)
    {
        using var output = new StringWriter();
        code = SdDiagnostic.Run(args, output, token, config);
        return JsonDocument.Parse(output.ToString());
    }
}
