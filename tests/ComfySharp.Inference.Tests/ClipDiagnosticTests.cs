using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.RuntimeProbe;
using Xunit;

namespace ComfySharp.Inference.Tests;

public sealed class ClipDiagnosticTests
{
    [Fact]
    public void InspectValidatesMetadataWithoutRequestingNativeExecutionOrPrompt()
    {
        WithFixture("l", path =>
        {
            var (code, json) = Run(Arguments(path).Concat(new[] { "--inspect" }).ToArray());
            using (json)
            {
                var report = json.RootElement;
                Assert.Equal(0, code);
                Assert.Equal("ok", report.GetProperty("status").GetString());
                Assert.Equal("inspect", report.GetProperty("operation").GetString());
                Assert.Equal("not_assessed", report.GetProperty("modelCompatibility").GetString());
                Assert.True(report.GetProperty("synthetic").GetBoolean());
                Assert.False(report.GetProperty("nativeInitialized").GetBoolean());
                Assert.Equal(8, report.GetProperty("config").GetProperty("hiddenSize").GetInt32());
                Assert.Equal(53, report.GetProperty("weightPlan").GetProperty("tensorCount").GetInt32());
                Assert.False(report.TryGetProperty("fileSha256", out _));
                Assert.False(report.TryGetProperty("executions", out _));
                Assert.DoesNotContain(path, report.GetRawText());
            }
        });
    }

    [Fact]
    public void HashingRequiresExplicitOptInAndDoesNotChangeTheFile()
    {
        WithFixture("l", path =>
        {
            string expected = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
            DateTime modified = File.GetLastWriteTimeUtc(path);
            var (code, json) = Run(Arguments(path).Concat(new[] { "--inspect", "--sha256" }).ToArray());
            using (json)
            {
                Assert.Equal(0, code);
                Assert.Equal(expected, json.RootElement.GetProperty("fileSha256").GetString());
                Assert.False(json.RootElement.GetProperty("nativeInitialized").GetBoolean());
            }
            Assert.Equal(modified, File.GetLastWriteTimeUtc(path));
            Assert.Equal(expected, Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))));
        });
    }

    [Theory]
    [InlineData("sd1-l", "l", "8,20,3,2,quick_gelu", "unprojected")]
    [InlineData("sdxl-l", "l", "8,20,3,2,quick_gelu", "projected")]
    [InlineData("sdxl-g", "g", "12,28,4,3,gelu", "projected")]
    public void EncodesActualFixtureWeightsWithRepeatedHashesAndPrivateInputSuppression(string profile, string fixture, string config, string pooledKind)
    {
        WithFixture(fixture, path =>
        {
            const string secretPrompt = "a private violet fox (inside the garden:1.3)";
            var args = new[] { "--weights", path, "--layout", "canonical", "--profile", profile,
                "--synthetic-config", config, "--text", secretPrompt, "--repeat", "2" };
            var (code, json) = Run(args);
            using (json)
            {
                var report = json.RootElement;
                Assert.Equal(0, code);
                Assert.Equal("ok", report.GetProperty("status").GetString());
                Assert.Equal("not_assessed", report.GetProperty("modelCompatibility").GetString());
                Assert.Equal(pooledKind, report.GetProperty("pooledKind").GetString());
                Assert.True(report.GetProperty("nativeInitialized").GetBoolean());
                Assert.True(report.GetProperty("outputHashesRepeat").GetBoolean());
                Assert.Equal(2, report.GetProperty("executions").GetArrayLength());
                var first = report.GetProperty("executions")[0];
                Assert.Equal("Float32", first.GetProperty("hidden").GetProperty("dtype").GetString());
                Assert.Equal(64, first.GetProperty("hidden").GetProperty("sha256").GetString()!.Length);
                Assert.Equal("little_endian_float32", first.GetProperty("hidden").GetProperty("hashEncoding").GetString());
                Assert.Equal(8, first.GetProperty("hidden").GetProperty("sample").GetArrayLength());
                Assert.Equal(77, first.GetProperty("hidden").GetProperty("shape")[1].GetInt32());
                Assert.DoesNotContain(path, report.GetRawText());
                Assert.DoesNotContain(secretPrompt, report.GetRawText());
                Assert.False(report.TryGetProperty("fileSha256", out _));
            }
        });
    }

    [Fact]
    public void ExplicitUnprojectedLSelectionIsReflectedInInspectionAndExecution()
    {
        WithFixture("l", path =>
        {
            var args = new[] { "--weights", path, "--layout", "canonical", "--profile", "sdxl-l",
                "--synthetic-config", "8,20,3,2,quick_gelu", "--text", "", "--unprojected-pooled" };
            foreach (bool inspect in new[] { false, true })
            {
                var (code, json) = Run(inspect ? args.Concat(new[] { "--inspect" }).ToArray() : args);
                using (json)
                {
                    Assert.Equal(0, code);
                    Assert.Equal("unprojected", json.RootElement.GetProperty("pooledKind").GetString());
                    Assert.False(json.RootElement.GetProperty("weightPlan").GetProperty("requireProjection").GetBoolean());
                }
            }
        });
    }

    [Fact]
    public void CheckpointShapeMismatchAndPrivatePathErrorsRemainSafeAndDistinctFromCancellation()
    {
        WithFixture("l", path =>
        {
            var (code, json) = Run(new[] { "--weights", path, "--layout", "canonical", "--profile", "sd1-l", "--inspect" });
            using (json)
            {
                Assert.Equal(2, code);
                Assert.Equal("inspect", json.RootElement.GetProperty("stage").GetString());
                Assert.False(json.RootElement.GetProperty("nativeInitialized").GetBoolean());
                Assert.DoesNotContain(path, json.RootElement.GetRawText());
            }
            (code, json) = Run(Arguments(path), new(true));
            using (json)
            {
                Assert.Equal(130, code);
                Assert.Equal("cancelled", json.RootElement.GetProperty("status").GetString());
            }
        });
        const string privatePath = "Z:/private-user/checkpoints/secret-checkpoint.safetensors";
        var (failure, error) = Run(Arguments(privatePath).Concat(new[] { "--inspect" }).ToArray());
        using (error)
        {
            Assert.Equal(2, failure);
            Assert.Equal("open", error.RootElement.GetProperty("stage").GetString());
            Assert.DoesNotContain("private-user", error.RootElement.GetRawText());
            Assert.DoesNotContain("secret-checkpoint", error.RootElement.GetRawText());
        }
    }

    [Theory]
    [InlineData("--repeat", "0")]
    [InlineData("--repeat", "101")]
    [InlineData("--repeat", "private-secret")]
    [InlineData("--synthetic-config", "3,4,2,2,gelu")]
    [InlineData("--synthetic-config", "4,4,2,2,unknown")]
    [InlineData("--unexpected-private-argument", "private-secret")]
    public void BadArgumentsHaveBoundedPrivateSafeDiagnostics(string flag, string value)
    {
        var (code, json) = Run(new[] { "--weights", "private-secret", "--layout", "canonical", "--profile", "sd1-l",
            "--text", "private-secret", flag, value });
        using (json)
        {
            Assert.Equal(2, code);
            Assert.Equal("arguments", json.RootElement.GetProperty("stage").GetString());
            Assert.DoesNotContain("private-secret", json.RootElement.GetRawText());
            Assert.DoesNotContain("unexpected-private-argument", json.RootElement.GetRawText());
        }
    }

    private static string[] Arguments(string path) => new[] { "--weights", path, "--layout", "canonical", "--profile", "sd1-l",
        "--synthetic-config", "8,20,3,2,quick_gelu" };

    private static (int Code, JsonDocument Report) Run(string[] args, CancellationToken cancellationToken = default)
    {
        using var output = new StringWriter();
        int code = ClipDiagnostic.Run(args, output, cancellationToken);
        return (code, JsonDocument.Parse(output.ToString()));
    }

    private static void WithFixture(string variant, Action<string> action)
    {
        string path = Path.GetTempFileName();
        try
        {
            using (var resource = typeof(ClipDiagnosticTests).Assembly.GetManifestResourceStream(
                $"ComfySharp.Inference.Tests.Fixtures.clip-tiny-{variant}.synthetic.safetensors"))
            using (var file = File.Create(path))
            {
                Assert.NotNull(resource);
                resource.CopyTo(file);
            }
            action(path);
        }
        finally { File.Delete(path); }
    }
}
