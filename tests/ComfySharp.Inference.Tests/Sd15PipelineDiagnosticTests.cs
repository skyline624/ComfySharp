using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.RuntimeProbe;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class Sd15PipelineDiagnosticTests
{
    [Fact]
    public void PlanReportsReducedScopeAndBudgetRejectionBeforeNativeInitialization()
    {
        using var output = new StringWriter();
        Assert.Equal(0, Sd15PipelineDiagnostic.Run(["--case", "two-chunks-separate"], output));
        using var result = JsonDocument.Parse(output.ToString());
        var root = result.RootElement;
        Assert.Equal("plan", root.GetProperty("operation").GetString());
        Assert.False(root.GetProperty("nativeInitialized").GetBoolean());
        Assert.False(root.GetProperty("weightsGenerated").GetBoolean());
        Assert.Equal("reduced_diagnostic", root.GetProperty("configurationKind").GetString());
        Assert.Equal("not_assessed", root.GetProperty("modelCompatibility").GetString());
        Assert.Equal("not_performed", root.GetProperty("numericalQualification").GetString());
        Assert.Equal(971, root.GetProperty("parameterCount").GetInt32());
        Assert.Equal(58_137_836, root.GetProperty("parameterBytes").GetInt64());
        Assert.Equal(Sd15PipelineCases.ProtocolSha256, root.GetProperty("protocolSha256").GetString());
        output.GetStringBuilder().Clear();
        Assert.Equal(3, Sd15PipelineDiagnostic.Run(["--case", "maximum-start", "--memory-budget-mib", "1"], output));
        using var rejected = JsonDocument.Parse(output.ToString());
        Assert.Equal("budget_exceeded", rejected.RootElement.GetProperty("status").GetString());
        Assert.False(rejected.RootElement.GetProperty("nativeInitialized").GetBoolean());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    public void InvalidCommandsDoNotCreateAnOutputOrEchoPrivateArguments(int scenario)
    {
        using var directory = new OutputDirectory();
        string[] args = scenario switch
        {
            0 => ["--case", "empty-one-step", "--execute", "--memory-budget-mib", "8192", "--output", directory.Path],
            1 => ["--case", "empty-one-step", "--execute", "--synthetic-reduced", "--output", directory.Path],
            2 => ["--case", "empty-one-step", "--output", directory.Path],
            3 => ["--case", "empty-one-step", "--case", "empty-one-step"],
            4 => ["--case", "empty-one-step", "--memory-budget-mib", long.MaxValue.ToString()],
            5 => ["--case", "private-unrecognized-case"],
            _ => ["--case", "empty-one-step", "--execute", "--synthetic-reduced", "--memory-budget-mib", "8192", "--output", "relative-private-output"]
        };
        using var output = new StringWriter();
        Assert.Equal(2, Sd15PipelineDiagnostic.Run(args, output));
        Assert.False(Directory.Exists(directory.Path));
        Assert.DoesNotContain(directory.Path, output.ToString());
        Assert.DoesNotContain("private-", output.ToString());
        using var result = JsonDocument.Parse(output.ToString());
        Assert.False(result.RootElement.GetProperty("nativeInitialized").GetBoolean());
        Assert.Equal("arguments", result.RootElement.GetProperty("stage").GetString());
    }

    [Fact]
    public void PreCancellationAndInsufficientExecutionBudgetLeaveNoEvidenceDirectory()
    {
        using var directory = new OutputDirectory();
        string[] args = ["--case", "empty-one-step", "--synthetic-reduced", "--execute",
            "--memory-budget-mib", "1", "--output", directory.Path];
        using var output = new StringWriter();
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.Equal(130, Sd15PipelineDiagnostic.Run(args, output, cancelled.Token));
        Assert.False(Directory.Exists(directory.Path));
        output.GetStringBuilder().Clear();
        Assert.Equal(3, Sd15PipelineDiagnostic.Run(args, output));
        Assert.False(Directory.Exists(directory.Path));
        Assert.Contains("budget_exceeded", output.ToString());
    }

    [Fact]
    public void SnapshotSerializesAViewWithoutChangingItsLayoutOrOwnershipAndRejectsNonfiniteData()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var source = arange(30, dtype: ScalarType.Float32).reshape(2, 3, 5);
        using var view = source.narrow(2, 1, 3).permute(0, 2, 1);
        var originalShape = view.shape;
        var originalStride = view.stride();
        long originalOffset = view.storage_offset();
        using var directory = new OutputDirectory();
        Directory.CreateDirectory(directory.Path);
        var observed = Sd15PipelineDiagnostic.Observe(view, directory.Path, "view.f32", default);
        Assert.False(observed.Contiguous);
        Assert.Equal(originalShape, observed.Shape);
        Assert.Equal(originalStride, observed.Stride);
        Assert.Equal(originalOffset, observed.StorageOffset);
        Assert.Equal(1, originalOffset);
        Assert.False(view.IsInvalid);
        Assert.Equal(originalStride, view.stride());
        using var contiguous = view.contiguous();
        var expected = contiguous.bytes.ToArray();
        Assert.Equal(expected, File.ReadAllBytes(System.IO.Path.Combine(directory.Path, "view.f32")));
        Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(expected)), observed.Sha256);
        Assert.Throws<IOException>(() => Sd15PipelineDiagnostic.Observe(view, directory.Path, "view.f32", default));
        Assert.Equal(expected, File.ReadAllBytes(System.IO.Path.Combine(directory.Path, "view.f32")));
        using var invalid = full([2], double.NaN, dtype: ScalarType.Float32);
        Assert.Throws<InvalidDataException>(() => Sd15PipelineDiagnostic.Observe(invalid, directory.Path, "invalid.f32", default));
        Assert.False(File.Exists(System.IO.Path.Combine(directory.Path, "invalid.f32")));
        Assert.False(view.IsInvalid);
    }

    [Fact]
    public void ExplicitExecutionRunsTheThreeGraphsAndCommitsOnlyRepeatableCompleteEvidence()
    {
        // Share the suite's one-time inter-op configuration. The CLI normally owns a
        // fresh process; this test must not consume that one-time setting behind the
        // reference suites' back when xUnit happens to run this class first.
        NativeRuntimeBootstrap.Initialize();
        int previousThreads = get_num_threads();
        set_num_threads(1);
        try
        {
        SdReferenceRuntime.Verify();
        using var directory = new OutputDirectory();
        using var output = new StringWriter();
        string[] args = ["--case", "weighted-three-step", "--synthetic-reduced", "--execute",
            "--memory-budget-mib", "8192", "--output", directory.Path];
        Assert.Equal(0, Sd15PipelineDiagnostic.Run(args, output));
        string manifestPath = System.IO.Path.Combine(directory.Path, "manifest.json");
        byte[] manifestBytes = File.ReadAllBytes(manifestPath);
        using var manifest = JsonDocument.Parse(manifestBytes);
        var root = manifest.RootElement;
        Assert.Equal("ok", root.GetProperty("status").GetString());
        Assert.Equal("not_assessed", root.GetProperty("modelCompatibility").GetString());
        Assert.Equal("not_performed", root.GetProperty("numericalQualification").GetString());
        Assert.True(root.GetProperty("outputHashesRepeat").GetBoolean());
        Assert.True(root.GetProperty("inputHashesUnchanged").GetBoolean());
        Assert.True(root.GetProperty("parameterHashesUnchanged").GetBoolean());
        Assert.Equal(971, root.GetProperty("parametersBefore").EnumerateObject().Count());
        var executions = root.GetProperty("executions");
        Assert.Equal(3, executions.GetArrayLength());
        Assert.False(executions[0].GetProperty("boundaryObserver").GetBoolean());
        Assert.True(executions[1].GetProperty("boundaryObserver").GetBoolean());
        Assert.False(executions[2].GetProperty("boundaryObserver").GetBoolean());
        var captures = root.GetProperty("boundaryCaptures");
        Assert.Equal(8, captures.EnumerateObject().Count());
        foreach (var capture in captures.EnumerateObject())
        {
            var entry = capture.Value;
            var bytes = File.ReadAllBytes(System.IO.Path.Combine(directory.Path, entry.GetProperty("file").GetString()!));
            Assert.Equal(entry.GetProperty("bytes").GetInt64(), bytes.Length);
            Assert.Equal(entry.GetProperty("sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(bytes)));
        }
        var image = captures.GetProperty("Image");
        Assert.Equal(new[] { 1, 32, 40, 3 }, image.GetProperty("shape").EnumerateArray().Select(n => n.GetInt32()));
        Assert.False(image.GetProperty("contiguous").GetBoolean());
        Assert.Equal(3 * 32 * 40 * 4, image.GetProperty("bytes").GetInt32());
        Assert.False(File.Exists(System.IO.Path.Combine(directory.Path, "manifest.json.tmp")));
        Assert.DoesNotContain(directory.Path, System.Text.Encoding.UTF8.GetString(manifestBytes));
        output.GetStringBuilder().Clear();
        Assert.Equal(2, Sd15PipelineDiagnostic.Run(args, output));
        Assert.Equal(manifestBytes, File.ReadAllBytes(manifestPath));
        }
        finally { set_num_threads(previousThreads); }
    }

    private sealed class OutputDirectory : IDisposable
    {
        private readonly string parent = System.IO.Path.GetFullPath(System.IO.Path.GetTempPath());
        internal string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "comfysharp-pipeline-test-" + Guid.NewGuid().ToString("N"));
        public void Dispose()
        {
            string resolved = System.IO.Path.GetFullPath(Path);
            Assert.Equal(parent.TrimEnd(System.IO.Path.DirectorySeparatorChar), System.IO.Path.GetDirectoryName(resolved));
            Assert.StartsWith("comfysharp-pipeline-test-", System.IO.Path.GetFileName(resolved), StringComparison.Ordinal);
            if (Directory.Exists(resolved)) Directory.Delete(resolved, recursive: true);
        }
    }
}
