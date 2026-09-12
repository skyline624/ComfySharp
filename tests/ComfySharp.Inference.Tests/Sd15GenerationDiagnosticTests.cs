using System.Text.Json;
using ComfySharp.RuntimeProbe;
using Xunit;
using ComfySharp.Inference;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class Sd15GenerationDiagnosticTests
{
    [Fact]
    public void Trace_requires_execution_and_an_absent_directory()
    {
        Assert.Throws<ArgumentException>(() => Sd15GenerationDiagnostic.Parse(["--checkpoint", "x", "--trace-dir", Path.GetTempPath()]));
        Assert.Null(Sd15GenerationDiagnostic.Parse(["--checkpoint", "x"]).TraceDirectory);
    }

    [Fact]
    public void Trace_preserves_view_order_and_refuses_replacement_and_invalid_values()
    {
        NativeRuntimeBootstrap.Initialize();
        string directory = Path.Combine(Path.GetTempPath(), "comfysharp-trace-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            using var scope = NewDisposeScope();
            var original = tensor(new float[] { 1, 2, 3, 4 }).reshape(2, 2);
            var view = original.transpose(0, 1);
            var record = Sd15GenerationDiagnostic.Capture(view, directory, "view");
            byte[] bytes = File.ReadAllBytes(Path.Combine(directory, record.File));
            Assert.Equal(new float[] { 1, 3, 2, 4 }, Enumerable.Range(0, 4).Select(i => BitConverter.ToSingle(bytes, i * 4)));
            Assert.Equal(new long[] { 2, 2 }, record.Shape);
            Assert.Equal(16, record.Bytes);
            Assert.Throws<IOException>(() => Sd15GenerationDiagnostic.Capture(view, directory, "view"));
            Assert.Throws<ArgumentException>(() => Sd15GenerationDiagnostic.Capture(view, directory, "../escape"));
            Assert.Throws<ArgumentException>(() => Sd15GenerationDiagnostic.Capture(tensor(float.NaN), directory, "invalid"));
            Assert.False(File.Exists(Path.Combine(directory, "invalid.f32")));
            Assert.Equal(new float[] { 1, 2, 3, 4 }, original.data<float>().ToArray());
        }
        finally { File.Delete(Path.Combine(directory, "view.f32")); Directory.Delete(directory); }
    }

    [Theory]
    [InlineData("--width", "513")]
    [InlineData("--height", "33")]
    [InlineData("--steps", "0")]
    [InlineData("--cfg", "NaN")]
    [InlineData("--cfg", "Infinity")]
    [InlineData("--seed", "-1")]
    [InlineData("--threads", "0")]
    [InlineData("--device", "cuda")]
    [InlineData("--weight-budget-mib", "0")]
    public void Invalid_options_fail_before_opening_weights(string key, string value)
    {
        using var output = new StringWriter();
        Assert.Equal(2, Sd15GenerationDiagnostic.Run(["--checkpoint", "missing.safetensors", key, value], output, TextWriter.Null));
        using var result = JsonDocument.Parse(output.ToString());
        Assert.Equal("arguments", result.RootElement.GetProperty("stage").GetString());
    }

    [Fact]
    public void Inspection_defaults_do_not_request_execution_or_output()
    {
        var options = Sd15GenerationDiagnostic.Parse(["--checkpoint", "shared.safetensors"]);
        Assert.False(options.Execute);
        Assert.False(options.ReportOutsideComponents);
        Assert.Null(options.Output);
        Assert.Equal(512, options.Width);
        Assert.Equal(20, options.Steps);
    }

    [Fact]
    public void Execute_requires_output_and_duplicate_options_are_rejected()
    {
        Assert.Throws<ArgumentException>(() => Sd15GenerationDiagnostic.Parse(["--checkpoint", "x", "--execute"]));
        Assert.Throws<ArgumentException>(() => Sd15GenerationDiagnostic.Parse(["--checkpoint", "x", "--steps", "2", "--steps", "3"]));
        Assert.Throws<ArgumentException>(() => Sd15GenerationDiagnostic.Parse(["--checkpoint", "x", "--execute", "--execute"]));
        Assert.Throws<ArgumentException>(() => Sd15GenerationDiagnostic.Parse(["--checkpoint", "x", "--report-outside-components", "--report-outside-components"]));
        Assert.True(Sd15GenerationDiagnostic.Parse(["--checkpoint", "x", "--report-outside-components"]).ReportOutsideComponents);
    }

    [Fact]
    public void Existing_output_is_preserved_before_checkpoint_access()
    {
        string destination = Path.Combine(Path.GetTempPath(), $"comfysharp-{Guid.NewGuid():N}.png");
        try
        {
            File.WriteAllText(destination, "keep");
            using var output = new StringWriter();
            Assert.Equal(2, Sd15GenerationDiagnostic.Run(["--checkpoint", "missing", "--execute", "--output", destination], output, TextWriter.Null));
            Assert.Equal("keep", File.ReadAllText(destination));
        }
        finally { File.Delete(destination); }
    }

    [Fact]
    public void Cancellation_prevents_checkpoint_access()
    {
        using var source = new CancellationTokenSource();
        source.Cancel();
        using var output = new StringWriter();
        Assert.Equal(130, Sd15GenerationDiagnostic.Run(["--checkpoint", "missing"], output, TextWriter.Null, source.Token));
    }

    [Fact]
    public void Malformed_checkpoint_is_an_inspection_failure()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "invalid weights");
            using var output = new StringWriter();
            Assert.Equal(1, Sd15GenerationDiagnostic.Run(["--checkpoint", path], output, TextWriter.Null));
            using var result = JsonDocument.Parse(output.ToString());
            Assert.Equal("inspect", result.RootElement.GetProperty("stage").GetString());
        }
        finally { File.Delete(path); }
    }
}
