using System.Text;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using ComfySharp.Desktop;
using ComfySharp.Testing;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class ImportJsonEditorTests
{
    [AvaloniaTheory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task Lossy_imports_display_warning_and_never_automatically_overwrite_source(bool png, bool api)
    {
        string json = api ? """{"p":{"class_type":"PreviewAny","inputs":{"source":NaN}}}""" :
            """{"version":0.4,"nodes":[],"unknown":[NaN,"NaN",Infinity]}""";
        byte[] bytes = png ? PngMetadataFixture.Build(("tEXt", PngMetadataFixture.Text("tEXt", api ? "prompt" : "workflow", json))) : Encoding.UTF8.GetBytes(json);
        var window = new MainWindow(false); window.Show();
        try
        {
            using var stream = new MemoryStream(bytes); string filename = png ? "original.png" : "original.json";
            await window.ImportWorkflowAsync(stream, filename, Path.Combine(Path.GetTempPath(), filename));
            Assert.Null(window.ActiveEditor.FilePath); Assert.NotNull(window.ActiveEditor.SuggestedFileName);
            Assert.Contains(ImportJson.NonFiniteWarning, window.FindControl<TextBox>("Messages")!.Text);
            if (api) Assert.Null(PromptCompiler.Compile(window.ActiveEditor.Document).Prompt!["p"]!["inputs"]!["source"]);
            else { Assert.Null(window.ActiveEditor.Document.Snapshot()["unknown"]![0]); Assert.Equal("NaN", window.ActiveEditor.Document.Snapshot()["unknown"]![1]!.GetValue<string>()); }
            Assert.Equal(bytes, stream.ToArray()); Assert.True(stream.CanRead);
        }
        finally { window.ActiveEditor.Document.MarkSaved(); window.Close(); }
    }

    [AvaloniaFact]
    public async Task Png_graph_failure_opens_valid_prompt_in_new_tab_and_displays_fallback_reason()
    {
        var window = new MainWindow(false); window.Show(); var previous = window.ActiveEditor;
        try
        {
            using var stream = new MemoryStream(PngMetadataFixture.Build(
                ("tEXt", PngMetadataFixture.Text("tEXt", "workflow", "invalid")),
                ("tEXt", PngMetadataFixture.Text("tEXt", "prompt", """{"p":{"class_type":"PreviewAny","inputs":{"source":"hello"}}}"""))));
            await window.ImportWorkflowAsync(stream, "fallback.png");
            Assert.NotSame(previous, window.ActiveEditor); Assert.Empty(previous.Document.Nodes);
            Assert.Contains("trying API prompt", window.FindControl<TextBox>("Messages")!.Text);
        }
        finally { window.Close(); }
    }
}
