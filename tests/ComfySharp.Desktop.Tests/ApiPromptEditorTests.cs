using System.Text;
using System.Text.Json.Nodes;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using ComfySharp.Desktop;
using ComfySharp.Testing;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class ApiPromptEditorTests
{
    [AvaloniaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Json_and_png_prompts_import_edit_compile_and_save_without_overwriting_the_source(bool png)
    {
        const string json = """
            {"01":{"class_type":"PrimitiveString","inputs":{"value":"original"},"vendor":{"keep":true}},
             "p":{"class_type":"PreviewAny","inputs":{"source":["01",0]}}}
            """;
        byte[] bytes = png ? PngMetadataFixture.Build(("iTXt", PngMetadataFixture.Text("iTXt", "prompt", json, 1))) : Encoding.UTF8.GetBytes(json);
        var window = new MainWindow(false); window.Show();
        try
        {
            using var stream = new MemoryStream(bytes); string name = png ? "prompt.PNG" : "prompt.json";
            await window.ImportWorkflowAsync(stream, name, Path.Combine(Path.GetTempPath(), name));
            Assert.Null(window.ActiveEditor.FilePath);
            Assert.Equal(png ? "prompt.json" : "prompt.workflow.json", window.ActiveEditor.SuggestedFileName);
            Assert.Equal(2, window.FindControl<TabControl>("Documents")!.Items.Count);
            var document = window.ActiveEditor.Document;
            Assert.Equal("STRING", document.Nodes[0].Data["outputs"]![0]!["type"]!.GetValue<string>());
            Assert.True(JsonNode.DeepEquals(JsonNode.Parse(json), PromptCompiler.Compile(document).Prompt));
            document.SetWidgets(new("01"), new JsonObject { ["value"] = "edited" });
            Assert.Equal("edited", PromptCompiler.Compile(document).Prompt!["01"]!["inputs"]!["value"]!.GetValue<string>());
            window.ActiveEditor.Undo(); Assert.True(JsonNode.DeepEquals(JsonNode.Parse(json), PromptCompiler.Compile(document).Prompt));
            Assert.True(stream.CanRead); Assert.Equal(bytes, stream.ToArray());
        }
        finally { window.ActiveEditor.Document.MarkSaved(); window.Close(); }
    }

    [AvaloniaFact]
    public async Task Unknown_nodes_are_visible_and_diagnosed_without_losing_fields()
    {
        var window = new MainWindow(false); window.Show();
        try
        {
            using var stream = new MemoryStream(Encoding.UTF8.GetBytes("""
                {"vendor":{"class_type":"UnknownExtension","inputs":{"setting":{"nested":[1]}},"opaque":true}}
                """));
            await window.ImportWorkflowAsync(stream, "extension.json");
            var view = Assert.Single(window.ActiveEditor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!.ItemsSource!.Cast<NodeView>());
            Assert.Contains("Unsupported / preserved", view.Subtitle);
            Assert.Contains(PromptCompiler.Compile(window.ActiveEditor.Document).Diagnostics, d => d.Code == "unsupported_node");
        }
        finally { window.ActiveEditor.Document.MarkSaved(); window.Close(); }
    }
}
