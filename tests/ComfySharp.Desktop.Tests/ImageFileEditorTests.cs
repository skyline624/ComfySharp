using System.Text.Json.Nodes;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using ComfySharp.Desktop;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class ImageFileEditorTests
{
    [AvaloniaFact]
    public void Image_file_templates_expose_chainable_outputs_and_persist_filename_editing()
    {
        var editor = new DocumentEditor(); var window = new Window { Content = editor, Width = 1000, Height = 700 }; window.Show();
        try
        {
            var image = editor.AddNode("EmptyImage"); var save = editor.AddNode("SaveImage"); var preview = editor.AddNode("PreviewImage");
            Assert.Equal("ComfyUI", editor.Document.Nodes.Single(n => n.Id == save).Data["widgets_values"]![0]!.GetValue<string>());
            foreach (var id in new[] { save, preview })
            {
                var node = editor.Document.Nodes.Single(n => n.Id == id);
                var output = Assert.Single(node.Data["outputs"]!.AsArray())!;
                Assert.Equal("images", output["name"]!.GetValue<string>()); Assert.Equal("IMAGE", output["type"]!.GetValue<string>());
                Assert.Equal("images", node.Data["inputs"]![0]!["name"]!.GetValue<string>());
            }
            editor.Document.SetWidgets(save, new JsonArray("album/render"));
            editor.Document.Connect(image, 0, save, 0); editor.Document.Connect(save, 0, preview, 0); editor.Reload();
            string saved = editor.Document.ToJson(); var compiled = PromptCompiler.Compile(WorkflowDocument.Parse(saved));
            Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
            Assert.Equal("album/render", compiled.Prompt![save.Value]!["inputs"]!["filename_prefix"]!.GetValue<string>());
            Assert.True(JsonNode.DeepEquals(new JsonArray(save.Value, 0), compiled.Prompt[preview.Value]!["inputs"]!["images"]));
            editor.Undo(); Assert.Single(editor.Document.Links); editor.Redo(); Assert.Equal(saved, editor.Document.ToJson());
        }
        finally { window.Close(); }
    }
}
