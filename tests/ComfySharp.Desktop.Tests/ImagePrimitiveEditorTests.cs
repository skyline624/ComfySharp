using System.Text.Json.Nodes;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using ComfySharp.Desktop;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class ImagePrimitiveEditorTests
{
    [AvaloniaFact]
    public void ImageTemplatesExposeTheNativeChainAndPersistWidgets()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1100, Height = 800 }; window.Show();
        try
        {
            var empty = editor.AddNode("EmptyImage");
            var repeat = editor.AddNode("RepeatImageBatch");
            var extract = editor.AddNode("ImageFromBatch");
            var invert = editor.AddNode("ImageInvert");
            var source = editor.Document.Nodes.Single(n => n.Id == empty);
            Assert.True(JsonNode.DeepEquals(new JsonArray(512, 512, 1, 0), source.Data["widgets_values"]));
            Assert.Equal(new[] { "width", "height", "batch_size", "color" }, source.Data["inputs"]!.AsArray().Select(i => i!["name"]!.GetValue<string>()));
            Assert.All(editor.Document.Nodes, n => Assert.Equal("IMAGE", Assert.Single(n.Data["outputs"]!.AsArray())!["type"]!.GetValue<string>()));
            editor.Document.SetWidgets(empty, new JsonArray(2, 1, 1, 0xff0000));
            editor.Document.SetWidgets(repeat, new JsonArray(3));
            editor.Document.SetWidgets(extract, new JsonArray(-1, 1));
            editor.Document.Connect(empty, 0, repeat, 0);
            editor.Document.Connect(repeat, 0, extract, 0);
            editor.Document.Connect(extract, 0, invert, 0);
            editor.Reload();
            var compiled = PromptCompiler.Compile(WorkflowDocument.Parse(editor.Document.ToJson()));
            Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
            Assert.Equal(-1, compiled.Prompt![extract.Value]!["inputs"]!["batch_index"]!.GetValue<int>());
            Assert.True(JsonNode.DeepEquals(new JsonArray(extract.Value, 0), compiled.Prompt[invert.Value]!["inputs"]!["image"]));
            editor.Undo(); Assert.Equal(2, editor.Document.Links.Count);
            editor.Redo(); Assert.Equal(3, editor.Document.Links.Count);
        }
        finally { window.Close(); }
    }
}
