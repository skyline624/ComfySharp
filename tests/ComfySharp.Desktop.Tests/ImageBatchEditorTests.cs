using System.Text.Json.Nodes;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using ComfySharp.Desktop;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class ImageBatchEditorTests
{
    [AvaloniaFact]
    public void BatchTemplateConnectsTwoImagesAndPersistsEditingWithUndo()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1100, Height = 800 };
        window.Show();
        try
        {
            var first = editor.AddNode("EmptyImage");
            var second = editor.AddNode("EmptyImage");
            var batch = editor.AddNode("ImageBatch");
            var invert = editor.AddNode("ImageInvert");
            var node = editor.Document.Nodes.Single(n => n.Id == batch);
            Assert.Equal("Batch Images (DEPRECATED)", node.Title);
            Assert.Empty(node.Data["widgets_values"]!.AsArray());
            Assert.Equal(new[] { "image1", "image2" }, node.Data["inputs"]!.AsArray().Select(p => p!["name"]!.GetValue<string>()));
            Assert.All(node.Data["inputs"]!.AsArray(), p => Assert.Equal("IMAGE", p!["type"]!.GetValue<string>()));
            var output = Assert.Single(node.Data["outputs"]!.AsArray());
            Assert.Equal("IMAGE", output!["name"]!.GetValue<string>());
            Assert.Equal("IMAGE", output["type"]!.GetValue<string>());
            editor.Document.SetWidgets(first, new JsonArray(2, 1, 1, 0xff0000));
            editor.Document.SetWidgets(second, new JsonArray(1, 3, 2, 0x0000ff));
            editor.Document.Connect(first, 0, batch, 0);
            editor.Document.Connect(second, 0, batch, 1);
            editor.Document.Connect(batch, 0, invert, 0);
            editor.Reload();
            string saved = editor.Document.ToJson();
            var reopened = WorkflowDocument.Parse(saved);
            var compiled = PromptCompiler.Compile(reopened);
            Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
            Assert.True(JsonNode.DeepEquals(new JsonArray(first.Value, 0), compiled.Prompt![batch.Value]!["inputs"]!["image1"]));
            Assert.True(JsonNode.DeepEquals(new JsonArray(second.Value, 0), compiled.Prompt[batch.Value]!["inputs"]!["image2"]));
            Assert.True(JsonNode.DeepEquals(new JsonArray(batch.Value, 0), compiled.Prompt[invert.Value]!["inputs"]!["image"]));
            Assert.Equal(3, compiled.Prompt[second.Value]!["inputs"]!["height"]!.GetValue<int>());
            Assert.Equal("Batch Images (DEPRECATED)", reopened.Nodes.Single(n => n.Id == batch).Title);
            editor.Undo();
            Assert.Equal(2, editor.Document.Links.Count);
            editor.Redo();
            Assert.Equal(saved, editor.Document.ToJson());
        }
        finally { window.Close(); }
    }
}
