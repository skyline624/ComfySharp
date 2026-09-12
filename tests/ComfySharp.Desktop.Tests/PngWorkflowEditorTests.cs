using System.Text.Json.Nodes;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using ComfySharp.Desktop;
using ComfySharp.Testing;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class PngWorkflowEditorTests
{
    [AvaloniaTheory]
    [InlineData(0.4)]
    [InlineData(1.0)]
    public async Task Import_preserves_document_and_offers_json_save_without_targeting_the_png(double version)
    {
        var window = new MainWindow(false); window.Show();
        try
        {
            var workflow = new JsonObject { ["version"] = version, ["nodes"] = new JsonArray(new JsonObject
                { ["id"] = "vendor:7", ["type"] = "Unported", ["unknown"] = new JsonArray("猫", 1, null) }), ["extra"] = new JsonObject { ["keep"] = true } };
            byte[] png = PngMetadataFixture.Build(("iTXt", PngMetadataFixture.Text("iTXt", "workflow", workflow.ToJsonString(), 1)));
            using var stream = new MemoryStream(png);
            await window.ImportWorkflowAsync(stream, "export.PNG", Path.Combine(Path.GetTempPath(), "export.PNG"));
            Assert.Equal(2, window.FindControl<TabControl>("Documents")!.Items.Count);
            Assert.True(JsonNode.DeepEquals(workflow, window.ActiveEditor.Document.Snapshot()));
            Assert.Null(window.ActiveEditor.FilePath); Assert.Equal("export.json", window.ActiveEditor.SuggestedFileName);
            Assert.Equal(png, stream.ToArray()); Assert.True(stream.CanRead);
            window.ActiveEditor.Document.Rename(new("vendor:7"), "edited");
            Assert.True(window.ActiveEditor.Document.IsDirty); window.ActiveEditor.Undo();
            Assert.True(JsonNode.DeepEquals(workflow, window.ActiveEditor.Document.Snapshot()));
        }
        finally { window.ActiveEditor.Document.MarkSaved(); window.Close(); }
    }

    [AvaloniaFact]
    public async Task Failed_import_preserves_existing_tab_and_graph()
    {
        var window = new MainWindow(false); window.Show(); window.ActiveEditor.AddNode("PrimitiveString");
        var original = window.ActiveEditor; string before = original.Document.ToJson();
        try
        {
            using var stream = new MemoryStream(PngMetadataFixture.Build(("tEXt", PngMetadataFixture.Text("tEXt", "prompt", "{}"))));
            await Assert.ThrowsAsync<NotSupportedException>(() => window.ImportWorkflowAsync(stream, "prompt-only.png"));
            Assert.Same(original, window.ActiveEditor); Assert.Equal(before, original.Document.ToJson());
            Assert.Single(window.FindControl<TabControl>("Documents")!.Items);
        }
        finally { original.Document.MarkSaved(); window.Close(); }
    }
}
