using System.Text.Json.Nodes;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Interactivity;
using Avalonia.VisualTree;
using ComfySharp.Desktop;
using ComfySharp.Workflow;
using Nodify.Avalonia;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class CaseConverterEditorTests
{
    [AvaloniaFact]
    public void SelectorCreatesTextAndComboPortsWithFirstModePersisted()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1100, Height = 800 }; window.Show();
        try
        {
            var id = Add(editor, "CaseConverter");
            var node = editor.Document.Nodes.Single(n => n.Id == id);
            Assert.Equal(new[] { "string", "mode" }, node.Data["inputs"]!.AsArray().Select(p => p!["name"]!.GetValue<string>()));
            Assert.Equal(new[] { "STRING", "COMBO" }, node.Data["inputs"]!.AsArray().Select(p => p!["type"]!.GetValue<string>()));
            Assert.True(JsonNode.DeepEquals(new JsonArray("", "UPPERCASE"), node.Data["widgets_values"]));
            Assert.Equal("STRING", Assert.Single(node.Data["outputs"]!.AsArray())!["type"]!.GetValue<string>());
            editor.SetAvailability(new HashSet<string> { "CaseConverter" });
            Assert.Contains("Available in Host", View(editor, id).Subtitle);
            Assert.True(PromptCompiler.Compile(editor.Document).Success);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void EditorSavesModeAndRestoresTextWidgetAfterDisconnectAndUndo()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1100, Height = 800 }; window.Show();
        try
        {
            var errors = new List<string>(); editor.Error += (_, e) => errors.Add(e);
            var text = Add(editor, "PrimitiveString");
            var converter = Add(editor, "CaseConverter");
            editor.FindControl<NodifyEditor>("Canvas")!.SelectedItem = View(editor, converter);
            Click(editor, "Inspect selection");
            editor.FindControl<TextBox>("Widgets")!.Text = new JsonArray("saved", "Title Case").ToJsonString();
            Click(editor, "Apply title and widgets");
            editor.FindControl<TextBox>("SourceId")!.Text = text.Value;
            editor.FindControl<TextBox>("SourceSlot")!.Text = "0";
            editor.FindControl<TextBox>("TargetId")!.Text = converter.Value;
            editor.FindControl<TextBox>("TargetSlot")!.Text = "0"; Click(editor, "Connect");
            Assert.Empty(errors);
            var compiled = PromptCompiler.Compile(WorkflowDocument.Parse(editor.Document.ToJson()));
            Assert.True(compiled.Success);
            Assert.Equal("Title Case", compiled.Prompt![converter.Value]!["inputs"]!["mode"]!.GetValue<string>());
            Assert.True(JsonNode.DeepEquals(new JsonArray(text.Value, 0), compiled.Prompt[converter.Value]!["inputs"]!["string"]));
            editor.FindControl<TextBox>("LinkId")!.Text = Assert.Single(editor.Document.Links).Id.ToString(); Click(editor, "Disconnect");
            Assert.Equal("saved", PromptCompiler.Compile(editor.Document).Prompt![converter.Value]!["inputs"]!["string"]!.GetValue<string>());
            editor.Undo();
            Assert.True(JsonNode.DeepEquals(compiled.Prompt, PromptCompiler.Compile(editor.Document).Prompt));
            Assert.Empty(errors);
        }
        finally { window.Close(); }
    }

    private static NodeId Add(DocumentEditor editor, string type)
    {
        var selector = editor.FindControl<ComboBox>("NodeTypes")!;
        selector.SelectedItem = selector.ItemsSource!.Cast<NodeChoice>().Single(c => c.Type == type);
        Click(editor, "Add node"); return editor.Document.Nodes[^1].Id;
    }
    private static NodeView View(DocumentEditor editor, NodeId id) => editor.FindControl<NodifyEditor>("Canvas")!
        .ItemsSource!.Cast<NodeView>().Single(n => n.Id == id);
    private static void Click(DocumentEditor editor, string text) => editor.GetVisualDescendants().OfType<Button>()
        .Single(b => b.Content as string == text).RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
}
