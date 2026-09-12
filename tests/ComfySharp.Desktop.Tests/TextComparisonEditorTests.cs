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

public sealed class TextComparisonEditorTests
{
    [AvaloniaFact]
    public void SelectorsCreateTextAndBooleanPortsWithSourceWidgetDefaults()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1100, Height = 800 }; window.Show();
        try
        {
            foreach (string type in new[] { "StringContains", "StringCompare" })
            {
                var id = Add(editor, type);
                var node = editor.Document.Nodes.Single(n => n.Id == id);
                bool contains = type == "StringContains";
                var names = contains ? new[] { "string", "substring", "case_sensitive" } : ["string_a", "string_b", "mode", "case_sensitive"];
                Assert.Equal(names, node.Data["inputs"]!.AsArray().Select(p => p!["name"]!.GetValue<string>()));
                Assert.True(node.Data["widgets_values"]!.AsArray()[^1]!.GetValue<bool>());
                if (!contains) Assert.Equal("Starts With", node.Data["widgets_values"]![2]!.GetValue<string>());
                Assert.Equal("BOOLEAN", Assert.Single(node.Data["outputs"]!.AsArray())!["type"]!.GetValue<string>());
                editor.SetAvailability(new HashSet<string> { type });
                Assert.Contains("Available in Host", View(editor, node.Id).Subtitle);
            }
            Assert.True(PromptCompiler.Compile(editor.Document).Success);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void EditorConnectsSensitivityAndPreservesItsWidgetAcrossSaveDisconnectAndUndo()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1100, Height = 800 }; window.Show();
        try
        {
            var errors = new List<string>(); editor.Error += (_, e) => errors.Add(e);
            var flag = Add(editor, "PrimitiveBoolean");
            var compare = Add(editor, "StringCompare");
            editor.FindControl<NodifyEditor>("Canvas")!.SelectedItem = View(editor, compare);
            Click(editor, "Inspect selection");
            editor.FindControl<TextBox>("Widgets")!.Text = new JsonArray("İ", "i\u0307", "Equal", true).ToJsonString();
            Click(editor, "Apply title and widgets");
            editor.FindControl<TextBox>("SourceId")!.Text = flag.Value;
            editor.FindControl<TextBox>("SourceSlot")!.Text = "0";
            editor.FindControl<TextBox>("TargetId")!.Text = compare.Value;
            editor.FindControl<TextBox>("TargetSlot")!.Text = "3"; Click(editor, "Connect");
            Assert.Empty(errors);
            var compiled = PromptCompiler.Compile(WorkflowDocument.Parse(editor.Document.ToJson()));
            Assert.True(compiled.Success);
            Assert.True(JsonNode.DeepEquals(new JsonArray(flag.Value, 0), compiled.Prompt![compare.Value]!["inputs"]!["case_sensitive"]));
            editor.FindControl<TextBox>("LinkId")!.Text = Assert.Single(editor.Document.Links).Id.ToString(); Click(editor, "Disconnect");
            Assert.True(PromptCompiler.Compile(editor.Document).Prompt![compare.Value]!["inputs"]!["case_sensitive"]!.GetValue<bool>());
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
