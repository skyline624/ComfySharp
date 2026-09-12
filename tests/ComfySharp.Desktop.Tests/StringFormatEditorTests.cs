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

public sealed class StringFormatEditorTests
{
    [AvaloniaFact]
    public void SelectorCreatesNamedPortsAndAFormatWidgetWithRealHostAvailability()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1100, Height = 850 };
        window.Show();
        try
        {
            var id = Add(editor, "StringFormat");
            var node = editor.Document.Nodes.Single(n => n.Id == id);
            var slots = node.Data["inputs"]!.AsArray();
            Assert.Equal(Enumerable.Range('a', 26).Select(c => "values." + (char)c).Append("f_string"),
                slots.Select(p => p!["name"]!.GetValue<string>()));
            Assert.All(slots.Take(26), p => Assert.Equal("*", p!["type"]!.GetValue<string>()));
            Assert.Equal("STRING", slots[26]!["type"]!.GetValue<string>());
            Assert.Equal("{a}", Assert.Single(node.Data["widgets_values"]!.AsArray())!.GetValue<string>());
            Assert.Equal("STRING", Assert.Single(node.Data["outputs"]!.AsArray())!["type"]!.GetValue<string>());
            Assert.Contains("Host availability unchecked", View(editor, id).Subtitle);
            editor.SetAvailability(new HashSet<string> { "PrimitiveString" });
            Assert.Contains("Unavailable in Host", View(editor, id).Subtitle);
            editor.SetAvailability(new HashSet<string> { "StringFormat" });
            Assert.Contains("Available in Host", View(editor, id).Subtitle);
            // A missing formatting name is a body error, not an artificial required value.a port.
            Assert.True(PromptCompiler.Compile(editor.Document).Success);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void EditorCommandsPreserveSparseConnectionsAndTheConnectedFormatAcrossSaveAndUndo()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1100, Height = 850 };
        window.Show();
        try
        {
            var errors = new List<string>(); editor.Error += (_, error) => errors.Add(error);
            var value = Add(editor, "PrimitiveString"); Configure(editor, value, new("Bonjour"));
            var pattern = Add(editor, "PrimitiveString"); Configure(editor, pattern, new("{z:>10}"));
            var format = Add(editor, "StringFormat"); Configure(editor, format, new("[{z}]\nline"));
            Connect(editor, value, format, 25);
            Connect(editor, pattern, format, 26);
            Assert.Empty(errors);
            string saved = editor.Document.ToJson();
            var reopened = WorkflowDocument.Parse(saved);
            var compiled = PromptCompiler.Compile(reopened);
            Assert.True(compiled.Success);
            var inputs = compiled.Prompt![format.Value]!["inputs"]!.AsObject();
            Assert.True(JsonNode.DeepEquals(new JsonArray(value.Value, 0), inputs["values.z"]));
            Assert.True(JsonNode.DeepEquals(new JsonArray(pattern.Value, 0), inputs["f_string"]));
            Assert.False(inputs.ContainsKey("values.a"));
            Assert.Equal(saved, reopened.ToJson());
            long link = editor.Document.Links.Single(l => l.Target == format && l.TargetSlot == 26).Id;
            editor.FindControl<TextBox>("LinkId")!.Text = link.ToString(); Click(editor, "Disconnect");
            Assert.Equal("[{z}]\nline", PromptCompiler.Compile(editor.Document).Prompt![format.Value]!["inputs"]!["f_string"]!.GetValue<string>());
            editor.Document.Undo();
            Assert.True(JsonNode.DeepEquals(compiled.Prompt, PromptCompiler.Compile(editor.Document).Prompt));
            Assert.Empty(errors);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void ImportedSparseSlotsAreDisplayedWithoutExpansionOrRenumbering()
    {
        var document = WorkflowDocument.Parse("""
            {"version":1,"nodes":[{"id":1,"type":"StringFormat","vendor":"preserved","widgets_values":["constant"],
             "inputs":[{"name":"values.z","type":"*","link":null},{"name":"values.a","type":"*","link":null}],
             "outputs":[{"name":"STRING","type":"STRING","links":[]}]}],"links":[]}
            """);
        string before = document.ToJson();
        var editor = new DocumentEditor(document);
        Assert.Equal(before, editor.Document.ToJson());
        Assert.Contains("0: values.z", View(editor, new("1")).Ports);
        Assert.Contains("1: values.a", View(editor, new("1")).Ports);
        Assert.DoesNotContain("values.b", View(editor, new("1")).Ports);
        Assert.True(PromptCompiler.Compile(editor.Document).Success);
    }

    private static NodeView View(DocumentEditor editor, NodeId id) => editor.FindControl<NodifyEditor>("Canvas")!
        .ItemsSource!.Cast<NodeView>().Single(n => n.Id == id);
    private static NodeId Add(DocumentEditor editor, string type)
    {
        var selector = editor.FindControl<ComboBox>("NodeTypes")!;
        selector.SelectedItem = selector.ItemsSource!.Cast<NodeChoice>().Single(c => c.Type == type);
        Click(editor, "Add node"); return editor.Document.Nodes[^1].Id;
    }
    private static void Configure(DocumentEditor editor, NodeId id, JsonArray values)
    {
        editor.FindControl<NodifyEditor>("Canvas")!.SelectedItem = View(editor, id);
        Click(editor, "Inspect selection");
        editor.FindControl<TextBox>("Widgets")!.Text = values.ToJsonString(); Click(editor, "Apply title and widgets");
    }
    private static void Connect(DocumentEditor editor, NodeId source, NodeId target, int slot)
    {
        editor.FindControl<TextBox>("SourceId")!.Text = source.Value;
        editor.FindControl<TextBox>("SourceSlot")!.Text = "0";
        editor.FindControl<TextBox>("TargetId")!.Text = target.Value;
        editor.FindControl<TextBox>("TargetSlot")!.Text = slot.ToString(); Click(editor, "Connect");
    }
    private static void Click(DocumentEditor editor, string content) => editor.GetVisualDescendants().OfType<Button>()
        .Single(b => b.Content as string == content).RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
}
