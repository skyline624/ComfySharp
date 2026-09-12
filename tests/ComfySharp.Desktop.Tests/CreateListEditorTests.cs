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

public sealed class CreateListEditorTests
{
    [AvaloniaFact]
    public void SelectorCreatesTenUsablePortsWithoutWidgetsOrPrematureHostAvailability()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1100, Height = 850 };
        window.Show();
        try
        {
            var id = Add(editor, "CreateList");
            var node = editor.Document.Nodes.Single(n => n.Id == id);
            Assert.Equal(Enumerable.Range(0, 10).Select(i => $"inputs.input{i}"), node.Data["inputs"]!.AsArray().Select(p => p!["name"]!.GetValue<string>()));
            Assert.All(node.Data["inputs"]!.AsArray(), p => Assert.Equal("*", p!["type"]!.GetValue<string>()));
            var output = Assert.Single(node.Data["outputs"]!.AsArray())!;
            Assert.Equal("list", output["name"]!.GetValue<string>());
            Assert.True(output["is_list"]!.GetValue<bool>());
            Assert.Empty(node.Data["widgets_values"]!.AsArray());
            Assert.Contains("inputs.input0 (required)", View(editor, id).Ports);
            Assert.Contains("list (list)", View(editor, id).Ports);
            Assert.Contains("Host availability unchecked", View(editor, id).Subtitle);
            editor.SetAvailability(new HashSet<string> { "PrimitiveString" });
            Assert.Contains("Unavailable in Host", View(editor, id).Subtitle);
            editor.SetAvailability(new HashSet<string> { "CreateList" });
            Assert.Contains("Available in Host", View(editor, id).Subtitle);
            Assert.Contains(PromptCompiler.Compile(editor.Document).Diagnostics, d => d.Code == "missing_input");
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task EditorCommandsSaveReopenUndoAndCompileTheConfiguredTextListGraph()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1100, Height = 850 };
        window.Show();
        try
        {
            var errors = new List<string>();
            editor.Error += (_, text) => errors.Add(text);
            var first = Add(editor, "PrimitiveString"); Configure(editor, first, new JsonArray("alpha"));
            var second = Add(editor, "PrimitiveString"); Configure(editor, second, new JsonArray("🌍"));
            var list = Add(editor, "CreateList");
            var length = Add(editor, "StringLength");
            var preview = Add(editor, "PreviewAny");
            Connect(editor, first, list, 0);
            Connect(editor, second, list, 2); // An optional hole is intentional.
            Connect(editor, list, length, 0);
            Connect(editor, length, preview, 0);
            Assert.Empty(errors);
            Assert.Equal(4, editor.Document.Links.Count);
            string connected = editor.Document.ToJson();
            long optionalLink = editor.Document.Links.Single(l => l.Target == list && l.TargetSlot == 2).Id;
            editor.FindControl<TextBox>("LinkId")!.Text = optionalLink.ToString();
            Click(editor, "Disconnect");
            Assert.Equal(3, editor.Document.Links.Count);
            Assert.Equal(10, editor.Document.Nodes.Single(n => n.Id == list).Data["inputs"]!.AsArray().Count);
            Assert.Null(editor.Document.Nodes.Single(n => n.Id == list).Data["inputs"]![2]!["link"]);
            editor.Undo(); Assert.Equal(connected, editor.Document.ToJson());
            editor.Redo(); Assert.Equal(3, editor.Document.Links.Count);
            editor.Undo(); Assert.Equal(connected, editor.Document.ToJson());
            string? saved = null;
            await DocumentPersistence.SaveAsync(editor.Document, snapshot => { saved = snapshot; return Task.CompletedTask; });
            Assert.NotNull(saved);
            Assert.False(editor.Document.IsDirty);
            var reopened = new DocumentEditor(WorkflowDocument.Parse(saved));
            Assert.Equal(connected, reopened.Document.ToJson());
            var compiled = PromptCompiler.Compile(reopened.Document);
            Assert.True(compiled.Success, string.Join("; ", compiled.Diagnostics));
            var prompt = compiled.Prompt!;
            Assert.Equal("alpha", prompt[first.Value]!["inputs"]!["value"]!.GetValue<string>());
            Assert.Equal("🌍", prompt[second.Value]!["inputs"]!["value"]!.GetValue<string>());
            Assert.Equal(new[] { "inputs.input0", "inputs.input2" }, prompt[list.Value]!["inputs"]!.AsObject().Select(p => p.Key));
            Link(prompt, list, "inputs.input0", first);
            Link(prompt, list, "inputs.input2", second);
            Link(prompt, length, "string", list);
            Link(prompt, preview, "source", length);
            Assert.Empty(errors);
            // This is the real editor/save/compiler path. Host execution and its
            // expected preview lengths are checked in the dedicated Host tests.
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void ImportedSparseSlotsAreRenderedWithoutExpandingOrRenumberingThem()
    {
        var document = WorkflowDocument.Parse("""
            {"version":1,"nodes":[{"id":"imported","type":"CreateList","vendor":true,
             "inputs":[{"name":"inputs.input8","type":"*","link":null},{"name":"inputs.input0","type":"*","link":null}],
             "outputs":[{"name":"list","type":"*","links":[]}]}],"links":[]}
            """);
        string before = document.ToJson();
        var editor = new DocumentEditor(document);
        editor.Reload(); editor.SetAvailability(new HashSet<string> { "CreateList" });
        Assert.Equal(before, editor.Document.ToJson());
        Assert.Equal(2, editor.Document.Nodes[0].Data["inputs"]!.AsArray().Count);
        Assert.Contains("← 0: inputs.input8", View(editor, new("imported")).Ports);
        Assert.Contains("← 1: inputs.input0 (required)", View(editor, new("imported")).Ports);
        Assert.Null(editor.Document.Nodes[0].Data["widgets_values"]);
    }

    private static NodeView View(DocumentEditor editor, NodeId id) => editor.FindControl<NodifyEditor>("Canvas")!.ItemsSource!.Cast<NodeView>().Single(n => n.Id == id);
    private static NodeId Add(DocumentEditor editor, string type)
    {
        var selector = editor.FindControl<ComboBox>("NodeTypes")!;
        selector.SelectedItem = selector.ItemsSource!.Cast<NodeChoice>().Single(c => c.Type == type);
        Click(editor, "Add node");
        return editor.Document.Nodes[^1].Id;
    }
    private static void Configure(DocumentEditor editor, NodeId id, JsonArray widgets)
    {
        editor.FindControl<NodifyEditor>("Canvas")!.SelectedItem = View(editor, id);
        Click(editor, "Inspect selection");
        editor.FindControl<TextBox>("Widgets")!.Text = widgets.ToJsonString();
        Click(editor, "Apply title and widgets");
    }
    private static void Connect(DocumentEditor editor, NodeId source, NodeId target, int targetSlot)
    {
        editor.FindControl<TextBox>("SourceId")!.Text = source.Value;
        editor.FindControl<TextBox>("SourceSlot")!.Text = "0";
        editor.FindControl<TextBox>("TargetId")!.Text = target.Value;
        editor.FindControl<TextBox>("TargetSlot")!.Text = targetSlot.ToString();
        Click(editor, "Connect");
    }
    private static void Click(DocumentEditor editor, string label) => editor.GetVisualDescendants().OfType<Button>()
        .Single(button => button.Content as string == label).RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
    private static void Link(JsonObject prompt, NodeId target, string name, NodeId source) =>
        Assert.True(JsonNode.DeepEquals(new JsonArray(source.Value, 0), prompt[target.Value]!["inputs"]![name]));
}
