using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.VisualTree;
using ComfySharp.Desktop;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class WorkflowGroupEditorTests
{
    [AvaloniaTheory]
    [InlineData("capture")]
    [InlineData("edit")]
    [InlineData("reload")]
    public void Group_drag_aborts_if_pointer_capture_is_lost_or_document_is_changed(string reason)
    {
        using var editor = new DocumentEditor(); var node = editor.AddNode("PrimitiveString"); SelectAll(editor);
        var window = new Window { Content = editor, Width = 1000, Height = 700 }; window.Show();
        try
        {
            editor.CreateGroupFromSelection(); window.UpdateLayout(); var header = window.GetVisualDescendants().OfType<Border>().Single(b => b.Classes.Contains("group-header"));
            var start = header.TranslatePoint(new Point(20, 15), window)!.Value; string before = editor.Document.ToJson(); IPointer? pointer = null;
            header.AddHandler(InputElement.PointerPressedEvent, (_, e) => pointer = e.Pointer, RoutingStrategies.Bubble, handledEventsToo: true);
            window.MouseDown(start, MouseButton.Left); window.MouseMove(start + new Vector(50, 20), RawInputModifiers.LeftMouseButton);
            Assert.Equal(before, editor.Document.ToJson()); Assert.NotNull(pointer);
            if (reason == "edit") editor.Document.Rename(node, "Concurrent edit");
            else if (reason == "reload") editor.Reload(); else pointer.Capture(null);
            string expected = editor.Document.ToJson(); window.MouseUp(start + new Vector(50, 20), MouseButton.Left);
            Assert.Equal(expected, editor.Document.ToJson()); Assert.Equal(70, editor.Document.Nodes[0].X);
            Assert.Equal(70, Canvas(editor).ItemsSource!.Cast<NodeView>().Single().Location.X);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void Pinned_group_header_cannot_be_dragged()
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); SelectAll(editor);
        var window = new Window { Content = editor, Width = 1000, Height = 700 }; window.Show();
        try
        {
            int index = editor.CreateGroupFromSelection(); editor.Document.SetGroupPinned(index, true); editor.Reload(); window.UpdateLayout();
            var header = window.GetVisualDescendants().OfType<Border>().Single(b => b.Classes.Contains("group-header")); var start = header.TranslatePoint(new Point(20, 15), window)!.Value;
            string before = editor.Document.ToJson(); window.MouseDown(start, MouseButton.Left); window.MouseMove(start + new Vector(80, 50), RawInputModifiers.LeftMouseButton); window.MouseUp(start + new Vector(80, 50), MouseButton.Left);
            Assert.Equal(before, editor.Document.ToJson());
        }
        finally { window.Close(); }
    }
    [AvaloniaFact]
    public void Group_button_renders_frame_and_inspector_edits_and_removal_do_not_remove_nodes()
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); SelectAll(editor);
        var window = new Window { Content = editor, Width = 1000, Height = 700 }; window.Show();
        try
        {
            editor.FindControl<Button>("CreateGroup")!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
            Assert.Single(editor.Document.Groups); Assert.Single(Canvas(editor).Decorators!.Cast<GroupView>());
            window.UpdateLayout(); Assert.Single(window.GetVisualDescendants().OfType<Border>(), b => b.Classes.Contains("group-header"));
            editor.FindControl<TextBox>("GroupTitle")!.Text = "Renamed"; editor.FindControl<TextBox>("GroupColor")!.Text = "#123456";
            editor.FindControl<TextBox>("GroupWidth")!.Text = "600"; editor.FindControl<TextBox>("GroupHeight")!.Text = "400";
            editor.FindControl<Button>("ApplyGroup")!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent)); Assert.Equal("Renamed", editor.Document.Groups[0].Title);
            Assert.Equal(600, editor.Document.Groups[0].Bounds.Width);
            editor.FindControl<Button>("PinGroup")!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent)); Assert.True(editor.Document.Groups[0].Pinned);
            Assert.Contains("Pinned", ((GroupView)editor.FindControl<ComboBox>("GroupChoices")!.SelectedItem!).Caption);
            editor.FindControl<Button>("RemoveGroup")!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent)); Assert.Empty(editor.Document.Groups); Assert.Single(editor.Document.Nodes);
            editor.Undo(); Assert.Single(editor.Document.Groups);
        }
        finally { window.Close(); }
    }

    [AvaloniaTheory]
    [InlineData(1.0, false)]
    [InlineData(1.5, false)]
    [InlineData(1.5, true)]
    public void Real_pointer_group_drag_respects_zoom_and_frame_only_and_is_one_edit(double zoom, bool frameOnly)
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); SelectAll(editor);
        var window = new Window { Content = editor, Width = 1100, Height = 800 }; window.Show();
        try
        {
            editor.CreateGroupFromSelection(); var canvas = Canvas(editor); canvas.ViewportZoom = zoom; window.UpdateLayout();
            var header = window.GetVisualDescendants().OfType<Border>().Single(b => b.Classes.Contains("group-header"));
            var start = header.TranslatePoint(new Point(20, 15), window)!.Value;
            string before = editor.Document.ToJson(); var node = editor.Document.Nodes[0]; var bounds = editor.Document.Groups[0].Bounds;
            var modifiers = frameOnly ? RawInputModifiers.Shift : RawInputModifiers.None;
            window.MouseDown(start, MouseButton.Left, modifiers); window.MouseMove(start + new Vector(30 * zoom, 20 * zoom), modifiers | RawInputModifiers.LeftMouseButton);
            Assert.Equal(before, editor.Document.ToJson()); // Preview is separate from the document until release.
            window.MouseUp(start + new Vector(30 * zoom, 20 * zoom), MouseButton.Left, modifiers);
            Assert.Equal(bounds.X + 30, editor.Document.Groups[0].Bounds.X, 6); Assert.Equal(bounds.Y + 20, editor.Document.Groups[0].Bounds.Y, 6);
            Assert.Equal(node.X + (frameOnly ? 0 : 30), editor.Document.Nodes[0].X, 6);
            editor.Undo(); Assert.Equal(before, editor.Document.ToJson()); editor.Redo(); Assert.Equal(bounds.X + 30, editor.Document.Groups[0].Bounds.X, 6);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void Group_model_move_keeps_prompt_and_invalidates_previous_submission()
    {
        using var editor = new DocumentEditor(); var a = editor.AddNode("PrimitiveString"); var b = editor.AddNode("PreviewAny");
        editor.Document.Connect(a, 0, b, 0); editor.Reload(); SelectAll(editor);
        var window = new Window { Content = editor }; window.Show();
        try
        {
            int group = editor.CreateGroupFromSelection(); var before = PromptCompiler.Compile(editor.Document).Prompt; var ticket = editor.BeginSubmission();
            editor.MoveGroup(group, 50, 20); Assert.True(System.Text.Json.Nodes.JsonNode.DeepEquals(before, PromptCompiler.Compile(editor.Document).Prompt));
            Assert.False(editor.ApplyUiOutputs(new(), ticket)); Assert.Single(editor.Document.Links);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void Malformed_group_displays_diagnostic_without_discarding_document_data()
    {
        var document = WorkflowDocument.Parse("{\"version\":0.4,\"nodes\":[],\"links\":[],\"groups\":[{\"bounding\":\"extension data\"}]}");
        string before = document.ToJson(); using var editor = new DocumentEditor(document);
        Assert.Contains("preserved", editor.FindControl<TextBlock>("GroupDiagnostic")!.Text); Assert.Equal(before, document.ToJson()); Assert.False(document.CanUndo);
    }
    private static Nodify.Avalonia.NodifyEditor Canvas(DocumentEditor editor) => editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!;
    private static void SelectAll(DocumentEditor editor) { var canvas = Canvas(editor); canvas.SelectedItems = canvas.ItemsSource!.Cast<NodeView>().ToList(); }
}
