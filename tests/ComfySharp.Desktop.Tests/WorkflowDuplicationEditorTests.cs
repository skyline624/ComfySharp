using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Interactivity;
using ComfySharp.Desktop;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class WorkflowDuplicationEditorTests
{
    [AvaloniaFact]
    public void Duplicate_button_copies_multiple_selected_nodes_selects_copies_and_undoes_as_one_edit()
    {
        using var editor = new DocumentEditor(); var source = editor.AddNode("PrimitiveString"); var target = editor.AddNode("PreviewAny");
        editor.Document.Connect(source, 0, target, 0); editor.Reload(); editor.Document.MarkSaved(); string before = editor.Document.ToJson();
        var canvas = editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!; canvas.SelectedItems = canvas.ItemsSource!.Cast<NodeView>().ToList();
        var ticket = editor.BeginSubmission(); editor.FindControl<Button>("DuplicateNodes")!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        Assert.Equal(4, editor.Document.Nodes.Count); Assert.Equal(2, editor.Document.Links.Count);
        Assert.Equal(2, canvas.SelectedItems!.Count); Assert.All(canvas.SelectedItems.Cast<NodeView>(), n => Assert.DoesNotContain(n.Id, new[] { source, target }));
        Assert.False(editor.ApplyUiOutputs(new(), ticket)); Assert.True(PromptCompiler.Compile(editor.Document).Success);
        editor.Undo(); Assert.Equal(before, editor.Document.ToJson()); editor.Redo(); Assert.Equal(4, editor.Document.Nodes.Count);
    }

    [AvaloniaTheory]
    [InlineData("DuplicateNodes", 1)]
    [InlineData("DuplicateWithInputs", 2)]
    public void Single_selection_buttons_respect_external_input_option(string button, int expectedLinks)
    {
        using var editor = new DocumentEditor(); var source = editor.AddNode("PrimitiveString"); var target = editor.AddNode("PreviewAny");
        editor.Document.Connect(source, 0, target, 0); editor.Reload();
        var canvas = editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!; canvas.SelectedItem = canvas.ItemsSource!.Cast<NodeView>().Single(n => n.Id == target);
        editor.FindControl<Button>(button)!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        Assert.Equal(3, editor.Document.Nodes.Count); Assert.Equal(expectedLinks, editor.Document.Links.Count); Assert.Single(canvas.SelectedItems!.Cast<NodeView>());
    }

    [AvaloniaFact]
    public void Empty_selection_has_clear_diagnostic_and_no_edit()
    {
        using var editor = new DocumentEditor(); string before = editor.Document.ToJson(); string? error = null; editor.Error += (_, message) => error = message;
        editor.FindControl<Button>("DuplicateNodes")!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        Assert.Contains("Select one or more", error); Assert.Equal(before, editor.Document.ToJson()); Assert.False(editor.Document.CanUndo);
    }
}
