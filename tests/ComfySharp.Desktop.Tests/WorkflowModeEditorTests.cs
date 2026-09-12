using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Interactivity;
using ComfySharp.Desktop;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class WorkflowModeEditorTests
{
    [AvaloniaTheory]
    [InlineData("MuteNode", 2, "Muted")]
    [InlineData("BypassNode", 4, "Bypassed")]
    public void Mode_buttons_update_document_badge_and_undo_without_changing_links(string button, int mode, string label)
    {
        var editor = new DocumentEditor(); var id = editor.AddNode("PrimitiveString"); var preview = editor.AddNode("PreviewAny");
        editor.Document.Connect(id, 0, preview, 0); editor.Reload(); editor.Document.MarkSaved(); string before = editor.Document.ToJson();
        try
        {
            var canvas = editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!;
            canvas.SelectedItem = canvas.ItemsSource!.Cast<NodeView>().Single(n => n.Id == preview);
            editor.FindControl<Button>(button)!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
            Assert.Equal(mode, editor.Document.Nodes.Single(n => n.Id == preview).Data["mode"]!.GetValue<int>());
            Assert.Contains(label, canvas.ItemsSource!.Cast<NodeView>().Single(n => n.Id == preview).Subtitle);
            Assert.Single(editor.Document.Links); Assert.True(editor.Document.IsDirty);
            editor.Undo(); Assert.Equal(before, editor.Document.ToJson()); editor.Redo();
            canvas.SelectedItem = canvas.ItemsSource!.Cast<NodeView>().Single(n => n.Id == preview);
            editor.FindControl<Button>("EnableNode")!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
            Assert.Equal(0, editor.Document.Nodes.Single(n => n.Id == preview).Data["mode"]!.GetValue<int>());
            Assert.True(PromptCompiler.Compile(editor.Document).Success);
        }
        finally { editor.Dispose(); }
    }

    [AvaloniaFact]
    public void Mode_change_invalidates_preview_and_old_submission()
    {
        using var editor = new DocumentEditor(); var preview = editor.AddNode("PreviewAny"); var ticket = editor.BeginSubmission();
        editor.SetExecutionMode(preview, 2);
        Assert.False(editor.ApplyUiOutputs(new System.Text.Json.Nodes.JsonObject(), ticket));
        Assert.Empty(PromptCompiler.Compile(editor.Document).Prompt!);
    }
}
