using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Interactivity;
using Avalonia.VisualTree;
using ComfySharp.Desktop;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class WorkflowDeletionEditorTests
{
    [AvaloniaFact]
    public async Task Canvas_uses_configured_platform_gestures_instead_of_inferring_them_from_the_os()
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); SelectAll(editor);
        var window = new Window { Content = editor }; window.Show();
        var gesture = new KeyGesture(Key.C, KeyModifiers.Control | KeyModifiers.Shift);
        var copy = window.GetPlatformSettings()!.HotkeyConfiguration.Copy; copy.Add(gesture);
        try
        {
            await window.Clipboard!.SetTextAsync("previous"); editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!.Focus();
            window.KeyPressQwerty(PhysicalKey.C, RawInputModifiers.Control | RawInputModifiers.Shift);
            window.KeyReleaseQwerty(PhysicalKey.C, RawInputModifiers.Control | RawInputModifiers.Shift);
            Assert.Contains("ComfySharp.nodes", await window.Clipboard!.TryGetTextAsync()); Assert.Single(editor.Document.Nodes);
        }
        finally { copy.Remove(gesture); window.Close(); }
    }
    [AvaloniaFact]
    public void Delete_button_removes_multiselection_in_one_undo_and_invalidates_previews()
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); editor.AddNode("PreviewAny"); SelectAll(editor);
        string before = editor.Document.ToJson(); var ticket = editor.BeginSubmission();
        editor.FindControl<Button>("DeleteNodes")!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        Assert.Empty(editor.Document.Nodes); Assert.False(editor.ApplyUiOutputs(new(), ticket));
        editor.Undo(); Assert.Equal(before, editor.Document.ToJson()); editor.Redo(); Assert.Empty(editor.Document.Nodes);
    }

    [AvaloniaFact]
    public async Task Cut_writes_to_platform_clipboard_and_pastes_into_a_second_editor()
    {
        using var source = new DocumentEditor(); var a = source.AddNode("PrimitiveString"); var b = source.AddNode("PreviewAny");
        source.Document.Connect(a, 0, b, 0); source.Reload(); SelectAll(source); string before = source.Document.ToJson();
        using var target = new DocumentEditor(); var window = new Window { Content = source }; window.Show();
        try
        {
            await source.CutSelectionAsync(window.Clipboard!); Assert.Empty(source.Document.Nodes);
            window.Content = target; await target.PasteSelectionAsync(window.Clipboard!); Assert.Equal(2, target.Document.Nodes.Count); Assert.Single(target.Document.Links);
            source.Undo(); Assert.Equal(before, source.Document.ToJson()); Assert.True(PromptCompiler.Compile(target.Document).Success);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Failed_clipboard_write_keeps_document_selection_undo_and_previews()
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); SelectAll(editor); editor.Document.MarkSaved();
        string before = editor.Document.ToJson(); var ticket = editor.BeginSubmission();
        await Assert.ThrowsAsync<IOException>(() => editor.CutSelectionAsync(_ => Task.FromException(new IOException("Clipboard busy"))));
        Assert.Equal(before, editor.Document.ToJson()); Assert.False(editor.Document.IsDirty); Assert.True(editor.ApplyUiOutputs(new(), ticket));
        Assert.Single(editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!.SelectedItems!.Cast<NodeView>());
    }

    [AvaloniaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Delayed_clipboard_write_cannot_remove_nodes_after_edit_or_disposal(bool dispose)
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); SelectAll(editor); var written = new TaskCompletionSource();
        string? text = null; var cutting = editor.CutSelectionAsync(value => { text = value; return written.Task; });
        Assert.Contains("ComfySharp.nodes", text); Assert.Single(editor.Document.Nodes);
        if (dispose) editor.Dispose(); else editor.AddNode("PreviewAny"); string before = editor.Document.ToJson(); written.SetResult();
        if (dispose) await Assert.ThrowsAsync<ObjectDisposedException>(() => cutting); else await Assert.ThrowsAsync<InvalidOperationException>(() => cutting);
        Assert.Equal(before, editor.Document.ToJson());
    }

    [AvaloniaFact]
    public async Task Failed_preflight_never_writes_clipboard_and_cut_uses_captured_selection()
    {
        var doc = WorkflowDocument.Parse("""{"version":0.4,"nodes":[{"id":1,"type":"Unknown","clonable":false}],"links":[]}""");
        using var restricted = new DocumentEditor(doc); SelectAll(restricted); bool wrote = false;
        await Assert.ThrowsAsync<InvalidOperationException>(() => restricted.CutSelectionAsync(_ => { wrote = true; return Task.CompletedTask; })); Assert.False(wrote);
        using var editor = new DocumentEditor(); var first = editor.AddNode("PrimitiveString"); var second = editor.AddNode("PreviewAny");
        var canvas = editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!; canvas.SelectedItem = canvas.ItemsSource!.Cast<NodeView>().Single(n => n.Id == first);
        var completion = new TaskCompletionSource(); var cut = editor.CutSelectionAsync(_ => completion.Task);
        canvas.SelectedItems = canvas.ItemsSource!.Cast<NodeView>().Where(n => n.Id == second).ToList(); completion.SetResult(); await cut;
        Assert.Equal(second, Assert.Single(editor.Document.Nodes).Id);
    }

    [AvaloniaFact]
    public async Task Canvas_cut_delete_and_backspace_shortcuts_preserve_text_editing()
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); SelectAll(editor);
        var window = new Window { Content = editor }; window.Show();
        try
        {
            var canvas = editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!; Assert.True(canvas.Focus());
            var modifier = WorkflowClipboardEditorTests.ClipboardModifiers(window);
            window.KeyPressQwerty(PhysicalKey.X, modifier); window.KeyReleaseQwerty(PhysicalKey.X, modifier);
            Assert.Empty(editor.Document.Nodes); Assert.Contains("ComfySharp.nodes", await window.Clipboard!.TryGetTextAsync());
            foreach (var key in new[] { PhysicalKey.Delete, PhysicalKey.Backspace })
            {
                editor.Undo(); SelectAll(editor); canvas.Focus(); window.KeyPressQwerty(key, RawInputModifiers.None); window.KeyReleaseQwerty(key, RawInputModifiers.None);
                Assert.Empty(editor.Document.Nodes);
            }
            editor.Undo(); var text = editor.FindControl<TextBox>("Widgets")!; text.Text = "widget text"; text.Focus(); text.SelectAll();
            window.KeyPressQwerty(PhysicalKey.X, modifier); window.KeyReleaseQwerty(PhysicalKey.X, modifier);
            Assert.Equal("", text.Text); Assert.Single(editor.Document.Nodes); Assert.Equal("widget text", await window.Clipboard!.TryGetTextAsync());
        }
        finally { window.Close(); }
    }

    private static void SelectAll(DocumentEditor editor)
    {
        var canvas = editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!; canvas.SelectedItems = canvas.ItemsSource!.Cast<NodeView>().ToList();
    }
}
