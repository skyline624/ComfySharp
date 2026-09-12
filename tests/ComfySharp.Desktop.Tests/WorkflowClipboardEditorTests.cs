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

public sealed class WorkflowClipboardEditorTests
{
    [AvaloniaFact]
    public async Task Canvas_shortcuts_copy_and_paste_while_widget_text_keeps_normal_clipboard_behavior()
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); SelectAll(editor);
        var window = new Window { Content = editor }; window.Show();
        try
        {
            var canvas = editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!; Assert.True(canvas.Focus());
            var modifier = ClipboardModifiers(window);
            window.KeyPressQwerty(PhysicalKey.C, modifier); window.KeyReleaseQwerty(PhysicalKey.C, modifier);
            Assert.Contains("ComfySharp.nodes", await window.Clipboard!.TryGetTextAsync());
            window.KeyPressQwerty(PhysicalKey.V, modifier); window.KeyReleaseQwerty(PhysicalKey.V, modifier);
            Assert.Equal(2, editor.Document.Nodes.Count);
            var text = editor.FindControl<TextBox>("Widgets")!; text.Text = "text only"; Assert.True(text.Focus()); text.SelectAll();
            window.KeyPressQwerty(PhysicalKey.C, modifier); window.KeyReleaseQwerty(PhysicalKey.C, modifier);
            Assert.Equal("text only", await window.Clipboard!.TryGetTextAsync()); Assert.Equal(2, editor.Document.Nodes.Count);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Platform_clipboard_transfers_selection_between_editors_and_undoes_once()
    {
        using var source = new DocumentEditor(); var a = source.AddNode("PrimitiveString"); var b = source.AddNode("PreviewAny");
        source.Document.Connect(a, 0, b, 0); source.Reload(); SelectAll(source); source.Document.MarkSaved(); string before = source.Document.ToJson();
        using var target = new DocumentEditor(); var window = new Window { Content = source }; window.Show();
        try
        {
            var clipboard = window.Clipboard!; await source.CopySelectionAsync(clipboard);
            Assert.Contains("ComfySharp.nodes", await clipboard.TryGetTextAsync());
            window.Content = target; var ticket = target.BeginSubmission(); await target.PasteSelectionAsync(clipboard, 420, 230);
            Assert.Equal(before, source.Document.ToJson()); Assert.False(source.Document.IsDirty);
            Assert.Equal(2, target.Document.Nodes.Count); Assert.Single(target.Document.Links);
            Assert.Equal(420, target.Document.Nodes.Min(n => n.X)); Assert.Equal(230, target.Document.Nodes.Min(n => n.Y));
            Assert.Equal(2, target.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!.SelectedItems!.Count);
            Assert.False(target.ApplyUiOutputs(new(), ticket)); Assert.True(PromptCompiler.Compile(target.Document).Success);
            target.Undo(); Assert.Empty(target.Document.Nodes); target.Redo(); Assert.Equal(2, target.Document.Nodes.Count);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Copy_button_does_not_replace_existing_clipboard_when_selection_is_empty()
    {
        using var editor = new DocumentEditor(); var window = new Window { Content = editor }; window.Show();
        try
        {
            await window.Clipboard!.SetTextAsync("existing text"); string? error = null; editor.Error += (_, message) => error = message;
            editor.FindControl<Button>("CopyNodes")!.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
            Assert.Contains("Select one or more", error); Assert.Equal("existing text", await window.Clipboard!.TryGetTextAsync());
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Clipboard_read_in_flight_cannot_modify_an_edited_or_disposed_document()
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); SelectAll(editor);
        var window = new Window { Content = editor }; window.Show();
        try
        {
            await editor.CopySelectionAsync(window.Clipboard!);
            var delayed = new TaskCompletionSource<string?>(); var paste = editor.PasteSelectionAsync(() => delayed.Task);
            editor.AddNode("PreviewAny"); string before = editor.Document.ToJson(); delayed.SetResult(await window.Clipboard!.TryGetTextAsync());
            await Assert.ThrowsAsync<InvalidOperationException>(() => paste); Assert.Equal(before, editor.Document.ToJson());
            delayed = new TaskCompletionSource<string?>(); paste = editor.PasteSelectionAsync(() => delayed.Task);
            editor.Dispose(); delayed.SetResult(await window.Clipboard!.TryGetTextAsync()); await Assert.ThrowsAsync<ObjectDisposedException>(() => paste);
            Assert.Equal(before, editor.Document.ToJson());
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Invalid_clipboard_has_diagnostic_and_keeps_document_and_selection()
    {
        using var editor = new DocumentEditor(); editor.AddNode("PrimitiveString"); SelectAll(editor);
        var window = new Window { Content = editor }; window.Show();
        try
        {
            string before = editor.Document.ToJson(); await window.Clipboard!.SetTextAsync("ordinary text");
            await Assert.ThrowsAnyAsync<System.Text.Json.JsonException>(() => editor.PasteSelectionAsync(window.Clipboard!));
            Assert.Equal(before, editor.Document.ToJson()); Assert.Single(editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!.SelectedItems!.Cast<NodeView>());
        }
        finally { window.Close(); }
    }

    private static void SelectAll(DocumentEditor editor)
    {
        var canvas = editor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!; canvas.SelectedItems = canvas.ItemsSource!.Cast<NodeView>().ToList();
    }
    internal static RawInputModifiers ClipboardModifiers(Window window)
    {
        var modifiers = window.GetPlatformSettings()!.HotkeyConfiguration.CommandModifiers;
        return (modifiers.HasFlag(KeyModifiers.Control) ? RawInputModifiers.Control : 0) |
            (modifiers.HasFlag(KeyModifiers.Meta) ? RawInputModifiers.Meta : 0) |
            (modifiers.HasFlag(KeyModifiers.Alt) ? RawInputModifiers.Alt : 0) |
            (modifiers.HasFlag(KeyModifiers.Shift) ? RawInputModifiers.Shift : 0);
    }
}
