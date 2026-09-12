using System.Buffers.Binary;
using System.Text.Json.Nodes;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using ComfySharp.Desktop;
using Nodify.Avalonia;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class BitmapPreviewTests
{
    private static readonly PreviewImageFile First = new("red.png", "", "output"), Second = new("blue.png", "", "temp");

    [AvaloniaFact]
    public async Task Navigation_decodes_known_pixels_and_disposes_replaced_native_bitmap()
    {
        var requests = new List<string>();
        using var preview = new NodeImagePreview([First, Second], (file, _) =>
        {
            requests.Add(file.Filename); return Task.FromResult(file == First ? PngPreviewFixture.Create() : PngPreviewFixture.Create(red: 0, blue: 255));
        });
        await preview.LoadAsync(); Assert.Equal(new PixelSize(2, 1), preview.Image!.PixelSize);
        Assert.Equal(new byte[] { 255,0,0,255,255,0,0,255 }, PngPreviewFixture.Pixels(preview.Image));
        var original = preview.Image; Assert.False(preview.CanPrevious); Assert.True(preview.CanNext);
        await preview.LoadAsync(1);
        Assert.Equal(new byte[] { 0,0,255,255,0,0,255,255 }, PngPreviewFixture.Pixels(preview.Image!));
        Assert.Throws<ObjectDisposedException>(() => PngPreviewFixture.Pixels(original));
        Assert.Equal("2 / 2", preview.Position); Assert.Equal("blue.png", preview.Filename);
        Assert.True(preview.CanPrevious); Assert.False(preview.CanNext); Assert.Equal(new[] { "red.png", "blue.png" }, requests);
        var last = preview.Image!; preview.Dispose(); Assert.Null(preview.Image);
        Assert.Throws<ObjectDisposedException>(() => PngPreviewFixture.Pixels(last));
    }

    [AvaloniaTheory]
    [InlineData(1024, 8, 512, 4)]
    [InlineData(8, 1024, 4, 512)]
    public void Large_frames_decode_to_bounded_thumbnails(int width, int height, int expectedWidth, int expectedHeight)
    {
        using var bitmap = ImagePreviewTransport.Decode(PngPreviewFixture.Create(width, height));
        Assert.Equal(new PixelSize(expectedWidth, expectedHeight), bitmap.PixelSize);
        Assert.Equal(new byte[] { 255,0,0,255 }, PngPreviewFixture.Pixels(bitmap)[..4]);
    }

    [AvaloniaFact]
    public async Task Late_response_cannot_replace_a_newer_navigation_and_disposal_cancels_loading()
    {
        var delayed = new TaskCompletionSource<byte[]>(); CancellationToken firstToken = default;
        using var preview = new NodeImagePreview([First, Second], (file, token) =>
        {
            if (file == First) { firstToken = token; return delayed.Task; }
            return Task.FromResult(PngPreviewFixture.Create(red: 0, blue: 255));
        });
        var first = preview.LoadAsync(); await preview.LoadAsync(1); Assert.True(firstToken.IsCancellationRequested);
        var current = preview.Image; delayed.SetResult(PngPreviewFixture.Create()); await first;
        Assert.Same(current, preview.Image); Assert.Equal("2 / 2", preview.Position);
        var pending = new TaskCompletionSource<byte[]>(); CancellationToken pendingToken = default;
        var disposed = new NodeImagePreview([First], (_, token) => { pendingToken = token; return pending.Task; });
        var load = disposed.LoadAsync(); disposed.Dispose(); Assert.True(pendingToken.IsCancellationRequested);
        pending.SetResult(PngPreviewFixture.Create()); await load; Assert.Null(disposed.Image);
    }

    [AvaloniaFact]
    public async Task Invalid_image_and_io_error_are_visible_without_a_bitmap()
    {
        using var preview = new NodeImagePreview([First, Second], (file, _) => file == First
            ? Task.FromResult(new byte[] { 1, 2 }) : Task.FromException<byte[]>(new IOException("missing image")));
        await preview.LoadAsync(); Assert.Null(preview.Image); Assert.Contains("Invalid PNG", preview.Status);
        await preview.LoadAsync(1); Assert.Null(preview.Image); Assert.Contains("missing image", preview.Status);
        var png = PngPreviewFixture.Create(); BinaryPrimitives.WriteUInt32BigEndian(png.AsSpan(16), uint.MaxValue);
        Assert.Throws<InvalidDataException>(() => ImagePreviewTransport.Decode(png));
    }

    [AvaloniaFact]
    public async Task Editor_shows_images_reloads_without_disposal_and_keeps_results_out_of_the_document()
    {
        using var editor = new DocumentEditor(); var id = editor.AddNode("PreviewImage"); editor.Document.MarkSaved();
        var window = new Window { Content = editor, Width = 1000, Height = 700 }; window.Show();
        try
        {
            string before = editor.Document.ToJson(); var submission = editor.BeginSubmission();
            Assert.True(await editor.ApplyUiOutputsAsync(Outputs(id.Value), submission, (_, _) => Task.FromResult(PngPreviewFixture.Create())));
            Dispatcher.UIThread.RunJobs(); var view = Views(editor).Single(); var owner = view.ImagePreview!; var bitmap = owner.Image!;
            Assert.Contains(window.GetVisualDescendants().OfType<Image>(), image => ReferenceEquals(bitmap, image.Source));
            Assert.Equal(before, editor.Document.ToJson()); Assert.False(editor.Document.IsDirty);
            editor.Reload(); Dispatcher.UIThread.RunJobs(); Assert.Same(owner, Views(editor).Single().ImagePreview); Assert.False(owner.IsDisposed);
            editor.Document.Rename(id, "edited"); Assert.True(owner.IsDisposed); Assert.Null(Views(editor).Single().ImagePreview);
            Assert.Throws<ObjectDisposedException>(() => PngPreviewFixture.Pixels(bitmap));
            Assert.False(await editor.ApplyUiOutputsAsync(Outputs(id.Value), submission, (_, _) => throw new InvalidOperationException("Stale result must not fetch")));
        }
        finally { window.Close(); }
    }

    [AvaloniaTheory]
    [InlineData("edit")]
    [InlineData("submission")]
    [InlineData("clear")]
    [InlineData("dispose")]
    public async Task Editor_invalidation_discards_inflight_frame(string action)
    {
        using var editor = new DocumentEditor(); var id = editor.AddNode("SaveImage"); var submission = editor.BeginSubmission();
        var delayed = new TaskCompletionSource<byte[]>(); CancellationToken token = default;
        var applying = editor.ApplyUiOutputsAsync(Outputs(id.Value), submission, (_, cancellation) => { token = cancellation; return delayed.Task; });
        var owner = Views(editor).Single().ImagePreview!;
        switch (action)
        {
            case "edit": editor.Document.Rename(id, "changed"); break;
            case "submission": editor.BeginSubmission(); break;
            case "clear": editor.ClearPreviews(); break;
            default: editor.Dispose(); break;
        }
        Assert.True(token.IsCancellationRequested); Assert.True(owner.IsDisposed);
        delayed.SetResult(PngPreviewFixture.Create()); Assert.False(await applying); Assert.Null(owner.Image);
    }

    [AvaloniaFact]
    public async Task Closing_the_window_releases_images_in_every_tab()
    {
        var window = new MainWindow(false); window.Show();
        var owners = new List<NodeImagePreview>();
        for (int tab = 0; tab < 2; tab++)
        {
            if (tab != 0) window.AddDocument(ComfySharp.Workflow.WorkflowDocument.Create(), null);
            var editor = window.ActiveEditor; var id = editor.AddNode("PreviewImage"); editor.Document.MarkSaved();
            Assert.True(await editor.ApplyUiOutputsAsync(Outputs(id.Value), editor.BeginSubmission(), (_, _) => Task.FromResult(PngPreviewFixture.Create())));
            owners.Add(Views(editor).Single().ImagePreview!);
        }
        window.Close(); Assert.False(window.IsVisible);
        Assert.All(owners, owner => { Assert.True(owner.IsDisposed); Assert.Null(owner.Image); });
    }

    private static JsonObject Outputs(string id) => new() { [id] = new JsonObject { ["images"] = new JsonArray(new JsonObject
        { ["filename"] = First.Filename, ["subfolder"] = First.Subfolder, ["type"] = First.Type }) } };
    private static IEnumerable<NodeView> Views(DocumentEditor editor) => editor.FindControl<NodifyEditor>("Canvas")!.ItemsSource!.Cast<NodeView>();
}
