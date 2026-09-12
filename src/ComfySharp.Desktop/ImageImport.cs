using System.Text.Json.Nodes;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;

namespace ComfySharp.Desktop;

public sealed partial class MainWindow
{
    private async void ImportImageClicked(object? sender, RoutedEventArgs e) => await RunAsync(async () =>
    {
        var selected = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Import image", AllowMultiple = false,
            FileTypeFilter = [new FilePickerFileType("Images") { Patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif", "*.bmp", "*.avif", "*.tif", "*.tiff"] }]
        });
        if (selected.Count == 0) return;
        await using var stream = await selected[0].OpenReadAsync();
        await ImportImageAsync(stream, selected[0].Name);
    });

    private async Task ImportImageAsync(Stream stream, string filename, bool replaceSingleInput = false)
    {
        var editor = ActiveEditor; var session = hostSession.Id;
        string name = await hostSession.ObserveAsync(host.UploadImageAsync(stream, filename, lifetime.Token), session);
        hostSession.Require(session);
        var inputs = editor.Document.Nodes.Where(n => n.Type == "LoadImage").ToArray();
        var id = replaceSingleInput && inputs.Length == 1 ? inputs[0].Id : editor.AddNode("LoadImage");
        editor.Document.SetWidgets(id, new JsonArray(name)); editor.Reload();
        Messages.Text = "Image imported. Connect LoadImage to VAEEncode or an image operation.";
    }
}
