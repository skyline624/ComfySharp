using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.VisualTree;
using ComfySharp.Desktop;
using ComfySharp.Workflow;
using Nodify.Avalonia;
using Xunit;

[assembly: AvaloniaTestApplication(typeof(ComfySharp.Desktop.Tests.TestApplication))]
namespace ComfySharp.Desktop.Tests;
public static class TestApplication
{
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>().UseHeadless(new AvaloniaHeadlessPlatformOptions());
}
public class DesktopTests
{
    [Fact] public async Task SaveMarksOnlyTheWrittenSnapshotCleanWhenEditingContinues()
    {
        var document = WorkflowDocument.Create();
        var id = document.AddNode("PrimitiveString");
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        string? written = null;
        var saving = DocumentPersistence.SaveAsync(document, async snapshot =>
        {
            written = snapshot; started.SetResult(); await release.Task;
        });
        await started.Task;
        document.Rename(id, "Edit made while the write is pending");
        release.SetResult(); await saving;
        Assert.DoesNotContain("Edit made while the write is pending", written!);
        Assert.True(document.IsDirty);
        document.Undo();
        Assert.Equal(written, document.ToJson()); Assert.False(document.IsDirty);
    }
    [Fact] public async Task FailedSaveDoesNotAdvanceTheSavedBaseline()
    {
        var document = WorkflowDocument.Create(); document.AddNode("PrimitiveString");
        await Assert.ThrowsAsync<IOException>(() => DocumentPersistence.SaveAsync(document, _ => Task.FromException(new IOException("write failed"))));
        Assert.True(document.IsDirty); document.Undo(); Assert.False(document.IsDirty);
    }
    [AvaloniaFact] public void MainWindowStartsWithTabAndProtectsUnsavedDocument()
    {
        var window = new MainWindow(false); window.Show();
        Assert.Single(window.FindControl<TabControl>("Documents")!.Items);
        window.ActiveEditor.AddNode("SaveImage"); window.Close();
        Assert.True(window.IsVisible);
        window.ActiveEditor.Document.MarkSaved(); window.Close(); Assert.False(window.IsVisible);
    }
    [AvaloniaFact] public void NativeEditorCreatesNodesAndUndoRedoPreservesDocument()
    {
        var editor = new DocumentEditor(); var window = new Window { Content = editor, Width = 1000, Height = 700 }; window.Show();
        var id = editor.AddNode("EmptyLatentImage");
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Single(editor.Document.Nodes); Assert.Equal(id, editor.Document.Nodes[0].Id);
        Assert.True(PromptCompiler.Compile(editor.Document).Success);
        editor.Undo(); Assert.Empty(editor.Document.Nodes); editor.Redo(); Assert.Single(editor.Document.Nodes);
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        var canvas = editor.FindControl<NodifyEditor>("Canvas")!;
        var container = Assert.Single(canvas.GetVisualDescendants().OfType<ItemContainer>());
        container.Location = new Point(310, 170);
        Assert.Equal(310, editor.Document.Nodes[0].X);
        editor.SetAvailability(new HashSet<string> { "PrimitiveString" });
        var unavailable = Assert.IsType<NodeView>(Assert.Single(canvas.ItemsSource!.Cast<object>()));
        Assert.Contains("Unavailable in Host", unavailable.Subtitle);
        editor.Document.MarkSaved(); window.Close();
    }
    [AvaloniaFact] public void NativeEditorLoadsUnknownNodesWithoutDestroyingData()
    {
        var doc = WorkflowDocument.Parse("""{"version":1,"nodes":[{"id":"vendor:1","type":"VendorNode","pos":[4,5],"custom":{"keep":true}}]}""");
        var editor = new DocumentEditor(doc); var window = new Window { Content = editor, Width = 1000, Height = 700 }; window.Show();
        var canvas = editor.FindControl<NodifyEditor>("Canvas")!;
        var node = Assert.IsType<NodeView>(Assert.Single(canvas.ItemsSource!.Cast<object>()));
        Assert.Contains("Unsupported", node.Subtitle); node.Location = new Point(100, 150);
        Assert.Equal(100, doc.Nodes[0].X); Assert.True(doc.Snapshot()["nodes"]![0]!["custom"]!["keep"]!.GetValue<bool>());
        editor.Undo(); Assert.Equal(4, doc.Nodes[0].X); window.Close();
    }
    [Fact] public void HostDiscoveryFindsDevelopmentDll()
    {
        var configured = Environment.GetEnvironmentVariable("COMFYSHARP_HOST_PATH");
        if (!string.IsNullOrWhiteSpace(configured)) return;
        var directory = Path.Combine(Path.GetTempPath(), "ComfySharpDiscovery-" + Guid.NewGuid().ToString("N"));
        var host = Path.Combine(directory, "src", "ComfySharp.Host", "bin", "Debug", "net10.0", "ComfySharp.Host.dll");
        try { Directory.CreateDirectory(Path.GetDirectoryName(host)!); File.WriteAllText(host, ""); Assert.Equal(host, HostSupervisor.DiscoverHost(Path.Combine(directory, "src", "ComfySharp.Desktop"))); }
        finally { Directory.Delete(directory, true); }
    }
}
