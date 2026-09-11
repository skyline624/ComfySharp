using System.Text.Json.Nodes;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using ComfySharp.Workflow;

namespace ComfySharp.Desktop;
public sealed partial class MainWindow : Window
{
    private readonly HostSupervisor host = new();
    private readonly string clientId = Guid.NewGuid().ToString();
    private string? lastPromptId;
    private HashSet<string>? availableNodes;
    public MainWindow() : this(true) { }
    public MainWindow(bool startHost)
    {
        InitializeComponent(); AddDocument(WorkflowDocument.Create(), null);
        host.Changed += (_, _) => Dispatcher.UIThread.Post(() =>
        {
            HostStatus.Text = host.Status;
            if (!host.Ready) { availableNodes = null; foreach (var tab in Documents.Items.OfType<TabItem>()) ((DocumentEditor)tab.Content!).SetAvailability(null); }
        });
        Opened += async (_, _) =>
        {
            if (startHost) await RunAsync(StartHostAsync);
            if (Program.SmokeTest)
            {
                try { await SmokeAsync(); Environment.ExitCode = 0; }
                catch (Exception error) { Messages.Text = error.Message; Console.Error.WriteLine(error); Environment.ExitCode = 1; }
                Close();
            }
        };
        Closing += (_, e) =>
        {
            if (!Program.SmokeTest && Documents.Items.OfType<TabItem>().Any(t => ((DocumentEditor)t.Content!).Document.IsDirty))
            { e.Cancel = true; Messages.Text = "Save all modified tabs before closing. Documents remain open."; }
        };
        Closed += (_, _) => host.Dispose();
    }
    public DocumentEditor ActiveEditor => (DocumentEditor)((TabItem)Documents.SelectedItem!).Content!;
    public void AddDocument(WorkflowDocument document, string? path)
    {
        var editor = new DocumentEditor(document) { FilePath = path };
        editor.SetAvailability(availableNodes);
        var tab = new TabItem { Content = editor };
        void Update() => tab.Header = (Path.GetFileName(editor.FilePath) is { Length: > 0 } name ? name : "Untitled") + (document.IsDirty ? " *" : "");
        document.Changed += (_, _) => Update(); Update();
        editor.Error += (_, message) => Messages.Text = message;
        Documents.Items.Add(tab); Documents.SelectedItem = tab;
    }
    private async Task RunAsync(Func<Task> action) { try { await action(); } catch (Exception error) { Messages.Text = error.Message; } }
    private async Task StartHostAsync()
    {
        availableNodes = null; lastPromptId = null;
        await host.StartAsync();
        availableNodes = (await host.GetAsync("/object_info")).Select(p => p.Key).ToHashSet(StringComparer.Ordinal);
        foreach (var tab in Documents.Items.OfType<TabItem>()) ((DocumentEditor)tab.Content!).SetAvailability(availableNodes);
        Messages.Text = $"Host provides {availableNodes.Count} node types. Queue validates the complete document before submission.";
    }
    private async Task SmokeAsync()
    {
        var id = ActiveEditor.AddNode("PrimitiveString");
        var accepted = await host.SubmitAsync(Compile(true), clientId, [id.Value]);
        var jobId = accepted["prompt_id"]!.GetValue<string>();
        for (var attempt = 0; attempt < 50; attempt++)
        {
            var history = await host.GetAsync("/history");
            if (history[jobId] is JsonObject entry)
            {
                if (entry["status"]?["completed"]?.GetValue<bool>() != true || entry["outputs"]?[id.Value]?[0]?[0]?.GetValue<string>() != "Hello from ComfySharp")
                    throw new InvalidOperationException("Scalar smoke execution failed: " + entry.ToJsonString());
                Console.WriteLine("ComfySharp Desktop smoke passed: native window, supervised Host, compiled prompt and verified scalar result.");
                return;
            }
            await Task.Delay(100);
        }
        throw new TimeoutException("Scalar smoke job did not complete.");
    }
    private void NewClicked(object? sender, RoutedEventArgs e) => AddDocument(WorkflowDocument.Create(), null);
    private async void OpenClicked(object? sender, RoutedEventArgs e) => await RunAsync(async () =>
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { Title = "Open workflow", AllowMultiple = true, FileTypeFilter = [JsonType] });
        foreach (var file in files) { await using var stream = await file.OpenReadAsync(); using var reader = new StreamReader(stream); AddDocument(WorkflowDocument.Parse(await reader.ReadToEndAsync()), file.TryGetLocalPath()); }
    });
    private static FilePickerFileType JsonType { get; } = new("Workflow JSON") { Patterns = ["*.json"] };
    private async Task SaveAsync(bool choosePath)
    {
        var editor = ActiveEditor;
        if (choosePath || editor.FilePath is null)
        {
            var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions { Title = "Save workflow", SuggestedFileName = "workflow.json", DefaultExtension = "json", FileTypeChoices = [JsonType] });
            if (file is null) return;
            await DocumentPersistence.SaveAsync(editor.Document, async snapshot =>
            {
                await using (var stream = await file.OpenWriteAsync()) { stream.SetLength(0); await using var writer = new StreamWriter(stream); await writer.WriteAsync(snapshot); }
                editor.FilePath = file.TryGetLocalPath();
            });
        }
        else
        {
            // Replace only after a complete write, leaving the previous workflow intact on write failure.
            var destination = editor.FilePath;
            await DocumentPersistence.SaveAsync(editor.Document, async snapshot =>
            {
                var temporary = destination + "." + Guid.NewGuid().ToString("N") + ".tmp";
                try { await File.WriteAllTextAsync(temporary, snapshot); File.Move(temporary, destination, true); }
                finally { if (File.Exists(temporary)) File.Delete(temporary); }
            });
        }
        Messages.Text = editor.Document.IsDirty ? "Snapshot saved. Newer edits remain unsaved." : "Workflow saved.";
    }
    private async void SaveClicked(object? sender, RoutedEventArgs e) => await RunAsync(() => SaveAsync(false));
    private async void SaveAsClicked(object? sender, RoutedEventArgs e) => await RunAsync(() => SaveAsync(true));
    private void UndoClicked(object? sender, RoutedEventArgs e) => ActiveEditor.Undo();
    private void RedoClicked(object? sender, RoutedEventArgs e) => ActiveEditor.Redo();
    private JsonObject Compile(bool requireHost)
    {
        if (requireHost && availableNodes is null) throw new InvalidOperationException("Start the Host before queueing.");
        var result = PromptCompiler.Compile(ActiveEditor.Document, availableNodes: requireHost ? availableNodes : null);
        if (!result.Success) throw new InvalidOperationException(string.Join(Environment.NewLine, result.Diagnostics.Select(d => $"{d.Node}: {d.Code} — {d.Message}")));
        return result.Prompt!;
    }
    private async void ExportClicked(object? sender, RoutedEventArgs e) => await RunAsync(async () =>
    {
        var prompt = Compile(false);
        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions { SuggestedFileName = "prompt.json", FileTypeChoices = [JsonType] });
        if (file is null) return;
        await using var stream = await file.OpenWriteAsync(); stream.SetLength(0); await using var writer = new StreamWriter(stream); await writer.WriteAsync(prompt.ToJsonString(new() { WriteIndented = true }));
        Messages.Text = "Prompt exported; backend availability is checked when queueing.";
    });
    private async void QueueClicked(object? sender, RoutedEventArgs e) => await RunAsync(async () =>
    {
        var prompt = Compile(true);
        var sources = ActiveEditor.Document.Links.Select(l => l.Source.Value).ToHashSet();
        var targets = prompt.Select(p => p.Key).Where(id => !sources.Contains(id)).ToArray();
        if (targets.Length == 0) throw new InvalidOperationException("Add a node before queueing.");
        var result = await host.SubmitAsync(prompt, clientId, targets); lastPromptId = result["prompt_id"]?.GetValue<string>();
        Messages.Text = $"Submitted {lastPromptId}. Use Queue / history to inspect completed scalar outputs.";
    });
    private async void HistoryClicked(object? sender, RoutedEventArgs e) => await RunAsync(async () => Messages.Text = $"Queue: {await host.GetAsync("/queue")}\nHistory: {await host.GetAsync("/history")}");
    private async void InterruptClicked(object? sender, RoutedEventArgs e) => await RunAsync(async () => { if (lastPromptId is null) throw new InvalidOperationException("No job has been submitted in this Host session."); Messages.Text = (await host.InterruptAsync(lastPromptId)).ToJsonString(); });
    private async void RestartClicked(object? sender, RoutedEventArgs e) => await RunAsync(StartHostAsync);
}
