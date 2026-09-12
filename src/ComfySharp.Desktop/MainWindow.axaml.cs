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
    private long submissionOrder;
    private HashSet<string>? availableNodes;
    private HashSet<string> outputNodes = new(StringComparer.Ordinal);
    private readonly CancellationTokenSource lifetime = new();
    private readonly HostSession hostSession = new();
    public MainWindow() : this(true) { }
    public MainWindow(bool startHost)
    {
        InitializeComponent(); AddDocument(WorkflowDocument.Create(), null);
        host.Changed += (_, _) => Dispatcher.UIThread.Post(() =>
        {
            HostStatus.Text = host.Status;
            if (!host.Ready) { availableNodes = null; outputNodes.Clear(); foreach (var tab in Documents.Items.OfType<TabItem>()) ((DocumentEditor)tab.Content!).SetAvailability(null); }
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
        Closed += (_, _) => { hostSession.Close(); lifetime.Cancel(); host.Dispose(); };
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
    private async Task RunAsync(Func<Task> action)
    {
        try { await action(); }
        catch (OperationCanceledException) { }
        catch (Exception error) { if (!lifetime.IsCancellationRequested) Messages.Text = error.Message; }
    }
    private async Task StartHostAsync()
    {
        availableNodes = null; lastPromptId = null; var session = hostSession.Restart();
        await host.StartAsync();
        hostSession.Require(session);
        var info = await hostSession.ObserveAsync(host.GetAsync("/object_info"), session);
        hostSession.Require(session);
        availableNodes = info.Select(p => p.Key).ToHashSet(StringComparer.Ordinal);
        outputNodes = info.Where(p => p.Value?["output_node"]?.GetValue<bool>() == true).Select(p => p.Key).ToHashSet(StringComparer.Ordinal);
        foreach (var tab in Documents.Items.OfType<TabItem>()) ((DocumentEditor)tab.Content!).SetAvailability(availableNodes);
        Messages.Text = $"Host provides {availableNodes.Count} node types. Queue validates the complete document before submission.";
    }
    private async Task SmokeAsync()
    {
        var id = ActiveEditor.AddNode("PrimitiveString");
        var textPreview = ActiveEditor.AddNode("PreviewAny");
        ActiveEditor.Document.Connect(id, 0, textPreview, 0);
        ActiveEditor.Reload();
        var accepted = await host.SubmitAsync(Compile(true), clientId, [textPreview.Value]);
        var textEntry = await WaitForJobAsync(accepted["prompt_id"]!.GetValue<string>(), hostSession.Id, 200);
        if (textEntry["outputs"]?[textPreview.Value]?["text"]?[0]?.GetValue<string>() != "Hello from ComfySharp")
            throw new InvalidOperationException("Text preview smoke failed: " + textEntry.ToJsonString());
        ActiveEditor.ApplyUiOutputs(textEntry["outputs"]!.AsObject());

        var schedule = ActiveEditor.AddNode("KarrasScheduler");
        ActiveEditor.Document.SetWidgets(schedule, new JsonArray(3, 3.0, 1.0, 1.0));
        var split = ActiveEditor.AddNode("SplitSigmas");
        ActiveEditor.Document.SetWidgets(split, new JsonArray(1));
        var sigmaPreview = ActiveEditor.AddNode("PreviewAny");
        ActiveEditor.Document.Connect(schedule, 0, split, 0);
        ActiveEditor.Document.Connect(split, 1, sigmaPreview, 0);
        ActiveEditor.Reload();
        accepted = await host.SubmitAsync(Compile(true), clientId, [sigmaPreview.Value]);
        var sigmaEntry = await WaitForJobAsync(accepted["prompt_id"]!.GetValue<string>(), hostSession.Id, 200);
        if (sigmaEntry["outputs"]?[sigmaPreview.Value]?["text"]?[0]?.GetValue<string>() != "tensor([2., 1., 0.])" || sigmaEntry["outputs"]!.AsObject().Count != 1)
            throw new InvalidOperationException("Native sigma preview smoke failed: " + sigmaEntry.ToJsonString());
        ActiveEditor.ApplyUiOutputs(sigmaEntry["outputs"]!.AsObject());
        var casePrompt = JsonNode.Parse("""
            {"preview":{"class_type":"PreviewAny","inputs":{"source":"lowercase"}},
             "PREVIEW":{"class_type":"PreviewAny","inputs":{"source":"uppercase"}}}
            """)!.AsObject();
        accepted = await host.SubmitAsync(casePrompt, clientId, ["preview", "PREVIEW"]);
        var caseEntry = await WaitForJobAsync(accepted["prompt_id"]!.GetValue<string>(), hostSession.Id, 200);
        var caseOutputs = caseEntry["outputs"]!.AsObject();
        if (caseOutputs.Count != 2 || caseOutputs["preview"]?["text"]?[0]?.GetValue<string>() != "lowercase" ||
            caseOutputs["PREVIEW"]?["text"]?[0]?.GetValue<string>() != "uppercase")
            throw new InvalidOperationException("Case-sensitive history smoke failed: " + caseEntry.ToJsonString());
        if (availableNodes?.Contains("StringFormat") != true)
            throw new InvalidOperationException("StringFormat smoke requires the real registered Host node.");
        // Exercise the real formatter through the supervised Host; this direct prompt
        // does not claim that the document editor exposes dynamic formatter ports.
        var formatPrompt = JsonNode.Parse("""
            {"format-input":{"class_type":"PrimitiveString","inputs":{"value":"xy"}},
             "format":{"class_type":"StringFormat","inputs":{"values.a":["format-input",0],"f_string":"{a:*>6}"}},
             "format-preview":{"class_type":"PreviewAny","inputs":{"source":["format",0]}}}
            """)!.AsObject();
        var formatSession = hostSession.Id;
        accepted = await hostSession.ObserveAsync(host.SubmitAsync(formatPrompt, clientId, ["format-preview"]), formatSession);
        var formatEntry = await WaitForJobAsync(accepted["prompt_id"]!.GetValue<string>(), formatSession, 200);
        hostSession.Require(formatSession);
        var formatOutputs = formatEntry["outputs"]!.AsObject();
        if (formatOutputs.Count != 1 || formatOutputs["format-preview"]?["text"] is not JsonArray formattedText ||
            formattedText.Count != 1 || formattedText[0]?.GetValue<string>() != "****xy")
            throw new InvalidOperationException("StringFormat Host preview smoke failed: " + formatEntry.ToJsonString());
        if (availableNodes?.Contains("StringContains") != true || !availableNodes.Contains("StringCompare"))
            throw new InvalidOperationException("Text comparison smoke requires both registered Host nodes.");
        var insensitive = ActiveEditor.AddNode("PrimitiveBoolean");
        ActiveEditor.Document.SetWidgets(insensitive, new JsonArray(false));
        var expectedComparisons = new Dictionary<string, string>();
        foreach (var (type, widgets, booleanSlot, expected) in new[]
        {
            ("StringContains", new JsonArray("İ", "i\u0307", true), 2, "True"),
            ("StringCompare", new JsonArray("ΟΣ", "Σ", "Ends With", true), 3, "False")
        })
        {
            var comparison = ActiveEditor.AddNode(type);
            ActiveEditor.Document.SetWidgets(comparison, widgets);
            ActiveEditor.Document.Connect(insensitive, 0, comparison, booleanSlot);
            var preview = ActiveEditor.AddNode("PreviewAny");
            ActiveEditor.Document.Connect(comparison, 0, preview, 0);
            expectedComparisons.Add(preview.Value, expected);
        }
        ActiveEditor.Reload();
        var comparisonSession = hostSession.Id;
        accepted = await hostSession.ObserveAsync(host.SubmitAsync(Compile(true), clientId, expectedComparisons.Keys.ToArray()), comparisonSession);
        var comparisonEntry = await WaitForJobAsync(accepted["prompt_id"]!.GetValue<string>(), comparisonSession, 200);
        hostSession.Require(comparisonSession);
        var comparisonOutputs = comparisonEntry["outputs"]!.AsObject();
        if (comparisonOutputs.Count != expectedComparisons.Count || expectedComparisons.Any(pair =>
            comparisonOutputs[pair.Key]?["text"] is not JsonArray values || values.Count != 1 || values[0]?.GetValue<string>() != pair.Value))
            throw new InvalidOperationException("Text comparison workflow smoke failed: " + comparisonEntry.ToJsonString());
        ActiveEditor.ApplyUiOutputs(comparisonOutputs);
        if (availableNodes?.Contains("CaseConverter") != true)
            throw new InvalidOperationException("Case conversion smoke requires CaseConverter in Host.");
        var casingText = ActiveEditor.AddNode("PrimitiveString");
        ActiveEditor.Document.SetWidgets(casingText, new JsonArray("ǳABC AΣ"));
        var expectedCasing = new Dictionary<string, string>();
        foreach (var (mode, expected) in new[]
        {
            ("UPPERCASE", "ǱABC AΣ"), ("lowercase", "ǳabc aς"),
            ("Capitalize", "ǲabc aς"), ("Title Case", "ǲabc Aς")
        })
        {
            var converter = ActiveEditor.AddNode("CaseConverter");
            ActiveEditor.Document.SetWidgets(converter, new JsonArray("saved text", mode));
            ActiveEditor.Document.Connect(casingText, 0, converter, 0);
            var preview = ActiveEditor.AddNode("PreviewAny");
            ActiveEditor.Document.Connect(converter, 0, preview, 0);
            expectedCasing.Add(preview.Value, expected);
        }
        ActiveEditor.Reload();
        var casingSession = hostSession.Id;
        accepted = await hostSession.ObserveAsync(host.SubmitAsync(Compile(true), clientId, expectedCasing.Keys.ToArray()), casingSession);
        var casingEntry = await WaitForJobAsync(accepted["prompt_id"]!.GetValue<string>(), casingSession, 200);
        hostSession.Require(casingSession);
        var casingOutputs = casingEntry["outputs"]!.AsObject();
        if (casingOutputs.Count != expectedCasing.Count || expectedCasing.Any(pair =>
            casingOutputs[pair.Key]?["text"] is not JsonArray values || values.Count != 1 || values[0]?.GetValue<string>() != pair.Value))
            throw new InvalidOperationException("Case conversion workflow smoke failed: " + casingEntry.ToJsonString());
        ActiveEditor.ApplyUiOutputs(casingOutputs);
        var image = ActiveEditor.AddNode("EmptyImage");
        ActiveEditor.Document.SetWidgets(image, new JsonArray(1, 1, 2, 0xff0000));
        var repeatedImage = ActiveEditor.AddNode("RepeatImageBatch");
        ActiveEditor.Document.SetWidgets(repeatedImage, new JsonArray(3));
        var extractedImage = ActiveEditor.AddNode("ImageFromBatch");
        ActiveEditor.Document.SetWidgets(extractedImage, new JsonArray(-1, 10));
        var invertedImage = ActiveEditor.AddNode("ImageInvert");
        var imageTextPreview = ActiveEditor.AddNode("PreviewAny");
        ActiveEditor.Document.Connect(image, 0, repeatedImage, 0);
        ActiveEditor.Document.Connect(repeatedImage, 0, extractedImage, 0);
        ActiveEditor.Document.Connect(extractedImage, 0, invertedImage, 0);
        ActiveEditor.Document.Connect(invertedImage, 0, imageTextPreview, 0);
        ActiveEditor.Reload();
        var imageSession = hostSession.Id;
        accepted = await hostSession.ObserveAsync(host.SubmitAsync(Compile(true), clientId, [imageTextPreview.Value]), imageSession);
        var imageEntry = await WaitForJobAsync(accepted["prompt_id"]!.GetValue<string>(), imageSession, 200);
        hostSession.Require(imageSession);
        var imageOutputs = imageEntry["outputs"]!.AsObject();
        if (imageOutputs.Count != 1 || imageOutputs[imageTextPreview.Value]?["text"] is not JsonArray imageTexts ||
            imageTexts.Count != 1 || imageTexts[0]?.GetValue<string>() != "tensor([[[[0., 1., 1.]]]])")
            throw new InvalidOperationException("Image primitive workflow smoke failed: " + imageEntry.ToJsonString());
        ActiveEditor.ApplyUiOutputs(imageOutputs);
        var secondImage = ActiveEditor.AddNode("EmptyImage");
        ActiveEditor.Document.SetWidgets(secondImage, new JsonArray(2, 2, 1, 0x00ff00));
        var imageBatch = ActiveEditor.AddNode("ImageBatch");
        var lastImage = ActiveEditor.AddNode("ImageFromBatch");
        ActiveEditor.Document.SetWidgets(lastImage, new JsonArray(-1, 1));
        var batchPreview = ActiveEditor.AddNode("PreviewAny");
        ActiveEditor.Document.Connect(image, 0, imageBatch, 0);
        ActiveEditor.Document.Connect(secondImage, 0, imageBatch, 1);
        ActiveEditor.Document.Connect(imageBatch, 0, lastImage, 0);
        ActiveEditor.Document.Connect(lastImage, 0, batchPreview, 0);
        ActiveEditor.Reload();
        var batchSession = hostSession.Id;
        accepted = await hostSession.ObserveAsync(host.SubmitAsync(Compile(true), clientId, [batchPreview.Value]), batchSession);
        var batchEntry = await WaitForJobAsync(accepted["prompt_id"]!.GetValue<string>(), batchSession, 200);
        hostSession.Require(batchSession);
        var batchOutputs = batchEntry["outputs"]!.AsObject();
        if (batchOutputs.Count != 1 || batchOutputs[batchPreview.Value]?["text"] is not JsonArray batchTexts ||
            batchTexts.Count != 1 || batchTexts[0]?.GetValue<string>() != "tensor([[[[0., 1., 0.]]]])")
            throw new InvalidOperationException("ImageBatch resize workflow smoke failed: " + batchEntry.ToJsonString());
        ActiveEditor.ApplyUiOutputs(batchOutputs);
        Console.WriteLine("ComfySharp Desktop smoke passed: native window, supervised Host, text and CPU sigma graphs, case-sensitive UI history, StringFormat Host preview, text comparison workflows, four CaseConverter modes, four IMAGE primitives with text preview, ImageBatch resize with text preview and native preview.");
    }
    private async Task<JsonObject> WaitForJobAsync(string jobId, int session, int? maxAttempts = null)
    {
        for (var attempt = 0; !maxAttempts.HasValue || attempt < maxAttempts.Value; attempt++)
        {
            lifetime.Token.ThrowIfCancellationRequested();
            hostSession.Require(session);
            if (!host.Ready) throw new OperationCanceledException("Host session ended; the job will not be replayed.");
            var history = await hostSession.ObserveAsync(host.GetAsync("/history"), session);
            hostSession.Require(session);
            if (history[jobId] is JsonObject entry)
            {
                if (entry["status"]?["completed"]?.GetValue<bool>() != true)
                    throw new InvalidOperationException("Job failed: " + entry.ToJsonString());
                return entry;
            }
            await Task.Delay(maxAttempts.HasValue ? 100 : 500, lifetime.Token);
        }
        throw new TimeoutException("Smoke job did not complete.");
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
        var editor = ActiveEditor;
        var session = hostSession.Id;
        var targets = prompt.Where(p => outputNodes.Contains(p.Value!["class_type"]!.GetValue<string>())).Select(p => p.Key).ToArray();
        if (targets.Length == 0) throw new InvalidOperationException("Connect the result to an available output node, such as PreviewAny, before queueing.");
        var submission = editor.BeginSubmission();
        var order = ++submissionOrder;
        var result = await hostSession.ObserveAsync(host.SubmitAsync(prompt, clientId, targets), session);
        hostSession.Require(session);
        var jobId = result["prompt_id"]?.GetValue<string>() ?? throw new InvalidOperationException("Host did not return a prompt ID.");
        if (order == submissionOrder) lastPromptId = jobId;
        Messages.Text = $"Submitted {jobId}. Waiting for output…";
        var entry = await WaitForJobAsync(jobId, session);
        hostSession.Require(session);
        Messages.Text = editor.ApplyUiOutputs(entry["outputs"]!.AsObject(), submission)
            ? $"Completed {jobId}. Outputs are shown in the document that submitted the job."
            : $"Completed {jobId}. The document changed or a newer job was submitted; this result remains in history.";
    });
    private async void HistoryClicked(object? sender, RoutedEventArgs e) => await RunAsync(async () => Messages.Text = $"Queue: {await host.GetAsync("/queue")}\nHistory: {await host.GetAsync("/history")}");
    private async void InterruptClicked(object? sender, RoutedEventArgs e) => await RunAsync(async () => { if (lastPromptId is null) throw new InvalidOperationException("No job has been submitted in this Host session."); Messages.Text = (await host.InterruptAsync(lastPromptId)).ToJsonString(); });
    private async void RestartClicked(object? sender, RoutedEventArgs e) => await RunAsync(StartHostAsync);
}
