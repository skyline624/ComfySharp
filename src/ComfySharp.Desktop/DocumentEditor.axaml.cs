using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Text.Json.Nodes;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media;
using ComfySharp.Workflow;

namespace ComfySharp.Desktop;
public sealed partial class DocumentEditor : UserControl
{
    public WorkflowDocument Document { get; }
    public string? FilePath { get; set; }
    public event EventHandler<string>? Error;
    private readonly ObservableCollection<NodeView> nodes = [];
    private readonly ObservableCollection<ConnectionView> connections = [];
    private readonly Dictionary<string, string> previews = new(StringComparer.Ordinal);
    private string documentState;
    private long revision, submissionSequence;
    public readonly record struct PreviewSubmission(long Revision, long Sequence);
    public PreviewSubmission BeginSubmission() => new(revision, ++submissionSequence);
    private ISet<string>? availableNodes;
    public DocumentEditor() : this(WorkflowDocument.Create()) { }
    public DocumentEditor(WorkflowDocument document)
    {
        Document = document; InitializeComponent();
        documentState = document.ToJson();
        document.Changed += (_, _) =>
        {
            var state = document.ToJson();
            if (state == documentState) return; // Saving the same document does not invalidate a result.
            documentState = state; revision++;
            previews.Clear();
            foreach (var view in nodes) view.PreviewText = "";
        };
        Canvas.ItemsSource = nodes; Canvas.Connections = connections;
        UpdateNodeChoices();
        Reload();
    }
    public void SetAvailability(ISet<string>? available) { availableNodes = available; UpdateNodeChoices(); Reload(); }
    private void UpdateNodeChoices()
    {
        NodeTypes.ItemsSource = PromptCompiler.BaseDefinitions.Keys.Select(type => new NodeChoice(type, type + (availableNodes?.Contains(type) == true ? "" : availableNodes is null ? " — Host unchecked" : " — unavailable"))).ToArray();
        NodeTypes.SelectedIndex = 0;
    }
    public void Reload()
    {
        nodes.Clear();
        foreach (var node in Document.Nodes)
        {
            var view = new NodeView(node, location => { Document.Move(node.Id, location.X, location.Y); RefreshConnections(); }, availableNodes);
            if (node.Type == "PreviewAny" && previews.TryGetValue(node.Id.Value, out var text)) view.PreviewText = text;
            nodes.Add(view);
        }
        RefreshConnections();
    }
    public bool ApplyUiOutputs(JsonObject outputs, PreviewSubmission? submission = null)
    {
        if (submission.HasValue && (submission.Value.Revision != revision || submission.Value.Sequence != submissionSequence)) return false;
        foreach (var node in Document.Nodes.Where(n => n.Type == "PreviewAny"))
        {
            if (outputs[node.Id.Value]?["text"] is not { } value) continue;
            var text = value is JsonArray array ? string.Join("\n\n", array.Select(v => v!.GetValue<string>())) : value.GetValue<string>();
            previews[node.Id.Value] = text;
            var view = nodes.FirstOrDefault(n => n.Id == node.Id);
            if (view is not null) view.PreviewText = text;
        }
        return true;
    }
    private void RefreshConnections()
    {
        connections.Clear();
        foreach (var link in Document.Links)
        {
            var source = nodes.FirstOrDefault(n => n.Id == link.Source); var target = nodes.FirstOrDefault(n => n.Id == link.Target);
            if (source is not null && target is not null) connections.Add(new(source.Location + new Vector(230, 50), target.Location + new Vector(0, 50)));
        }
        LinkSummary.Text = string.Join("\n", Document.Links.Select(l => $"Link {l.Id}: {l.Source}[{l.SourceSlot}] → {l.Target}[{l.TargetSlot}]"));
    }
    public void Undo() { Document.Undo(); Reload(); }
    public void Redo() { Document.Redo(); Reload(); }
    public NodeId AddNode(string type)
    {
        var id = Document.AddNode(type, 70 + nodes.Count * 35, 70 + nodes.Count * 35, NodeTemplates.Create(type)); Reload(); return id;
    }
    private void Try(Action action) { try { action(); } catch (Exception error) { Error?.Invoke(this, error.Message); } }
    private NodeView Selected => Canvas.SelectedItem as NodeView ?? throw new InvalidOperationException("Select a node first.");
    private void AddClicked(object? sender, RoutedEventArgs e) => Try(() => AddNode(((NodeChoice)NodeTypes.SelectedItem!).Type));
    private void InspectClicked(object? sender, RoutedEventArgs e) => Try(() => { var node = Document.Nodes.Single(n => n.Id == Selected.Id); NodeTitle.Text = node.Title; Widgets.Text = node.Data["widgets_values"]?.ToJsonString() ?? "[]"; });
    private void ApplyClicked(object? sender, RoutedEventArgs e) => Try(() => { var id = Selected.Id; var values = JsonNode.Parse(Widgets.Text ?? "[]") ?? throw new FormatException("Enter widget JSON."); Document.SetWidgets(id, values); Document.Rename(id, NodeTitle.Text ?? ""); Reload(); });
    private void DeleteClicked(object? sender, RoutedEventArgs e) => Try(() => { Document.Delete(Selected.Id); Reload(); });
    private void ConnectClicked(object? sender, RoutedEventArgs e) => Try(() => { Document.Connect(new(SourceId.Text ?? ""), int.Parse(SourceSlot.Text ?? "0"), new(TargetId.Text ?? ""), int.Parse(TargetSlot.Text ?? "0")); Reload(); });
    private void DisconnectClicked(object? sender, RoutedEventArgs e) => Try(() => { Document.Disconnect(long.Parse(LinkId.Text ?? "")); Reload(); });
}
public sealed record ConnectionView(Point Source, Point Target);
public sealed record NodeChoice(string Type, string Label) { public override string ToString() => Label; }
public sealed class NodeView : INotifyPropertyChanged
{
    private Point location;
    private string previewText = "";
    private readonly Action<Point> move;
    public event PropertyChangedEventHandler? PropertyChanged;
    public NodeId Id { get; }
    public string Title { get; }
    public string Subtitle { get; }
    public string Ports { get; }
    public IBrush Outline { get; }
    public bool IsPreview { get; }
    public string PreviewText { get => previewText; set { if (previewText == value) return; previewText = value; PropertyChanged?.Invoke(this, new(nameof(PreviewText))); } }
    public Point Location { get => location; set { if (location == value) return; location = value; PropertyChanged?.Invoke(this, new(nameof(Location))); move(value); } }
    public NodeView(GraphNode node, Action<Point> move, ISet<string>? availableNodes = null)
    {
        this.move = move; location = new(node.X, node.Y); Id = node.Id; Title = node.Title;
        IsPreview = node.Type == "PreviewAny";
        var known = PromptCompiler.BaseDefinitions.ContainsKey(node.Type);
        var available = availableNodes?.Contains(node.Type) == true;
        Subtitle = $"#{node.Id} · {node.Type}" + (!known ? "\nUnsupported / preserved" : available ? "\nAvailable in Host" : availableNodes is null ? "\nHost availability unchecked" : "\nUnavailable in Host / preserved"); Outline = known && available ? Brushes.SlateBlue : Brushes.Orange;
        Ports = string.Join("\n", (node.Data["inputs"] as JsonArray ?? []).Select((n, i) => $"← {i}: {n?["name"]}" +
            (node.Type == "CreateList" && n?["name"]?.ToString() == "inputs.input0" ? " (required)" : ""))
            .Concat((node.Data["outputs"] as JsonArray ?? []).Select((n, i) => $"→ {i}: {n?["name"]}" + (node.Type == "CreateList" ? " (list)" : ""))));
    }
}
internal static class NodeTemplates
{
    public static JsonObject Create(string type)
    {
        var input = new JsonArray(); var output = new JsonArray(); JsonArray widgets = [];
        void In(string name, string dataType) => input.Add(new JsonObject { ["name"] = name, ["type"] = dataType, ["link"] = null });
        void Out(string name, string dataType) => output.Add(new JsonObject { ["name"] = name, ["type"] = dataType, ["links"] = new JsonArray() });
        switch (type)
        {
            case "PrimitiveString": case "PrimitiveStringMultiline": widgets = new("Hello from ComfySharp"); Out("STRING", "STRING"); break;
            case "PrimitiveInt": widgets = new(0, "fixed"); Out("INT", "INT"); break;
            case "PrimitiveFloat": widgets = new(0.0); Out("FLOAT", "FLOAT"); break;
            case "PrimitiveBoolean": widgets = new(false); Out("BOOLEAN", "BOOLEAN"); break;
            case "StringConcatenate": widgets = new("", "", " "); In("string_a", "STRING"); In("string_b", "STRING"); Out("STRING", "STRING"); break;
            case "StringSubstring": widgets = new("", 0, 10); In("string", "STRING"); Out("STRING", "STRING"); break;
            case "StringLength": widgets = new(""); In("string", "STRING"); Out("length", "INT"); break;
            case "StringReplace": widgets = new("", "", ""); In("string", "STRING"); Out("STRING", "STRING"); break;
            case "StringTrim": widgets = new("", "Both"); In("string", "STRING"); Out("STRING", "STRING"); break;
            case "StringContains": widgets = new("", "", true); In("string", "STRING"); In("substring", "STRING"); In("case_sensitive", "BOOLEAN"); Out("contains", "BOOLEAN"); break;
            case "StringCompare": widgets = new("", "", "Starts With", true); In("string_a", "STRING"); In("string_b", "STRING"); In("mode", "COMBO"); In("case_sensitive", "BOOLEAN"); Out("BOOLEAN", "BOOLEAN"); break;
            case "JsonExtractString": widgets = new("{}", "key"); In("json_string", "STRING"); Out("STRING", "STRING"); break;
            case "ComfyNotNode": In("value", "*"); Out("BOOLEAN", "BOOLEAN"); break;
            case "ComfySwitchNode": widgets = new(false); In("on_false", "*"); In("on_true", "*"); Out("output", "*"); break;
            case "CreateList":
                // Fixed wildcard ports are a bounded UI, not source Autogrow or
                // shared MatchType propagation. Imported nodes stay untouched.
                for (int index = 0; index < 10; index++) In($"inputs.input{index}", "*");
                Out("list", "*"); output[0]!["is_list"] = true;
                break;
            case "StringFormat":
                // Fixed named ports; imported sparse documents retain their original slots.
                for (char name = 'a'; name <= 'z'; name++) In($"values.{name}", "*");
                In("f_string", "STRING"); widgets = new("{a}"); Out("STRING", "STRING");
                break;
            case "KarrasScheduler": widgets = new(20, 14.614642, .0291675, 7.0); Out("SIGMAS", "SIGMAS"); break;
            case "ExponentialScheduler": widgets = new(20, 14.614642, .0291675); Out("SIGMAS", "SIGMAS"); break;
            case "PolyexponentialScheduler": widgets = new(20, 14.614642, .0291675, 1.0); Out("SIGMAS", "SIGMAS"); break;
            case "LaplaceScheduler": widgets = new(20, 14.614642, .0291675, 0.0, .5); Out("SIGMAS", "SIGMAS"); break;
            case "VPScheduler": widgets = new(20, 19.9, .1, .001); Out("SIGMAS", "SIGMAS"); break;
            case "SplitSigmas": widgets = new(0); In("sigmas", "SIGMAS"); Out("high_sigmas", "SIGMAS"); Out("low_sigmas", "SIGMAS"); break;
            case "SplitSigmasDenoise": widgets = new(1.0); In("sigmas", "SIGMAS"); Out("high_sigmas", "SIGMAS"); Out("low_sigmas", "SIGMAS"); break;
            case "FlipSigmas": In("sigmas", "SIGMAS"); Out("SIGMAS", "SIGMAS"); break;
            case "SetFirstSigma": widgets = new(136.0); In("sigmas", "SIGMAS"); Out("SIGMAS", "SIGMAS"); break;
            case "ExtendIntermediateSigmas": widgets = new(2, -1.0, 12.0, "linear"); In("sigmas", "SIGMAS"); Out("SIGMAS", "SIGMAS"); break;
            case "ManualSigmas": widgets = new("1, 0.5"); Out("SIGMAS", "SIGMAS"); break;
            case "PreviewAny": In("source", "*"); Out("STRING", "STRING"); break;
            case "CheckpointLoaderSimple": widgets = new("select-checkpoint.safetensors"); Out("MODEL", "MODEL"); Out("CLIP", "CLIP"); Out("VAE", "VAE"); break;
            case "CLIPTextEncode": widgets = new(""); In("clip", "CLIP"); Out("CONDITIONING", "CONDITIONING"); break;
            case "EmptyLatentImage": widgets = new(512, 512, 1); Out("LATENT", "LATENT"); break;
            case "KSampler": widgets = new(0, "fixed", 20, 8.0, "euler", "normal", 1.0); In("model", "MODEL"); In("positive", "CONDITIONING"); In("negative", "CONDITIONING"); In("latent_image", "LATENT"); Out("LATENT", "LATENT"); break;
            case "VAEDecode": In("samples", "LATENT"); In("vae", "VAE"); Out("IMAGE", "IMAGE"); break;
            case "SaveImage": widgets = new("ComfySharp"); In("images", "IMAGE"); break;
        }
        return new JsonObject { ["size"] = new JsonArray(230, type == "StringFormat" ? 720 : type == "CreateList" ? 360 : 160), ["flags"] = new JsonObject(), ["mode"] = 0, ["order"] = 0, ["properties"] = new JsonObject(), ["inputs"] = input, ["outputs"] = output, ["widgets_values"] = widgets };
    }
}
