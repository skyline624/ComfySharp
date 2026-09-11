using System.Text.Json;
using System.Text.Json.Nodes;
using System.Globalization;

namespace ComfySharp.Workflow;

public readonly record struct NodeId(string Value)
{
    public static NodeId From(JsonNode? value)
    {
        if (value is JsonValue v && v.TryGetValue<string>(out var text)) return new(text);
        // Upstream parses numbers as JavaScript doubles: 1, 1.0 and 1e0 have the same identity.
        // Keep the source token in the document; normalize only the projected lookup/prompt key.
        if (value?.GetValueKind() != JsonValueKind.Number ||
            !double.TryParse(value.ToJsonString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var number) ||
            !double.IsFinite(number) || Math.Truncate(number) != number)
            throw new FormatException("Node ID must be a string or an integer-valued number.");
        return new(number == 0 ? "0" : number.ToString("0", CultureInfo.InvariantCulture));
    }
    public override string ToString() => Value;
}
public sealed record GraphNode(NodeId Id, string Type, string Title, double X, double Y, JsonObject Data);
public sealed record GraphLink(long Id, NodeId Source, int SourceSlot, NodeId Target, int TargetSlot, JsonNode? Type);

/// <summary>JSON is the authoritative document. Unknown extension data survives every edit and undo.</summary>
public sealed class WorkflowDocument
{
    private JsonObject root;
    private readonly Stack<string> undo = new(), redo = new();
    private string saved;
    public event EventHandler? Changed;
    private WorkflowDocument(JsonObject root) { this.root = root; saved = ToJson(); }
    public static WorkflowDocument Create() => Parse("{\"version\":0.4,\"last_node_id\":0,\"last_link_id\":0,\"nodes\":[],\"links\":[],\"groups\":[],\"extra\":{}}");
    public static WorkflowDocument Parse(string json)
    {
        var root = JsonNode.Parse(json) as JsonObject ?? throw new FormatException("A workflow must be a JSON object.");
        var version = root["version"]?.GetValue<double>();
        if (version is not (0.4 or 1)) throw new FormatException("Only workflow versions 0.4 and 1 are supported.");
        if (root["nodes"] is not JsonArray) throw new FormatException("Workflow nodes must be an array.");
        var result = new WorkflowDocument(root);
        if (result.Nodes.Select(n => n.Id).Distinct().Count() != result.Nodes.Count) throw new FormatException("Duplicate node IDs cannot be edited safely.");
        return result;
    }
    public string ToJson() => root.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    public JsonObject Snapshot() => (JsonObject)root.DeepClone();
    public bool IsDirty => saved != ToJson();
    public bool CanUndo => undo.Count > 0;
    public bool CanRedo => redo.Count > 0;
    public void MarkSaved() => MarkSaved(ToJson());
    public void MarkSaved(string writtenSnapshot) { saved = writtenSnapshot; Changed?.Invoke(this, EventArgs.Empty); }
    public IReadOnlyList<GraphNode> Nodes => ((JsonArray)root["nodes"]!).Select(n =>
    {
        var data = n as JsonObject ?? throw new FormatException("Each node must be an object.");
        return new GraphNode(NodeId.From(data["id"]), data["type"]?.GetValue<string>() ?? "Unknown", data["title"]?.GetValue<string>() ?? data["type"]?.GetValue<string>() ?? "Unknown", Coordinate(data["pos"], 0), Coordinate(data["pos"], 1), (JsonObject)data.DeepClone());
    }).ToArray();
    public IReadOnlyList<GraphLink> Links => (root["links"] as JsonArray ?? []).Select(ParseLink).ToArray();
    private static double Coordinate(JsonNode? position, int index) => (position is JsonArray a && a.Count > index ? a[index] : position is JsonObject o ? o[index.ToString()] : null)?.GetValue<double>() ?? 0;
    private static int Slot(JsonNode? value) => int.Parse(NodeId.From(value).Value, System.Globalization.CultureInfo.InvariantCulture);
    private static GraphLink ParseLink(JsonNode? item) => item switch
    {
        JsonArray a when a.Count >= 6 => new(a[0]!.GetValue<long>(), NodeId.From(a[1]), Slot(a[2]), NodeId.From(a[3]), Slot(a[4]), a[5]?.DeepClone()),
        JsonObject o => new(o["id"]!.GetValue<long>(), NodeId.From(o["origin_id"]), Slot(o["origin_slot"]), NodeId.From(o["target_id"]), Slot(o["target_slot"]), o["type"]?.DeepClone()),
        _ => throw new FormatException("Invalid workflow link.")
    };
    private JsonObject Find(NodeId id) => ((JsonArray)root["nodes"]!).OfType<JsonObject>().Single(n => NodeId.From(n["id"]) == id);
    private void Edit(Action action)
    {
        var before = ToJson();
        try { action(); } catch { root = (JsonObject)JsonNode.Parse(before)!; throw; }
        if (before == ToJson()) return;
        undo.Push(before); redo.Clear(); Changed?.Invoke(this, EventArgs.Empty);
    }
    public void Move(NodeId id, double x, double y) => Edit(() =>
    {
        if (!double.IsFinite(x) || !double.IsFinite(y)) throw new ArgumentOutOfRangeException(nameof(x));
        var node = Find(id);
        if (node["pos"] is JsonObject position) { position["0"] = x; position["1"] = y; }
        else node["pos"] = new JsonArray(x, y);
    });
    public void Rename(NodeId id, string title) => Edit(() => Find(id)["title"] = title);
    public void SetWidgets(NodeId id, JsonNode values) => Edit(() =>
    {
        if (values is not (JsonArray or JsonObject)) throw new ArgumentException("Widget values must be an array or object.");
        Find(id)["widgets_values"] = values.DeepClone();
    });
    public NodeId AddNode(string type, double x = 100, double y = 100, JsonObject? template = null)
    {
        var next = Nodes.Select(n => long.TryParse(n.Id.Value, out var id) ? id : 0).DefaultIfEmpty().Max() + 1;
        var id = new NodeId(next.ToString());
        Edit(() => {
            var data = template is null ? new JsonObject { ["size"] = new JsonArray(220, 100), ["mode"] = 0, ["order"] = Nodes.Count, ["flags"] = new JsonObject(), ["properties"] = new JsonObject(), ["inputs"] = new JsonArray(), ["outputs"] = new JsonArray() } : (JsonObject)template.DeepClone();
            data["id"] = next; data["type"] = type; data["pos"] = new JsonArray(x, y);
            ((JsonArray)root["nodes"]!).Add(data); UpdateCounter("last_node_id", "lastNodeId", next);
        });
        return id;
    }
    public void Delete(NodeId id) => Edit(() =>
    {
        foreach (var link in Links.Where(l => l.Source == id || l.Target == id).ToArray()) DisconnectCore(link.Id);
        ((JsonArray)root["nodes"]!).Remove(Find(id));
    });
    public void Connect(NodeId source, int output, NodeId target, int input) => Edit(() =>
    {
        var from = Find(source); var to = Find(target);
        var outputData = (from["outputs"] as JsonArray)?[output] as JsonObject ?? throw new ArgumentException("Output slot is missing.");
        var inputData = (to["inputs"] as JsonArray)?[input] as JsonObject ?? throw new ArgumentException("Input slot is missing.");
        var a = outputData["type"]?.ToJsonString(); var b = inputData["type"]?.ToJsonString();
        if (a != b && a != "\"*\"" && b != "\"*\"") throw new ArgumentException("The selected ports have incompatible types.");
        foreach (var link in Links.Where(l => l.Target == target && l.TargetSlot == input).ToArray()) DisconnectCore(link.Id);
        var next = Links.Select(l => l.Id).DefaultIfEmpty().Max() + 1;
        var links = root["links"] as JsonArray;
        if (links is null) root["links"] = links = new JsonArray();
        links.Add(root["version"]!.GetValue<double>() == 1
            ? new JsonObject { ["id"] = next, ["origin_id"] = from["id"]!.DeepClone(), ["origin_slot"] = output, ["target_id"] = to["id"]!.DeepClone(), ["target_slot"] = input, ["type"] = outputData["type"]?.DeepClone() }
            : new JsonArray(JsonValue.Create(next), from["id"]!.DeepClone(), JsonValue.Create(output), to["id"]!.DeepClone(), JsonValue.Create(input), outputData["type"]?.DeepClone()));
        inputData["link"] = next;
        if (outputData["links"] is not JsonArray) outputData["links"] = new JsonArray();
        ((JsonArray)outputData["links"]!).Add(next);
        UpdateCounter("last_link_id", "lastLinkId", next);
    });
    public void Disconnect(long id) => Edit(() => DisconnectCore(id));
    private void DisconnectCore(long id)
    {
        var links = (JsonArray)root["links"]!;
        var raw = links.Single(l => ParseLink(l).Id == id); var link = ParseLink(raw);
        if ((Find(link.Target)["inputs"] as JsonArray)?[link.TargetSlot] is JsonObject input) input["link"] = null;
        if ((Find(link.Source)["outputs"] as JsonArray)?[link.SourceSlot]?["links"] is JsonArray outputs)
            foreach (var item in outputs.Where(i => i?.GetValue<long>() == id).ToArray()) outputs.Remove(item);
        links.Remove(raw);
    }
    private void UpdateCounter(string legacy, string modern, long value)
    {
        if (root["version"]!.GetValue<double>() == 1) { if (root["state"] is not JsonObject) root["state"] = new JsonObject(); root["state"]![modern] = value; }
        else root[legacy] = value;
    }
    public void Undo() { if (undo.TryPop(out var json)) { redo.Push(ToJson()); root = (JsonObject)JsonNode.Parse(json)!; Changed?.Invoke(this, EventArgs.Empty); } }
    public void Redo() { if (redo.TryPop(out var json)) { undo.Push(ToJson()); root = (JsonObject)JsonNode.Parse(json)!; Changed?.Invoke(this, EventArgs.Empty); } }
}
