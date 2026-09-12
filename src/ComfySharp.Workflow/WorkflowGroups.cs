using System.Globalization;
using System.Text.Json.Nodes;

namespace ComfySharp.Workflow;

public readonly record struct GraphRect(double X, double Y, double Width, double Height)
{
    public GraphRect Checked()
    {
        if (!double.IsFinite(X) || !double.IsFinite(Y) || !double.IsFinite(Width) || !double.IsFinite(Height) ||
            !double.IsFinite(X + Width) || !double.IsFinite(Y + Height) || Width < 0 || Height < 0)
            throw new FormatException("Graph bounds must have finite coordinates and nonnegative dimensions.");
        return this;
    }
    public bool Contains(double x, double y) => x >= X && y >= Y && x <= X + Width && y <= Y + Height;
    public bool Contains(GraphRect other) => Contains(other.X, other.Y) && Contains(other.X + other.Width, other.Y + other.Height);
}
public sealed record GraphGroup(int Index, string Title, string Color, GraphRect Bounds, bool Pinned, JsonObject Data);

public sealed partial class WorkflowDocument
{
    public IReadOnlyList<GraphGroup> Groups => (root["groups"] is null ? new JsonArray() : root["groups"] as JsonArray ?? throw new FormatException("Workflow groups must be an array."))
        .Select((item, index) =>
        {
            var data = item as JsonObject ?? throw new FormatException("Each group must be an object.");
            var bounds = data["bounding"] ?? throw new FormatException("Group bounding rectangle is missing.");
            if (bounds is not (JsonArray or JsonObject) || bounds is JsonObject map && Enumerable.Range(0, 4).Any(i => map[i.ToString(CultureInfo.InvariantCulture)] is null))
                throw new FormatException("Group bounding rectangle requires four coordinates.");
            if (bounds is JsonArray { Count: < 4 }) throw new FormatException("Group bounding rectangle requires four coordinates.");
            var rectangle = new GraphRect(Coordinate(bounds, 0), Coordinate(bounds, 1), Coordinate(bounds, 2), Coordinate(bounds, 3)).Checked();
            return new GraphGroup(index, data["title"]?.GetValue<string>() ?? "Group", data["color"]?.GetValue<string>() ?? "#335", rectangle,
                data["flags"] is JsonObject flags && Flag(flags, "pinned", true), data.DeepClone().AsObject());
        }).ToArray();

    private JsonObject FindGroup(int index) => (root["groups"] as JsonArray)?[index] as JsonObject ?? throw new ArgumentException("Group is missing.", nameof(index));
    private static void SetCoordinate(JsonObject data, string field, int index, double value)
    {
        if (!double.IsFinite(value)) throw new FormatException("Graph coordinates must remain finite.");
        if (data[field] is JsonArray array) { while (array.Count <= index) array.Add(0); array[index] = value; }
        else if (data[field] is JsonObject map) map[index.ToString(CultureInfo.InvariantCulture)] = value;
        else { data[field] = new JsonArray(); SetCoordinate(data, field, index, value); }
    }
    public int AddGroup(string title, GraphRect bounds, string color = "#335")
    {
        ArgumentNullException.ThrowIfNull(title); ArgumentNullException.ThrowIfNull(color); bounds.Checked();
        var current = Groups; int index = current.Count;
        long maximum = current.Select(g => g.Data["id"] is JsonValue v && v.TryGetValue<long>(out var id) ? id : 0).Append(0).Max();
        if ((root["state"] as JsonObject)?["lastGroupId"] is JsonValue counter && counter.TryGetValue<long>(out var value)) maximum = Math.Max(maximum, value);
        if (maximum >= 9007199254740991L) throw new InvalidOperationException("No further exact group IDs are available.");
        Edit(() =>
        {
            if (root["groups"] is null) root["groups"] = new JsonArray();
            root["groups"]!.AsArray().Add(new JsonObject { ["id"] = maximum + 1, ["title"] = title, ["color"] = color,
                ["bounding"] = new JsonArray(bounds.X, bounds.Y, Math.Max(140, bounds.Width), Math.Max(80, bounds.Height)), ["flags"] = new JsonObject() });
            if (root["version"]!.GetValue<double>() == 1)
            {
                if (root["state"] is null) root["state"] = new JsonObject(); root["state"]!["lastGroupId"] = maximum + 1;
            }
        });
        return index;
    }
    public int GroupNodes(IEnumerable<NodeId> selection, string title = "Group", IReadOnlyDictionary<NodeId, GraphRect>? measuredBounds = null)
    {
        ArgumentNullException.ThrowIfNull(selection); var ids = selection.Distinct().ToArray();
        if (ids.Length == 0) throw new ArgumentException("Select one or more nodes to group.", nameof(selection));
        var nodes = Nodes.ToDictionary(n => n.Id); var rectangles = ids.Select(id => NodeBounds(nodes.TryGetValue(id, out var node) ? node : throw new ArgumentException("Selected node is missing."), measuredBounds)).ToArray();
        double left = rectangles.Min(r => r.X) - 10, top = rectangles.Min(r => r.Y) - 40;
        return AddGroup(title, new(left, top, rectangles.Max(r => r.X + r.Width) + 10 - left, rectangles.Max(r => r.Y + r.Height) + 10 - top));
    }
    private static GraphRect NodeBounds(GraphNode node, IReadOnlyDictionary<NodeId, GraphRect>? measuredBounds)
    {
        if (measuredBounds?.TryGetValue(node.Id, out var measured) == true) return measured.Checked();
        if (node.Data["flags"] is JsonObject flags && Flag(flags, "collapsed", true))
            throw new NotSupportedException("Collapsed node group membership requires measured bounds from the editor.");
        double width = node.Data["size"] is null ? 220 : Coordinate(node.Data["size"], 0);
        double height = node.Data["size"] is null ? 100 : Coordinate(node.Data["size"], 1);
        return new GraphRect(node.X, node.Y - 30, width, height + 30).Checked();
    }
    public void UpdateGroup(int index, string title, string color, double width, double height) => Edit(() =>
    {
        ArgumentNullException.ThrowIfNull(title); ArgumentNullException.ThrowIfNull(color);
        var group = Groups[index]; new GraphRect(group.Bounds.X, group.Bounds.Y, width, height).Checked();
        var data = FindGroup(index); data["title"] = title; data["color"] = color;
        if (!group.Pinned) { SetCoordinate(data, "bounding", 2, Math.Max(140, width)); SetCoordinate(data, "bounding", 3, Math.Max(80, height)); }
    });
    public void SetGroupPinned(int index, bool pinned) => Edit(() =>
    {
        var group = FindGroup(index); if (group["flags"] is null) group["flags"] = new JsonObject();
        if (pinned) group["flags"]!["pinned"] = true; else group["flags"]!.AsObject().Remove("pinned");
    });
    public void RemoveGroup(int index) => Edit(() => root["groups"]!.AsArray().RemoveAt(index));
    public void MoveGroup(int index, double deltaX, double deltaY, bool skipChildren = false, IReadOnlyDictionary<NodeId, GraphRect>? measuredBounds = null)
    {
        if (!double.IsFinite(deltaX) || !double.IsFinite(deltaY)) throw new ArgumentOutOfRangeException(nameof(deltaX));
        var groups = Groups; var group = groups[index]; if (group.Pinned || deltaX == 0 && deltaY == 0) return;
        var containedNodes = skipChildren ? [] : Nodes.Where(n => { var b = NodeBounds(n, measuredBounds); return group.Bounds.Contains(b.X + b.Width / 2, b.Y + b.Height / 2); }).ToArray();
        var containedGroups = skipChildren ? [] : groups.Where(g => g.Index != index && group.Bounds.Contains(g.Bounds) && !g.Pinned).ToArray();
        Edit(() =>
        {
            foreach (var moving in containedGroups.Prepend(group))
            {
                new GraphRect(moving.Bounds.X + deltaX, moving.Bounds.Y + deltaY, moving.Bounds.Width, moving.Bounds.Height).Checked();
                var data = FindGroup(moving.Index); SetCoordinate(data, "bounding", 0, moving.Bounds.X + deltaX); SetCoordinate(data, "bounding", 1, moving.Bounds.Y + deltaY);
            }
            foreach (var node in containedNodes)
            {
                if (node.Data["flags"] is JsonObject flags && Flag(flags, "pinned", true)) continue;
                var data = Find(node.Id); SetCoordinate(data, "pos", 0, node.X + deltaX); SetCoordinate(data, "pos", 1, node.Y + deltaY);
            }
            if (!skipChildren)
            {
                var reroutes = root["version"]!.GetValue<double>() == 1 ? root["reroutes"] : (root["extra"] as JsonObject)?["reroutes"];
                if (reroutes is not null && reroutes is not JsonArray) throw new FormatException("Reroutes must be an array for group movement.");
                foreach (var item in (reroutes as JsonArray ?? []))
                {
                    var reroute = item as JsonObject ?? throw new FormatException("Reroutes must be objects for group movement.");
                    if (reroute["pos"] is not (JsonArray { Count: >= 2 } or JsonObject) || reroute["pos"] is JsonObject point && (point["0"] is null || point["1"] is null))
                        throw new FormatException("Reroute position is missing or malformed.");
                    double x = Coordinate(reroute["pos"], 0), y = Coordinate(reroute["pos"], 1);
                    new GraphRect(x, y, 0, 0).Checked();
                    if (group.Bounds.Contains(x, y)) { SetCoordinate(reroute, "pos", 0, x + deltaX); SetCoordinate(reroute, "pos", 1, y + deltaY); }
                }
            }
        });
    }
}
