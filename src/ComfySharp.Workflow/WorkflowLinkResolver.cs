using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace ComfySharp.Workflow;

/// <summary>Resolves static workflow links through muted and bypass nodes, without editing the document.</summary>
internal sealed class WorkflowLinkResolver(IReadOnlyDictionary<NodeId, GraphNode> nodes, IReadOnlyList<GraphLink> links,
    List<CompilationDiagnostic> diagnostics, List<CompilationDiagnostic> warnings)
{
    public static int Mode(GraphNode node) => node.Data["mode"] is null ? 0 :
        node.Data["mode"] is JsonValue value && value.TryGetValue<int>(out int mode) ? mode : -1;

    public GraphLink? Resolve(GraphNode target, int targetSlot)
    {
        var node = target; int slot = targetSlot;
        var visited = new HashSet<(NodeId Node, int Slot, bool Output)>();
        while (true)
        {
            if (!visited.Add((node.Id, slot, false))) return Cycle(node);
            if (node.Data["inputs"] is not JsonArray inputs || slot < 0 || slot >= inputs.Count || inputs[slot] is not JsonObject input)
            { diagnostics.Add(new("invalid_input", "A bypass route refers to a missing input slot.", node.Id)); return null; }
            if (input["link"] is null) return null;
            if (input["link"] is not JsonValue value || !value.TryGetValue<long>(out long id))
            { diagnostics.Add(new("invalid_link", "Input link ID must be an integer.", node.Id)); return null; }
            var matches = links.Where(link => link.Id == id && link.Target == node.Id && link.TargetSlot == slot).ToArray();
            if (matches.Length != 1 || !nodes.TryGetValue(matches[0].Source, out var source))
            { diagnostics.Add(new("invalid_link", "Input has a dangling or inconsistent link.", node.Id)); return null; }
            var link = matches[0];
            if (!visited.Add((source.Id, link.SourceSlot, true))) return Cycle(source);
            int mode = Mode(source);
            if (mode == 2) return null;
            if (source.Data["outputs"] is not JsonArray outputs || link.SourceSlot < 0 || link.SourceSlot >= outputs.Count)
            { diagnostics.Add(new("invalid_output", "Source output slot does not exist.", source.Id)); return null; }
            if (mode != 4) return link with { Target = target.Id, TargetSlot = targetSlot };
            var candidates = source.Data["inputs"] as JsonArray ?? [];
            JsonNode? wanted = input["type"], outputType = outputs[link.SourceSlot]?["type"];
            int chosen = BypassSlot(candidates, link.SourceSlot, wanted, outputType);
            if (chosen < 0)
            { warnings.Add(new("bypass_no_match", "No bypass input matches the destination type; its connection is omitted.", source.Id)); return null; }
            // Each bypass hop resolves its selected input with that input's own type, as upstream does.
            node = source; slot = chosen;
        }
    }

    private GraphLink? Cycle(GraphNode node)
    { diagnostics.Add(new("cycle", "Circular reference while resolving a bypass connection.", node.Id)); return null; }

    private static int BypassSlot(JsonArray inputs, int outputSlot, JsonNode? wanted, JsonNode? outputType)
    {
        if (wanted is JsonValue v && v.TryGetValue<string>(out var text) && text is "*" or "")
            return inputs.Count > outputSlot ? outputSlot : 0;
        if (outputSlot < inputs.Count && Compatible(inputs[outputSlot]?["type"], outputType) && Compatible(inputs[outputSlot]?["type"], wanted)) return outputSlot;
        for (int i = 0; i < inputs.Count; i++)
            if (StrictTypeEqual(inputs[i]?["type"], wanted)) return i;
        for (int i = 0; i < inputs.Count; i++)
            if (Compatible(inputs[i]?["type"], outputType) && Compatible(inputs[i]?["type"], wanted)) return i;
        return -1;
    }

    private static bool StrictTypeEqual(JsonNode? a, JsonNode? b) =>
        a is not (JsonArray or JsonObject) && b is not (JsonArray or JsonObject) && JsonNode.DeepEquals(a, b);

    private static bool Generic(JsonNode? type) => type is null || type.GetValueKind() is JsonValueKind.False ||
        type is JsonValue value && (value.TryGetValue<string>(out var s) && s is "" or "*" ||
            type.GetValueKind() == JsonValueKind.Number && double.Parse(type.ToJsonString(), CultureInfo.InvariantCulture) == 0);

    private static string Text(JsonNode? type) => type switch
    {
        null => "",
        JsonArray array => string.Join(",", array.Select(Text)),
        JsonObject => "[object Object]",
        JsonValue value when value.TryGetValue<string>(out var text) => text,
        _ when type.GetValueKind() == JsonValueKind.Number => double.Parse(type.ToJsonString(), CultureInfo.InvariantCulture).ToString("G", CultureInfo.InvariantCulture),
        _ => type.ToJsonString()
    };

    private static bool Compatible(JsonNode? a, JsonNode? b)
    {
        if (Generic(a) || Generic(b) || StrictTypeEqual(a, b)) return true;
        string left = Text(a).ToLowerInvariant(), right = Text(b).ToLowerInvariant();
        foreach (string first in left.Split(','))
            foreach (string second in right.Split(','))
                if (first is "" or "*" || second is "" or "*" || first == second) return true;
        return false;
    }
}
