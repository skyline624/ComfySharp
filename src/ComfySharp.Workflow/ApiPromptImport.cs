using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace ComfySharp.Workflow;

/// <summary>Reconstructs an editable graph without running node constructors or extension code.</summary>
public static class ApiPromptImport
{
    internal const string PropertyName = "comfysharp.api_import";
    public const int MaximumInferredOutputs = 4096;

    public static bool IsPrompt(JsonNode? value) => value is JsonObject prompt && prompt.Count > 0 &&
        prompt.All(pair => pair.Value is JsonObject node && node["class_type"] is JsonValue type &&
            type.TryGetValue<string>(out _) && node["inputs"] is JsonObject);

    public static WorkflowDocument Parse(string json, Func<string, JsonObject>? templateFactory = null)
    {
        var prompt = JsonNode.Parse(json);
        if (!IsPrompt(prompt)) throw new FormatException("An API prompt must be a nonempty object of nodes with class_type and inputs.");
        var nodes = new JsonArray(); var links = new JsonArray();
        var lookup = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
        var suppliedOutputs = new HashSet<string>(StringComparer.Ordinal);
        int order = 0; long lastLink = 0;
        foreach (var (id, value) in prompt!.AsObject())
        {
            var api = value!.AsObject(); string type = api["class_type"]!.GetValue<string>();
            var node = templateFactory?.Invoke(type)?.DeepClone().AsObject() ?? new JsonObject();
            if (node["outputs"] is JsonArray outputs && outputs.Count > 0) suppliedOutputs.Add(id);
            node["id"] = id; node["type"] = type;
            string title = (api["_meta"] as JsonObject)?["title"] is JsonValue titleValue && titleValue.TryGetValue<string>(out var text) ? text : type;
            node["title"] = title; node["mode"] = 0; node["order"] = order;
            // A stable grid gives every node a position even for cyclic or disconnected prompts.
            node["pos"] = new JsonArray(60 + order % 4 * 300, 60 + order / 4 * 800); order++;
            node["size"] ??= new JsonArray(230, 160); node["flags"] ??= new JsonObject();
            node["inputs"] ??= new JsonArray(); node["outputs"] ??= new JsonArray();
            foreach (var input in node["inputs"]!.AsArray()) input!["link"] = null;
            foreach (var output in node["outputs"]!.AsArray()) output!["links"] = new JsonArray();
            node["widgets_values"] = new JsonObject(); // Never inject template defaults into the submitted inputs.
            var fields = api.DeepClone().AsObject(); fields.Remove("inputs"); fields.Remove("class_type");
            node["properties"] ??= new JsonObject();
            node["properties"]![PropertyName] = new JsonObject { ["version"] = 1, ["title"] = title, ["fields"] = fields };
            nodes.Add(node); lookup.Add(id, node);
        }
        foreach (var (id, value) in prompt.AsObject())
        {
            var node = lookup[id]; var inputs = node["inputs"]!.AsArray();
            foreach (var (name, inputValue) in value!["inputs"]!.AsObject())
            {
                var matches = inputs.OfType<JsonObject>().Where(p => p["name"]?.GetValue<string>() == name).ToArray();
                if (matches.Length > 1) throw new FormatException($"Ambiguous input template: {name}.");
                var port = matches.SingleOrDefault();
                if (port is null) { port = new JsonObject { ["name"] = name, ["type"] = "*", ["link"] = null }; inputs.Add(port); }
                // Explicit named binding also permits a connected widget to be edited after disconnection.
                port["widget"] = new JsonObject { ["name"] = name };
                if (TryLink(inputValue, out var sourceId, out int sourceSlot))
                {
                    long linkId = ++lastLink; port["link"] = linkId;
                    links.Add(new JsonArray(linkId, sourceId, sourceSlot, id, inputs.IndexOf(port), port["type"]?.DeepClone()));
                    if (lookup.TryGetValue(sourceId, out var source))
                    {
                        var outputs = source["outputs"]!.AsArray();
                        if (!suppliedOutputs.Contains(sourceId) && sourceSlot >= 0)
                        {
                            if (sourceSlot >= MaximumInferredOutputs) throw new FormatException("API prompt exceeds the inferred output slot limit.");
                            while (outputs.Count <= sourceSlot) outputs.Add(new JsonObject { ["name"] = $"output_{outputs.Count}", ["type"] = "*", ["links"] = new JsonArray() });
                        }
                        if (sourceSlot >= 0 && sourceSlot < outputs.Count) outputs[sourceSlot]!["links"]!.AsArray().Add(linkId);
                    }
                }
                else node["widgets_values"]![name] = inputValue?.DeepClone();
            }
        }
        return WorkflowDocument.Parse(new JsonObject
        {
            ["version"] = 0.4, ["last_node_id"] = 0, ["last_link_id"] = lastLink,
            ["nodes"] = nodes, ["links"] = links, ["groups"] = new JsonArray(), ["extra"] = new JsonObject()
        }.ToJsonString());
    }

    private static bool TryLink(JsonNode? value, out string source, out int slot)
    {
        source = ""; slot = 0;
        if (value is not JsonArray { Count: 2 } array || array[0] is not JsonValue id || !id.TryGetValue<string>(out var text)) return false;
        // graph_utils.is_link accepts numeric slots. Invalid fractional/out-of-range slots cannot be represented by GraphLink.
        if (array[1]?.GetValueKind() is not (JsonValueKind.Number or JsonValueKind.True or JsonValueKind.False)) return false;
        double number = array[1]!.GetValueKind() switch
        {
            JsonValueKind.True => 1, JsonValueKind.False => 0,
            _ => double.Parse(array[1]!.ToJsonString(), CultureInfo.InvariantCulture)
        };
        if (!double.IsFinite(number) || Math.Truncate(number) != number || number < int.MinValue || number > int.MaxValue)
            throw new FormatException("API link output slots must be representable integers.");
        source = text; slot = (int)number; return true;
    }
}
