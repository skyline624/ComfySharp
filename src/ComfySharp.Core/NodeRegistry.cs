using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Core;

public sealed class NodeRegistry
{
    private readonly Dictionary<string, IRuntimeNode> nodes = new(StringComparer.Ordinal);
    public IEnumerable<IRuntimeNode> Nodes => nodes.Values;
    public void Register(IRuntimeNode node)
    {
        LegacyHiddenInputs.Validate(node.Schema);
        nodes.Add(node.Schema.ClassType, node);
    }
    public bool TryGet(string classType, out IRuntimeNode node) => nodes.TryGetValue(classType, out node!);
    public JsonObject ToObjectInfo() => Describe(nodes.Values.Select(n => n.Schema));
    /// <summary>Projects schema metadata without constructing nodes or opening their configured services.</summary>
    public static JsonObject Describe(IEnumerable<NodeSchema> schemas)
    {
        var result = new JsonObject();
        foreach (var s in schemas)
        {
            var required = new JsonObject();
            var optional = new JsonObject();
            var requiredOrder = new JsonArray();
            var optionalOrder = new JsonArray();
            foreach (var input in s.Inputs)
            {
                var options = input.Autogrow is not null ? NodeInputExpansion.TemplateOptions(input)
                    : input.Options?.DeepClone().AsObject() ?? new JsonObject();
                JsonNode type = JsonValue.Create(input.Type)!;
                if (!(s.V3ObjectInfo && input.Type == "COMBO") && options.Remove("options", out var choices) && choices is not null)
                    type = choices;
                if (input.Lazy) options["lazy"] = true;
                (input.Required ? required : optional)[input.Name] = new JsonArray(type, options);
                (input.Required ? requiredOrder : optionalOrder).Add(input.Name);
            }
            result[s.ClassType] = new JsonObject
            {
                ["name"] = s.ClassType, ["display_name"] = s.DisplayName, ["category"] = s.Category,
                ["input"] = new JsonObject { ["required"] = required, ["optional"] = optional },
                ["input_order"] = new JsonObject { ["required"] = requiredOrder, ["optional"] = optionalOrder },
                ["output"] = new JsonArray(s.Outputs.Select(o => (JsonNode?)JsonValue.Create(o.Type)).ToArray()),
                ["output_name"] = new JsonArray(s.Outputs.Select(o => (JsonNode?)JsonValue.Create(o.Name ?? o.Type)).ToArray()),
                ["output_is_list"] = new JsonArray(s.Outputs.Select(o => (JsonNode?)JsonValue.Create(o.IsList)).ToArray()),
                ["output_matchtypes"] = s.Outputs.Any(o => o.MatchTemplate is not null) ? new JsonArray(s.Outputs.Select(o => (JsonNode?)JsonValue.Create(o.MatchTemplate)).ToArray()) : null,
                ["output_node"] = s.OutputNode, ["is_input_list"] = s.InputIsList, ["experimental"] = s.Experimental
            };
            var info = result[s.ClassType]!.AsObject();
            if (s.OmitEmptyOptionalInputs && optional.Count == 0)
            {
                info["input"]!.AsObject().Remove("optional");
                info["input_order"]!.AsObject().Remove("optional");
            }
            if (s.HiddenInputs is { Count: > 0 } hiddenInputs)
            {
                var hidden = new JsonObject();
                var order = new JsonArray();
                foreach (var input in hiddenInputs) { hidden.Add(input.Name, input.Type); order.Add(input.Name); }
                info["input"]!["hidden"] = hidden;
                info["input_order"]!["hidden"] = order;
            }
            if(s.V3HiddenInputs is {Count:>0} v3Hidden)
            {
                var hidden=new JsonObject();var order=new JsonArray();
                foreach(string type in v3Hidden){string name=type.ToLowerInvariant();hidden[name]=new JsonArray(type);order.Add(name);}
                info["input"]!["hidden"]=hidden;info["input_order"]!["hidden"]=order;
            }
            if (s.Description is not null) info["description"] = s.Description;
            if (s.SearchAliases is not null) info["search_aliases"] = new JsonArray(s.SearchAliases.Select(a => (JsonNode?)JsonValue.Create(a)).ToArray());
            if (s.PythonModule is not null) info["python_module"] = s.PythonModule;
            if (s.EssentialsCategory is not null) info["essentials_category"] = s.EssentialsCategory;
            if (s.Deprecated) info["deprecated"] = true;
            if (s.V3ObjectInfo)
            {
                // Explicit V3 projection for newly ported contracts. Existing node documents keep their prior shape.
                if (optional.Count == 0)
                {
                    info["input"]!.AsObject().Remove("optional");
                    info["input_order"]!.AsObject().Remove("optional");
                }
                info["description"] = s.Description ?? "";
                info["output_tooltips"] = new JsonArray(s.Outputs.Select(_ => (JsonNode?)null).ToArray());
                info["has_intermediate_output"] = false;
                info["deprecated"] = s.Deprecated;
                info["dev_only"] = false;
                info["api_node"] = false;
                info["price_badge"] = null;
                info["essentials_category"] = s.EssentialsCategory;
                info["search_aliases"] = s.SearchAliases is { Count: > 0 }
                    ? new JsonArray(s.SearchAliases.Select(a => (JsonNode?)JsonValue.Create(a)).ToArray()) : null;
                info["python_module"] = s.PythonModule ?? "nodes";
            }
        }
        return result;
    }
}
