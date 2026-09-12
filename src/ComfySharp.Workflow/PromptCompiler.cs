using System.Text.Json.Nodes;

namespace ComfySharp.Workflow;

public sealed record WidgetBinding(string Name, bool Serialize = true);
public sealed record NodeDefinition(string ClassType, IReadOnlyList<WidgetBinding> Widgets);
public sealed record CompilationDiagnostic(string Code, string Message, NodeId? Node = null);
public sealed record CompilationResult(JsonObject? Prompt, IReadOnlyList<CompilationDiagnostic> Diagnostics)
{
    public bool Success => Prompt is not null && Diagnostics.Count == 0;
}
public static class PromptCompiler
{
    // Positional bindings are explicit: object_info alone cannot describe frontend-only widgets.
    public static IReadOnlyDictionary<string, NodeDefinition> BaseDefinitions { get; } = new Dictionary<string, NodeDefinition>
    {
        ["PrimitiveString"] = new("PrimitiveString", [new("value")]),
        ["PrimitiveStringMultiline"] = new("PrimitiveStringMultiline", [new("value")]),
        ["PrimitiveInt"] = new("PrimitiveInt", [new("value"), new("control_after_generate", false)]),
        ["PrimitiveFloat"] = new("PrimitiveFloat", [new("value")]),
        ["PrimitiveBoolean"] = new("PrimitiveBoolean", [new("value")]),
        ["StringConcatenate"] = new("StringConcatenate", [new("string_a"), new("string_b"), new("delimiter")]),
        ["StringSubstring"] = new("StringSubstring", [new("string"), new("start"), new("end")]),
        ["StringLength"] = new("StringLength", [new("string")]),
        ["StringReplace"] = new("StringReplace", [new("string"), new("find"), new("replace")]),
        ["StringTrim"] = new("StringTrim", [new("string"), new("mode")]),
        ["JsonExtractString"] = new("JsonExtractString", [new("json_string"), new("key")]),
        ["ComfyNotNode"] = new("ComfyNotNode", []),
        ["ComfySwitchNode"] = new("ComfySwitchNode", [new("switch")]),
        ["CreateList"] = new("CreateList", []),
        ["StringFormat"] = new("StringFormat", [new("f_string")]),
        ["KarrasScheduler"] = new("KarrasScheduler", [new("steps"), new("sigma_max"), new("sigma_min"), new("rho")]),
        ["ExponentialScheduler"] = new("ExponentialScheduler", [new("steps"), new("sigma_max"), new("sigma_min")]),
        ["PolyexponentialScheduler"] = new("PolyexponentialScheduler", [new("steps"), new("sigma_max"), new("sigma_min"), new("rho")]),
        ["LaplaceScheduler"] = new("LaplaceScheduler", [new("steps"), new("sigma_max"), new("sigma_min"), new("mu"), new("beta")]),
        ["VPScheduler"] = new("VPScheduler", [new("steps"), new("beta_d"), new("beta_min"), new("eps_s")]),
        ["SplitSigmas"] = new("SplitSigmas", [new("step")]),
        ["SplitSigmasDenoise"] = new("SplitSigmasDenoise", [new("denoise")]),
        ["FlipSigmas"] = new("FlipSigmas", []),
        ["SetFirstSigma"] = new("SetFirstSigma", [new("sigma")]),
        ["ExtendIntermediateSigmas"] = new("ExtendIntermediateSigmas", [new("steps"), new("start_at_sigma"), new("end_at_sigma"), new("spacing")]),
        ["ManualSigmas"] = new("ManualSigmas", [new("sigmas")]),
        // preview_text and preview_mode are ephemeral frontend widgets: neither is persisted.
        ["PreviewAny"] = new("PreviewAny", []),
        ["CheckpointLoaderSimple"] = new("CheckpointLoaderSimple", [new("ckpt_name")]),
        ["CLIPTextEncode"] = new("CLIPTextEncode", [new("text")]),
        ["EmptyLatentImage"] = new("EmptyLatentImage", [new("width"), new("height"), new("batch_size")]),
        ["KSampler"] = new("KSampler", [new("seed"), new("control_after_generate", false), new("steps"), new("cfg"), new("sampler_name"), new("scheduler"), new("denoise")]),
        ["VAEDecode"] = new("VAEDecode", []),
        ["SaveImage"] = new("SaveImage", [new("filename_prefix")])
    };
    public static CompilationResult Compile(WorkflowDocument document, IReadOnlyDictionary<string, NodeDefinition>? definitions = null, ISet<string>? availableNodes = null)
    {
        definitions ??= BaseDefinitions;
        var diagnostics = new List<CompilationDiagnostic>(); var prompt = new JsonObject();
        var root = document.Snapshot(); var nodes = document.Nodes.ToDictionary(n => n.Id);
        if (root["definitions"] is JsonObject d && d.Count > 0) diagnostics.Add(new("unsupported_subgraphs", "Subgraph definitions are preserved but cannot yet be compiled."));
        foreach (var node in nodes.Values)
        {
            if ((node.Data["mode"]?.GetValue<int>() ?? 0) != 0) diagnostics.Add(new("unsupported_mode", "Muted, bypass and event modes require frontend execution semantics that are not implemented.", node.Id));
            if (!definitions.TryGetValue(node.Type, out var definition)) { diagnostics.Add(new("unsupported_node", $"No explicit widget/compiler definition exists for {node.Type}.", node.Id)); continue; }
            if (availableNodes is not null && !availableNodes.Contains(node.Type)) diagnostics.Add(new("unavailable_node", $"The Host does not provide {node.Type}.", node.Id));
            var inputs = new JsonObject();
            var values = node.Data["widgets_values"];
            if (values is JsonArray array)
            {
                if (array.Count != definition.Widgets.Count) diagnostics.Add(new("widget_layout", $"Expected {definition.Widgets.Count} persisted widgets, found {array.Count}; refusing positional guessing.", node.Id));
                for (var i = 0; i < Math.Min(array.Count, definition.Widgets.Count); i++)
                    if (definition.Widgets[i].Serialize) inputs[definition.Widgets[i].Name] = Literal(array[i]);
            }
            else if (values is JsonObject named)
            {
                foreach (var pair in named)
                    if (definition.Widgets.Any(w => w.Name == pair.Key && w.Serialize)) inputs[pair.Key] = Literal(pair.Value);
                    else if (!definition.Widgets.Any(w => w.Name == pair.Key)) diagnostics.Add(new("unknown_widget", $"Widget {pair.Key} has no serialization contract.", node.Id));
            }
            var nodeInputs = node.Data["inputs"] as JsonArray ?? [];
            // Imported slot order is authoritative; never expand or renumber imports.
            var dynamicNames = node.Type is "CreateList" or "StringFormat" ? DynamicInputNames(node, diagnostics) : null;
            if (node.Type == "CreateList" && values is not null and not JsonArray and not JsonObject)
                diagnostics.Add(new("widget_layout", "CreateList has no persisted widgets.", node.Id));
            for (var slot = 0; slot < nodeInputs.Count; slot++)
            {
                if (dynamicNames is not null && !dynamicNames.ContainsKey(slot)) continue;
                var input = nodeInputs[slot];
                if (input?["link"] is null) continue;
                long id;
                if (dynamicNames is not null)
                {
                    if (input["link"] is not JsonValue linkValue || !linkValue.TryGetValue(out id))
                    { diagnostics.Add(new("invalid_link", $"Input slot {slot} has an invalid link ID.", node.Id)); continue; }
                }
                else id = input["link"]!.GetValue<long>();
                var links = document.Links.Where(l => l.Id == id && l.Target == node.Id && l.TargetSlot == slot).ToArray();
                if (links.Length != 1 || !nodes.TryGetValue(links[0].Source, out var source)) { diagnostics.Add(new("invalid_link", $"Input slot {slot} has a dangling or inconsistent link.", node.Id)); continue; }
                var link = links[0];
                if (link.SourceSlot < 0 || link.SourceSlot >= ((source.Data["outputs"] as JsonArray)?.Count ?? 0)) diagnostics.Add(new("invalid_output", "Source output slot does not exist.", node.Id));
                inputs[dynamicNames is null ? input["name"]!.GetValue<string>() : dynamicNames[slot]] = new JsonArray(link.Source.Value, link.SourceSlot);
            }
            if (node.Type == "CreateList" && !inputs.ContainsKey("inputs.input0"))
                diagnostics.Add(new("missing_input", "CreateList requires a connection to inputs.input0.", node.Id));
            foreach (var widget in definition.Widgets.Where(w => w.Serialize))
                if (!inputs.ContainsKey(widget.Name)) diagnostics.Add(new("missing_widgets", $"Required widget {widget.Name} has neither a persisted value nor an input connection.", node.Id));
            prompt[node.Id.Value] = new JsonObject { ["class_type"] = definition.ClassType, ["inputs"] = inputs, ["_meta"] = new JsonObject { ["title"] = node.Title } };
        }
        var allLinks = document.Links;
        if (allLinks.Select(l => l.Id).Distinct().Count() != allLinks.Count) diagnostics.Add(new("duplicate_link", "The document contains duplicate link IDs."));
        foreach (var link in allLinks)
        {
            if (!nodes.ContainsKey(link.Source) || !nodes.TryGetValue(link.Target, out var target)) { diagnostics.Add(new("dangling_link", $"Link {link.Id} refers to a missing node.")); continue; }
            var slots = target.Data["inputs"] as JsonArray;
            if (target.Type is "CreateList" or "StringFormat")
            {
                if (slots is null || link.TargetSlot < 0 || link.TargetSlot >= slots.Count ||
                    slots[link.TargetSlot] is not JsonObject input || input["link"] is not JsonValue value ||
                    !value.TryGetValue<long>(out var referenced) || referenced != link.Id)
                    diagnostics.Add(new("inconsistent_link", $"Link {link.Id} is not referenced by its target input.", target.Id));
                continue;
            }
            if (slots is null || link.TargetSlot < 0 || link.TargetSlot >= slots.Count || slots[link.TargetSlot]?["link"]?.GetValue<long>() != link.Id)
                diagnostics.Add(new("inconsistent_link", $"Link {link.Id} is not referenced by its target input."));
        }
        var state = new Dictionary<NodeId, int>();
        bool Visit(NodeId id)
        {
            if (state.TryGetValue(id, out var seen)) return seen == 1;
            state[id] = 1;
            foreach (var link in allLinks.Where(l => l.Source == id && nodes.ContainsKey(l.Target)))
                if (Visit(link.Target)) return true;
            state[id] = 2; return false;
        }
        if (nodes.Keys.Any(Visit)) diagnostics.Add(new("cycle", "The executable graph contains a cycle."));
        return new(diagnostics.Count == 0 ? prompt : null, diagnostics);
    }
    private static Dictionary<int, string> DynamicInputNames(GraphNode node, List<CompilationDiagnostic> diagnostics)
    {
        var names = new Dictionary<int, string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        bool createList = node.Type == "CreateList";
        // A constant StringFormat can have only its persisted widget and no connection slots.
        if (!createList && node.Data["inputs"] is null) return names;
        if (node.Data["inputs"] is not JsonArray slots)
        {
            diagnostics.Add(new("invalid_input_name", $"{node.Type} input ports must be an array of named slots.", node.Id));
            return names;
        }
        for (int slot = 0; slot < slots.Count; slot++)
        {
            if (slots[slot] is not JsonObject input || input["name"] is not JsonValue value ||
                !value.TryGetValue<string>(out var name) || !(createList
                    ? name.Length == 13 && name.StartsWith("inputs.input", StringComparison.Ordinal) && name[12] is >= '0' and <= '9'
                    : name == "f_string" || name.Length == 8 && name.StartsWith("values.", StringComparison.Ordinal) && name[7] is >= 'a' and <= 'z'))
            {
                string allowed = createList ? "inputs.input0 through inputs.input9" : "values.a through values.z or f_string";
                diagnostics.Add(new("invalid_input_name", $"{node.Type} input slot {slot} must be named {allowed}.", node.Id));
                continue;
            }
            if (!seen.Add(name))
            {
                diagnostics.Add(new("duplicate_input_name", $"{node.Type} has more than one input slot named {name}.", node.Id));
                continue;
            }
            names.Add(slot, name);
        }
        return names;
    }
    private static JsonNode? Literal(JsonNode? value) => value is JsonArray ? new JsonObject { ["__value__"] = value.DeepClone() } : value?.DeepClone();
}
