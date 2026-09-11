using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;

namespace ComfySharp.Host;

/// <summary>The HTTP contract selects real output nodes. Internal tools can still target any node through EngineService.</summary>
public sealed record PromptSubmission(ValidationResult Validation, JsonObject? Error, JsonObject NodeErrors)
{
    // https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L1133
    public static PromptSubmission Validate(EngineService engine, JsonObject prompt, IReadOnlyCollection<string>? requestedTargets)
    {
        var targets = new List<string>();
        var filter = requestedTargets?.ToHashSet(StringComparer.Ordinal);
        foreach (var (id, value) in prompt)
        {
            var node = value as JsonObject;
            var type = String(node?["class_type"]);
            var title = node?["_meta"] is JsonObject meta ? String(meta["title"]) : null;
            if (type is null || !engine.Registry.TryGet(type, out var implementation))
            {
                var error = ErrorDocument("missing_node_type", type is null
                    ? $"Node '{title ?? $"ID #{id}"}' has no class_type. The workflow may be corrupted or a custom node is missing."
                    : $"Node '{title ?? type}' not found. The custom node may not be installed.", $"Node ID '#{id}'");
                error["extra_info"] = new JsonObject { ["node_id"] = id, ["class_type"] = type, ["node_title"] = title ?? type };
                return new(new([], []), error, []);
            }
            if (implementation.Schema.OutputNode && (filter is null || filter.Contains(id))) targets.Add(id);
        }
        if (targets.Count == 0)
            return new(new([], []), ErrorDocument("prompt_no_outputs", "Prompt has no outputs"), []);
        var validation = engine.Validate(prompt, targets);
        var nodeErrors = new JsonObject();
        foreach (var group in validation.Diagnostics.Where(d => d.NodeId is not null).GroupBy(d => d.NodeId!, StringComparer.Ordinal))
        {
            var details = new JsonArray();
            foreach (var diagnostic in group.DistinctBy(d => (d.Code, d.Message, d.InputName)))
            {
                var detail = ErrorDocument(diagnostic.Code, diagnostic.Message);
                if (diagnostic.InputName is not null) detail["extra_info"] = new JsonObject { ["input_name"] = diagnostic.InputName };
                details.Add(detail);
            }
            nodeErrors[group.Key] = new JsonObject
            {
                ["errors"] = details,
                ["dependent_outputs"] = new JsonArray(group.Select(d => d.TargetId).Where(t => t is not null).Distinct(StringComparer.Ordinal).Select(t => (JsonNode?)JsonValue.Create(t)).ToArray()),
                ["class_type"] = prompt[group.Key] is JsonObject node ? node["class_type"]?.DeepClone() : null
            };
        }
        return new(validation, validation.IsValid ? null : ErrorDocument("prompt_outputs_failed_validation", "Prompt outputs failed validation"), nodeErrors);
    }

    private static JsonObject ErrorDocument(string type, string message, string details = "") => new()
    { ["type"] = type, ["message"] = message, ["details"] = details, ["extra_info"] = new JsonObject() };
    private static string? String(JsonNode? node) => node is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;
}
