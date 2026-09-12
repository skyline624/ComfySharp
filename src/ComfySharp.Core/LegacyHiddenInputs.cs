using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Core;

// execution.py/get_input_data at the pinned backend revision. V3 hidden context and
// the live DynamicPrompt object require separate contracts; neither is a JSON substitute.
internal static class LegacyHiddenInputs
{
    public static void Validate(NodeSchema schema)
    {
        if (schema.HiddenInputs is not { Count: > 0 } hidden) return;
        if (schema.V3ObjectInfo || schema.Inputs.Any(i => i.Autogrow is not null))
            throw new NotSupportedException("Legacy hidden inputs cannot be used as V3 hidden context.");
        var names = schema.Inputs.Select(i => i.Name).ToHashSet(StringComparer.Ordinal);
        foreach (var input in hidden)
        {
            if (string.IsNullOrEmpty(input.Name) || !names.Add(input.Name))
                throw new ArgumentException("Hidden input names must be nonempty and distinct from visible inputs.");
            if (input.Type is not ("PROMPT" or "EXTRA_PNGINFO" or "UNIQUE_ID"))
                throw new NotSupportedException($"Hidden input kind '{input.Type}' is not implemented.");
        }
    }

    public static JsonNode? Resolve(string type, JsonObject originalPrompt, JsonObject? extraData, string nodeId) => type switch
    {
        "PROMPT" => originalPrompt,
        "EXTRA_PNGINFO" => extraData?["extra_pnginfo"],
        "UNIQUE_ID" => JsonValue.Create(nodeId),
        _ => throw new NotSupportedException($"Hidden input kind '{type}' is not implemented.")
    };
}
