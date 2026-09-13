using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Core;

// execution.py/get_input_data at the pinned backend revision. V3 metadata uses
// RuntimeHiddenContext, outside execute arguments. The live DynamicPrompt object
// still requires a separate contract and is never represented by a JSON substitute.
internal static class LegacyHiddenInputs
{
    public static void Validate(NodeSchema schema)
    {
        if(schema.V3HiddenInputs is {Count:>0} v3)
        {
            if(!schema.V3ObjectInfo||schema.HiddenInputs is {Count:>0})throw new ArgumentException("V3 context requires a V3 schema without legacy hidden arguments.");
            if(v3.Distinct(StringComparer.Ordinal).Count()!=v3.Count||v3.Any(k=>k is not("PROMPT" or "EXTRA_PNGINFO" or "UNIQUE_ID")))
                throw new NotSupportedException("Only distinct V3 prompt, PNG metadata and unique ID context fields are implemented.");
        }
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

    public static RuntimeHiddenContext V3Context(NodeSchema schema,JsonObject prompt,JsonObject? extraData,string nodeId)
        =>new(schema.V3HiddenInputs?.Contains("PROMPT")==true?prompt:null,
            schema.V3HiddenInputs?.Contains("EXTRA_PNGINFO")==true?extraData?["extra_pnginfo"]:null,
            schema.V3HiddenInputs?.Contains("UNIQUE_ID")==true?nodeId:null);

    public static JsonNode? Resolve(string type, JsonObject originalPrompt, JsonObject? extraData, string nodeId) => type switch
    {
        "PROMPT" => originalPrompt,
        "EXTRA_PNGINFO" => extraData?["extra_pnginfo"],
        "UNIQUE_ID" => JsonValue.Create(nodeId),
        _ => throw new NotSupportedException($"Hidden input kind '{type}' is not implemented.")
    };
}
