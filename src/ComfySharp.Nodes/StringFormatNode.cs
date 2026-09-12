using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Nodes;

/// <summary>Partial text-only port of nodes_string.py StringFormat at backend 1d48d9cf.</summary>
public sealed class StringFormatNode : IRuntimeNode
{
    public NodeSchema Schema { get; } = new("StringFormat", "Format Text", "text",
        Array.AsReadOnly(new[]
        {
            new InputSchema("values", "COMFY_AUTOGROW_V3", Autogrow: new AutogrowNamesTemplate(
                new("value", "*"), Enumerable.Range('a', 26).Select(c => ((char)c).ToString()).ToArray(), 0)),
            new InputSchema("f_string", "STRING", Options: new JsonObject { ["default"] = "{a}", ["multiline"] = true })
        }), Array.AsReadOnly(new[] { new OutputSchema("STRING") }),
        Description: "partial python-format-text-v1: named text, escaped braces, !s and Unicode text alignment/precision; null, Boolean and Int64 with empty spec. No full Python formatting, attributes, indexing, nested specs, floats or containers. Format 64Ki code points; width, precision and output 1Mi.",
        SearchAliases: Array.AsReadOnly(new[] { "string", "format" }),
        PythonModule: "comfy_extras.nodes_string", V3ObjectInfo: true);

    public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
        IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!inputs.TryGetValue("values", out var group) || group.Kind != RuntimeValueKind.Map)
            throw new ArgumentException("StringFormat requires its finalized values map.");
        if (!inputs.TryGetValue("f_string", out var value) || value.Kind != RuntimeValueKind.Json ||
            value.ToJson() is not JsonValue json || !json.TryGetValue<string>(out var format))
            throw new ArgumentException("StringFormat requires f_string text.");
        var result = PythonTextFormat.Format(format, group.Properties, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(new NodeExecutionOutput([context.Json(JsonValue.Create(result))]));
    }
}
