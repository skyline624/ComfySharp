using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Nodes;

/// <summary>Frozen nodes_string.py CaseConverter on valid Unicode scalar text.</summary>
public sealed class CaseConverterNode : IRuntimeNode
{
    public NodeSchema Schema { get; } = new("CaseConverter", "Convert Text Case", "text",
        [new("string", "STRING", Options: new() { ["multiline"] = true }),
         new("mode", "COMBO", Options: new() { ["multiselect"] = false,
             ["options"] = new JsonArray("UPPERCASE", "lowercase", "Capitalize", "Title Case") })],
        [new("STRING")], SearchAliases: ["case converter", "text case", "uppercase", "lowercase", "capitalize"],
        PythonModule: "comfy_extras.nodes_string", V3ObjectInfo: true);

    public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
        IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string text = ReadText(inputs, "string"), mode = ReadText(inputs, "mode");
        string result;
        switch (mode)
        {
            case "UPPERCASE": result = PythonUnicodeCase.Upper(text, cancellationToken); break;
            case "lowercase": result = PythonUnicodeCase.Lower(text, cancellationToken); break;
            case "Capitalize": result = PythonUnicodeCase.Capitalize(text, cancellationToken); break;
            case "Title Case": result = PythonUnicodeCase.Title(text, cancellationToken); break;
            default:
                // The source body returns the input for an unknown mode. The engine's
                // COMBO admission rejects that mode before invoking this body.
                PythonUnicodeLower.ValidateUnicode(text, cancellationToken);
                result = text;
                break;
        }
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(new NodeExecutionOutput([context.Json(JsonValue.Create(result))]));
    }

    private static string ReadText(IReadOnlyDictionary<string, RuntimeValue> inputs, string name)
    {
        if (!inputs.TryGetValue(name, out var value) || value.Kind != RuntimeValueKind.Json ||
            value.ToJson() is not JsonValue json || !json.TryGetValue<string>(out var text))
            throw new ArgumentException($"CaseConverter requires {name} text.");
        return text;
    }
}
