using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Nodes;

/// <summary>Port of the frozen nodes_string.py StringContains for valid Unicode scalar text.</summary>
public sealed class StringContainsNode : IRuntimeNode
{
    public NodeSchema Schema { get; } = new("StringContains", "Contains Text", "text",
        [TextComparisonInputs.Text("string"), TextComparisonInputs.Text("substring"), TextComparisonInputs.CaseSensitive()],
        [new("BOOLEAN", "contains")],
        SearchAliases: ["contains", "text includes", "string includes"],
        PythonModule: "comfy_extras.nodes_string", V3ObjectInfo: true);

    public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
        IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
    {
        var (text, substring) = TextComparisonInputs.Prepare(inputs, "string", "substring", cancellationToken);
        bool result = text.Contains(substring, StringComparison.Ordinal);
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(new NodeExecutionOutput([context.Json(JsonValue.Create(result))]));
    }
}

/// <summary>Port of the frozen nodes_string.py StringCompare for valid Unicode scalar text.</summary>
public sealed class StringCompareNode : IRuntimeNode
{
    public NodeSchema Schema { get; } = new("StringCompare", "Compare Text", "text",
        [TextComparisonInputs.Text("string_a"), TextComparisonInputs.Text("string_b"),
            new("mode", "COMBO", Options: new() { ["multiselect"] = false, ["options"] = new JsonArray("Starts With", "Ends With", "Equal") }),
            TextComparisonInputs.CaseSensitive()], [new("BOOLEAN")],
        SearchAliases: ["compare", "text match", "string equals", "starts with", "ends with"],
        PythonModule: "comfy_extras.nodes_string", V3ObjectInfo: true);

    public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
        IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
    {
        var (a, b) = TextComparisonInputs.Prepare(inputs, "string_a", "string_b", cancellationToken);
        string mode = TextComparisonInputs.ReadText(inputs, "mode");
        bool result = mode switch
        {
            "Equal" => string.Equals(a, b, StringComparison.Ordinal),
            "Starts With" => a.StartsWith(b, StringComparison.Ordinal),
            "Ends With" => a.EndsWith(b, StringComparison.Ordinal),
            _ => throw new ArgumentException("StringCompare mode must be Starts With, Ends With or Equal.")
        };
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(new NodeExecutionOutput([context.Json(JsonValue.Create(result))]));
    }
}

internal static class TextComparisonInputs
{
    internal static InputSchema Text(string name) => new(name, "STRING", Options: new() { ["multiline"] = true });
    internal static InputSchema CaseSensitive() => new("case_sensitive", "BOOLEAN",
        Options: new() { ["default"] = true, ["advanced"] = true });

    internal static (string A, string B) Prepare(IReadOnlyDictionary<string, RuntimeValue> inputs,
        string first, string second, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string a = ReadText(inputs, first), b = ReadText(inputs, second);
        if (!inputs.TryGetValue("case_sensitive", out var value) || value.Kind != RuntimeValueKind.Json ||
            value.ToJson() is not JsonValue json || !json.TryGetValue<bool>(out bool sensitive))
            throw new ArgumentException("Text comparison requires case_sensitive Boolean.");
        if (!sensitive)
            return (PythonUnicodeLower.Lower(a, cancellationToken), PythonUnicodeLower.Lower(b, cancellationToken));
        PythonUnicodeLower.ValidateUnicode(a, cancellationToken);
        PythonUnicodeLower.ValidateUnicode(b, cancellationToken);
        return (a, b);
    }

    internal static string ReadText(IReadOnlyDictionary<string, RuntimeValue> inputs, string name)
    {
        if (!inputs.TryGetValue(name, out var value) || value.Kind != RuntimeValueKind.Json ||
            value.ToJson() is not JsonValue json || !json.TryGetValue<string>(out var text))
            throw new ArgumentException($"Text comparison requires {name} text.");
        return text;
    }
}
