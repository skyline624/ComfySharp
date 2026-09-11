using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;

namespace ComfySharp.Nodes;

/// <summary>
/// Ports from ComfyUI 1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a:
/// comfy_extras/nodes_primitive.py, nodes_string.py and nodes_logic.py.
/// No image/model node is registered until its implementation exists.
/// </summary>
public static class BuiltInNodes
{
    public static NodeRegistry CreateRegistry()
    {
        var registry = new NodeRegistry();
        void Add(string id, string display, string category, InputSchema[] inputs, OutputSchema[] outputs,
            Func<IReadOnlyDictionary<string, JsonNode?>, JsonNode?> execute) =>
            registry.Register(new FunctionNode(new(id, display, category, inputs, outputs), execute));
        foreach (var (id, display, type) in new[]
        {
            ("PrimitiveString", "Text", "STRING"), ("PrimitiveStringMultiline", "Text (Multiline)", "STRING"),
            ("PrimitiveInt", "Int", "INT"), ("PrimitiveFloat", "Float", "FLOAT"), ("PrimitiveBoolean", "Boolean", "BOOLEAN")
        })
        {
            var options = new JsonObject();
            if (type is "INT" or "FLOAT") { options["min"] = -long.MaxValue; options["max"] = long.MaxValue; }
            if (type == "FLOAT") options["step"] = 0.1;
            if (type == "INT") options["control_after_generate"] = "fixed";
            if (type == "STRING") options["multiline"] = id == "PrimitiveStringMultiline";
            Add(id, display, "utilities/primitive", [new("value", type, Options: options)], [new(type)], i => i["value"]?.DeepClone());
        }
        Add("StringConcatenate", "Concatenate Text", "text", [Text("string_a"), Text("string_b"), new("delimiter", "STRING", Options: new() { ["default"] = "", ["multiline"] = false })],
            [new("STRING")], i => JsonValue.Create(S(i, "string_a") + S(i, "delimiter") + S(i, "string_b")));
        Add("StringSubstring", "Substring", "text", [Text("string"), new("start", "INT"), new("end", "INT")], [new("STRING")], i =>
        {
            // Python indexes Unicode code points, while .NET string indexes UTF-16 code units.
            var runes = S(i, "string").EnumerateRunes().ToArray();
            int Bound(long n) => (int)Math.Clamp(n < 0 ? n + runes.Length : n, 0, runes.Length);
            var start = Bound(PythonValues.Integer(i["start"]));
            var end = Bound(PythonValues.Integer(i["end"]));
            return JsonValue.Create(string.Concat(runes.Skip(start).Take(Math.Max(0, end - start)).Select(r => r.ToString())));
        });
        Add("StringLength", "Text Length", "text", [Text("string")], [new("INT", "length")], i => JsonValue.Create(S(i, "string").EnumerateRunes().Count()));
        Add("StringReplace", "Replace Text", "text", [Text("string"), Text("find"), Text("replace")], [new("STRING")], i =>
        {
            var text = S(i, "string"); var find = S(i, "find"); var replacement = S(i, "replace");
            return JsonValue.Create(find.Length != 0 ? text.Replace(find, replacement, StringComparison.Ordinal)
                : replacement + string.Join(replacement, text.EnumerateRunes().Select(r => r.ToString())) + (text.Length == 0 ? "" : replacement));
        });
        Add("StringTrim", "Trim Text", "text", [Text("string"), new("mode", "COMBO", Options: new() { ["options"] = new JsonArray("Both", "Left", "Right") })], [new("STRING")], i =>
        {
            var text = S(i, "string"); var mode = S(i, "mode");
            int left = 0, right = text.Length;
            if (mode is "Both" or "Left") while (left < right && PythonWhitespace(text[left])) left++;
            if (mode is "Both" or "Right") while (right > left && PythonWhitespace(text[right - 1])) right--;
            return JsonValue.Create(text[left..right]);
        });
        Add("JsonExtractString", "Extract Text from JSON", "text", [Text("json_string"), new("key", "STRING", Options: new() { ["multiline"] = false })], [new("STRING")], i =>
        {
            try
            {
                using var document = JsonDocument.Parse(S(i, "json_string"));
                var root = document.RootElement;
                return JsonValue.Create(root.ValueKind == JsonValueKind.Object && root.TryGetProperty(S(i, "key"), out var value) && value.ValueKind != JsonValueKind.Null
                    ? PythonValues.String(MaterializeLastKeyWins(value)) : "");
            }
            catch (JsonException) { return JsonValue.Create(""); }
        });
        Add("ComfyNotNode", "Not", "utilities/logic", [new("value", "*")], [new("BOOLEAN")], i => JsonValue.Create(!PythonValues.Truth(i["value"])));
        registry.Register(new SwitchNode());
        return registry;
    }
    private static InputSchema Text(string name) => new(name, "STRING", Options: new() { ["multiline"] = true });
    private static string S(IReadOnlyDictionary<string, JsonNode?> inputs, string name) => inputs[name]!.GetValue<string>();
    private static bool PythonWhitespace(char c) => char.IsWhiteSpace(c) || c is >= '\u001c' and <= '\u001f';
    private static JsonNode? MaterializeLastKeyWins(JsonElement element)
    {
        // Python json.loads keeps the final value while preserving the key's first position.
        // Materialize recursively because nested objects may be stringified by PythonValues.
        if (element.ValueKind == JsonValueKind.Object)
        {
            var result = new JsonObject();
            foreach (var property in element.EnumerateObject()) result[property.Name] = MaterializeLastKeyWins(property.Value);
            return result;
        }
        if (element.ValueKind == JsonValueKind.Array)
            return new JsonArray(element.EnumerateArray().Select(MaterializeLastKeyWins).ToArray());
        return JsonNode.Parse(element.GetRawText());
    }
    private sealed class FunctionNode(NodeSchema schema, Func<IReadOnlyDictionary<string, JsonNode?>, JsonNode?> execute) : INode
    {
        public NodeSchema Schema { get; } = schema;
        public ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<JsonNode?>>([execute(inputs)]);
        }
    }
    private sealed class SwitchNode : IRuntimeNode
    {
        public NodeSchema Schema { get; } = new("ComfySwitchNode", "If/Else Switch", "utilities/logic",
            [new("switch", "BOOLEAN"), Match("on_false"), Match("on_true")], [new("COMFY_MATCHTYPE_V3", "output", MatchTemplate: "switch")], Experimental: true);
        private static InputSchema Match(string name) => new(name, "COMFY_MATCHTYPE_V3", Required: false,
            Options: new() { ["template"] = new JsonObject { ["template_id"] = "switch", ["allowed_types"] = "*" } }, Lazy: true);
        public IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> resolvedInputs) =>
            resolvedInputs["switch"].Select(v => PythonValues.Truth(v.ToJson()) ? "on_true" : "on_false").Distinct().ToArray();
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            inputs.TryGetValue(PythonValues.Truth(inputs["switch"].ToJson()) ? "on_true" : "on_false", out var selected);
            return ValueTask.FromResult(new NodeExecutionOutput([selected ?? context.Json(null)]));
        }
    }
}
