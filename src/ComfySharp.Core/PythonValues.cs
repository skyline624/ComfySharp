using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace ComfySharp.Core;

/// <summary>Compatibility for JSON-domain scalar coercions; does not implement arbitrary Python objects.</summary>
public static class PythonValues
{
    public static bool Truth(JsonNode? node) => node switch
    {
        null => false,
        JsonArray a => a.Count != 0,
        JsonObject o => o.Count != 0,
        JsonValue v when v.TryGetValue<bool>(out var b) => b,
        JsonValue v when v.TryGetValue<string>(out var s) => s.Length != 0,
        _ => Float(node) != 0
    };
    public static long Integer(JsonNode? node)
    {
        if (node is null) throw new FormatException("Cannot convert null to INT.");
        if (node is JsonValue v && v.TryGetValue<bool>(out var b)) return b ? 1 : 0;
        if (node is JsonValue s && s.TryGetValue<string>(out var text)) return long.Parse(text, NumberStyles.Integer, CultureInfo.InvariantCulture);
        if (node is JsonValue i && i.TryGetValue<long>(out var integer)) return integer;
        return checked((long)Math.Truncate(Float(node)));
    }
    public static double Float(JsonNode? node)
    {
        if (node is null) throw new FormatException("Cannot convert null to FLOAT.");
        if (node is JsonValue v && v.TryGetValue<bool>(out var b)) return b ? 1 : 0;
        var text = node is JsonValue s && s.TryGetValue<string>(out var value) ? value : node.ToJsonString();
        var number = double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
        if (!double.IsFinite(number)) throw new FormatException("Non-finite numbers are unsupported in JSON execution.");
        return number;
    }
    public static string String(JsonNode? node) => node switch
    {
        null => "None",
        JsonValue v when v.TryGetValue<string>(out var s) => s,
        JsonValue v when v.TryGetValue<bool>(out var b) => b ? "True" : "False",
        JsonArray a => "[" + string.Join(", ", a.Select(Repr)) + "]",
        JsonObject o => "{" + string.Join(", ", o.Select(p => Repr(JsonValue.Create(p.Key)) + ": " + Repr(p.Value))) + "}",
        _ => node.ToJsonString()
    };
    private static string Repr(JsonNode? node) => node is JsonValue v && v.TryGetValue<string>(out var s)
        ? "'" + s.Replace("\\", "\\\\").Replace("'", "\\'").Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t") + "'" : String(node);
}
