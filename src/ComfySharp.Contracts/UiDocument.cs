using System.Text.Json;
using System.Text.Json.Nodes;

namespace ComfySharp.Contracts;

/// <summary>Copies explicit JSON without invoking converters on arbitrary CLR or native objects.</summary>
public static class UiDocument
{
    public static JsonObject Snapshot(JsonObject document) => Copy(document)!.AsObject();

    private static JsonNode? Copy(JsonNode? node)
    {
        if (node is null) return null;
        if (node is JsonObject obj)
            return new JsonObject(obj.Select(p => new KeyValuePair<string, JsonNode?>(p.Key, Copy(p.Value))));
        if (node is JsonArray array) return new JsonArray(array.Select(Copy).ToArray());
        var value = ((JsonValue)node).GetValue<object>();
        // JsonElement is already a JSON document; no user-defined serializer runs here.
        if (value is JsonElement element) return JsonNode.Parse(element.GetRawText());
        if (value is string or bool or byte or sbyte or short or ushort or int or uint or long or ulong or decimal ||
            value is double number && double.IsFinite(number) || value is float single && float.IsFinite(single))
            return node.DeepClone();
        throw new InvalidOperationException("UI values must be JSON primitives, arrays and objects; arbitrary CLR objects and non-finite numbers are not supported.");
    }
}
