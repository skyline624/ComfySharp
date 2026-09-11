using System.Text.Json.Nodes;

namespace ComfySharp.Host;

internal static class RequestFields
{
    public static T? Optional<T>(JsonObject request, string name)
    {
        if (!request.ContainsKey(name)) return default;
        if (request[name] is JsonValue value && value.TryGetValue<T>(out var result)) return result;
        throw new InvalidDataException($"{name} must be a {typeof(T).Name}.");
    }

    public static string[]? Strings(JsonObject request, string name)
    {
        if (!request.ContainsKey(name)) return null;
        if (request[name] is not JsonArray array) throw new InvalidDataException($"{name} must be an array of strings.");
        return array.Select((item, index) => item is JsonValue value && value.TryGetValue<string>(out var text)
            ? text : throw new InvalidDataException($"{name}[{index}] must be a string.")).ToArray();
    }

    public static JsonObject? Object(JsonObject request, string name)
    {
        if (!request.ContainsKey(name)) return null;
        return request[name] as JsonObject ?? throw new InvalidDataException($"{name} must be an object.");
    }
}
