using System.Text.Json.Nodes;

namespace ComfySharp.Catalog;

/// <summary>Structural release gate; evidence claims still require independent hardware audit.</summary>
public static class ReleaseQualification
{
    public static IReadOnlyList<string> RequiredPlatforms { get; } = Array.AsReadOnly(new[]
    {
        "win-x64-cpu", "win-x64-cuda", "linux-x64-cpu", "linux-x64-cuda", "osx-arm64-cpu", "osx-arm64-mps"
    });

    public static IReadOnlyList<string> Validate(JsonObject manifest)
    {
        var errors = new List<string>();
        if (!IsTrue(manifest["catalogueComplete"])) errors.Add("Catalogue completeness has not been established, including dynamic/conditional registrations.");
        if (manifest["capabilities"] is not JsonArray capabilities || capabilities.Count == 0)
        {
            errors.Add("Capabilities must be a nonempty array.");
            return errors;
        }
        foreach (var item in capabilities)
        {
            if (item is not JsonObject row) { errors.Add("Capability must be an object."); continue; }
            if (Text(row["scope"]) != "local") continue;
            var platforms = row["platforms"] as JsonObject;
            var evidence = row["evidence"] as JsonArray;
            // Each platform needs an identifiable hardware run and its durable evidence reference.
            // An empty object, arbitrary JSON value, or another platform's run cannot qualify it.
            if (Text(row["implementation"]) != "implemented" || !IsTrue(row["unitTests"]) || !IsTrue(row["realWorkflow"]) ||
                RequiredPlatforms.Any(platform => !IsTrue(platforms?[platform]) || evidence is null ||
                    !evidence.Any(e => e is JsonObject run && Text(run["platform"]) == platform &&
                        !string.IsNullOrWhiteSpace(Text(run["hardware"])) &&
                        !string.IsNullOrWhiteSpace(Text(run["scenario"])) &&
                        !string.IsNullOrWhiteSpace(Text(run["artifact"])) && IsTrue(run["passed"]))))
                errors.Add($"V1 qualification incomplete: {Text(row["id"]) ?? "<missing id>"}");
        }
        return errors;
    }

    private static bool IsTrue(JsonNode? node) => node is JsonValue value && value.TryGetValue<bool>(out var result) && result;
    private static string? Text(JsonNode? node) => node is JsonValue value && value.TryGetValue<string>(out var result) ? result : null;
}
