using System.Text.Json.Nodes;

namespace ComfySharp.Catalog;

/// <summary>Cross-checks source evidence documents. It does not establish runtime registration or numerical parity.</summary>
public static class NodeCatalogueEvidence
{
    public static IReadOnlyList<string> Validate(JsonObject manifest, JsonObject registrations, JsonObject schemas, JsonObject? staticResolutions = null)
    {
        var errors = new List<string>();
        var revision = Text(manifest["backendCommit"]);
        if (revision is null || revision.Length != 40 || !revision.All(Uri.IsHexDigit))
        { errors.Add("Backend commit must be a full Git SHA."); return errors; }
        foreach (var (label, document) in new[] { ("registration", registrations), ("schema", schemas) })
        {
            if (Number(document["schemaVersion"]) != 1 || Text(document["backendCommit"]) != revision)
                errors.Add($"Node {label} evidence version or backend commit mismatch.");
            if (document["unresolvedCases"] is not JsonArray)
                errors.Add($"Node {label} evidence must retain an unresolvedCases array.");
        }
        if (IsTrue(registrations["registrationComplete"]) && registrations["unresolvedCases"] is JsonArray { Count: > 0 })
            errors.Add("Registration completeness conflicts with unresolved cases.");
        CheckProvenance(registrations, "registrations");
        CheckProvenance(schemas, "schemas");
        errors.AddRange(NodeSchemaStaticResolution.ValidateAccounting(schemas));
        bool hasStaticResolutions = schemas.ContainsKey("staticResolution") ||
            (schemas["records"] as JsonArray)?.OfType<JsonObject>().Any(r => r.ContainsKey("resolvedFields")) == true;
        if (hasStaticResolutions)
        {
            if (staticResolutions is null) errors.Add("Missing reviewed static resolution plan.");
            else
            {
                CheckProvenance(staticResolutions, "staticResolutions");
                errors.AddRange(NodeSchemaStaticResolution.ValidateApplied(schemas, staticResolutions));
            }
        }
        else if (staticResolutions is not null) errors.Add("Static resolution plan supplied without applied evidence.");

        var modules = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
        var orders = new HashSet<int>();
        foreach (var module in Objects(registrations["modules"], "loader modules", errors))
        {
            var name = Text(module["module"]);
            var order = Number(module["loadOrder"]);
            if (name is null || !modules.TryAdd(name, module)) errors.Add("Missing or duplicate loader module.");
            if (order is null || order < 0 || !orders.Add(order.Value)) errors.Add($"Invalid or duplicate load order: {name}");
            if (module["conditions"] is not JsonArray) errors.Add($"Missing module conditions: {name}");
            CheckSource(module["source"], $"module {name}");
        }

        var records = Objects(registrations["records"], "registration records", errors).ToArray();
        var candidates = records.Where(r => Text(r["registrationState"]) == "builtin-candidate").ToArray();
        var registered = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
        foreach (var record in records)
        {
            var key = Key(record);
            var moduleName = Text(record["module"]);
            if (key is null || !registered.TryAdd(key, record)) errors.Add("Missing or duplicate registration identity.");
            var state = Text(record["registrationState"]);
            if (Text(record["scope"]) is not ("local" or "excluded-remote" or "reference-test"))
                errors.Add($"Invalid registration scope: {key}");
            if (state is not ("builtin-candidate" or "declaration-not-registered" or "external-custom-reference"))
                errors.Add($"Invalid registration state: {key}");
            if (state == "builtin-candidate")
            {
                if (moduleName is null || !modules.TryGetValue(moduleName, out var module))
                    errors.Add($"Registration has no loader module: {key}");
                else if (Number(record["loadOrder"]) != Number(module["loadOrder"]) || Text(record["scope"]) != Text(module["scope"]))
                    errors.Add($"Registration differs from its loader order or scope: {key}");
            }
            else if (record["loadOrder"] is not null) errors.Add($"Unregistered declaration has a loader order: {key}");
            if (record["conditions"] is not JsonArray) errors.Add($"Missing registration conditions: {key}");
            CheckSource(record["source"], $"registration {key}");
            CheckSource(record["classSource"], $"class {key}");
        }
        foreach (var (name, module) in modules)
            if (Number(module["candidateCount"]) != candidates.Count(r => Text(r["module"]) == name))
                errors.Add($"Loader candidate count does not match records: {name}");
        foreach (var (field, actual) in new[]
        {
            ("loaderModules", modules.Count),
            ("localLoaderModules", modules.Values.Count(m => Text(m["scope"]) == "local")),
            ("remoteLoaderModules", modules.Values.Count(m => Text(m["scope"]) == "excluded-remote")),
            ("localRegistrations", candidates.Count(r => Text(r["scope"]) == "local")),
            ("remoteRegistrations", candidates.Count(r => Text(r["scope"]) == "excluded-remote")),
            ("retainedUnregisteredDeclarations", records.Length - candidates.Length)
        })
            if (Number(registrations["coverage"]?[field]) != actual) errors.Add($"Registration coverage count is stale: {field}");

        var schemaRecords = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
        foreach (var record in Objects(schemas["records"], "schema records", errors))
        {
            var key = Key(record);
            if (key is null || !schemaRecords.TryAdd(key, record)) errors.Add("Missing or duplicate schema identity.");
            CheckSource(record["classSource"], $"schema class {key}");
            var status = Text(record["schemaStatus"]);
            if (status is not ("resolved-static" or "partial-static" or "unresolved")) errors.Add($"Invalid schema status: {key}");
            if (record["unresolvedFields"] is not JsonArray unresolved) errors.Add($"Missing unresolved schema fields: {key}");
            else if (status == "resolved-static" && unresolved.Count != 0) errors.Add($"Resolved schema contains unresolved expressions: {key}");
            if (record["schemaSources"] is not JsonArray sources || sources.Count == 0) errors.Add($"Missing schema provenance: {key}");
            else foreach (var source in sources) CheckSource(source, $"schema {key}");
            if (record["parameterSources"] is not JsonArray parameterSources) errors.Add($"Missing parameter provenance: {key}");
            else foreach (var parameter in parameterSources)
                if (parameter is JsonObject obj) CheckSource(obj["source"], $"parameter {Text(obj["path"])} of {key}");
                else errors.Add($"Invalid parameter provenance: {key}");
            // Every dynamic expression must keep its expression and exact source; do not replace it with a guessed literal.
            var dynamicCount = CheckExpressions(record, key ?? "<missing identity>");
            if (status == "resolved-static" && dynamicCount != 0) errors.Add($"Resolved schema contains dynamic values: {key}");
            if (IsTrue(schemas["schemaComplete"]) && (status != "resolved-static" || dynamicCount != 0))
                errors.Add($"Schema completeness conflicts with unresolved values: {key}");
        }
        if (IsTrue(schemas["schemaComplete"]) && schemas["unresolvedCases"] is JsonArray { Count: > 0 })
            errors.Add("Schema completeness conflicts with unresolved cases.");

        var capabilities = Objects(manifest["capabilities"], "manifest capabilities", errors).ToArray();
        foreach (var (key, registration) in registered)
        {
            if (Text(registration["scope"]) is "local" or "reference-test" && !schemaRecords.ContainsKey(key))
                errors.Add($"Local or reference node has no schema evidence: {key}");
            if (!capabilities.Any(c => Text(c["kind"]) == "node" && Text(c["scope"]) == Text(registration["scope"]) &&
                Text(c["name"]) == Text(registration["nodeId"]) && SourceModule(c["source"]) == Text(registration["module"])))
                errors.Add($"Node has no manifest capability in its declared scope: {key}");
        }
        return errors;

        void CheckSource(JsonNode? node, string owner)
        {
            if (SourceModule(node) is null) errors.Add($"Unpinned or invalid source for {owner}");
        }
        void CheckProvenance(JsonNode? node, string path)
        {
            if (node is JsonObject obj)
            {
                foreach (var (name, value) in obj)
                {
                    var childPath = $"{path}/{name}";
                    // Evidence includes nested constructors, shared io types and inherited definitions.
                    // A legacy input may itself be named "source" and contain an array/object, not provenance.
                    if (name.EndsWith("Source", StringComparison.Ordinal) || name == "source" && value is not (JsonArray or JsonObject))
                        CheckSource(value, childPath);
                    if (name is "sources" or "schemaSources" or "registrationSources")
                    {
                        if (value is not JsonArray array) errors.Add($"Invalid source list: {childPath}");
                        else foreach (var source in array) CheckSource(source, childPath);
                    }
                    CheckProvenance(value, childPath);
                }
            }
            else if (node is JsonArray array)
                for (int i = 0; i < array.Count; i++) CheckProvenance(array[i], $"{path}/{i}");
        }
        string? SourceModule(JsonNode? node)
        {
            if (!Uri.TryCreate(Text(node), UriKind.Absolute, out var uri) || uri.Scheme != "https" || uri.Host != "github.com" || uri.Query.Length != 0)
                return null;
            var prefix = $"/comfy-org/ComfyUI/blob/{revision}/";
            if (!uri.AbsolutePath.StartsWith(prefix, StringComparison.Ordinal) || uri.Fragment.Length < 3 ||
                !uri.Fragment.StartsWith("#L", StringComparison.Ordinal) || !int.TryParse(uri.Fragment.AsSpan(2), out var line) || line < 1) return null;
            return uri.AbsolutePath[prefix.Length..];
        }
        int CheckExpressions(JsonNode? node, string owner)
        {
            int count = 0;
            if (node is JsonObject obj)
            {
                if (Text(obj["resolution"]) == "unresolved")
                {
                    count++;
                    if (string.IsNullOrWhiteSpace(Text(obj["expression"]))) errors.Add($"Missing dynamic expression: {owner}");
                    CheckSource(obj["source"], $"dynamic expression in {owner}");
                }
                foreach (var property in obj) count += CheckExpressions(property.Value, owner);
            }
            else if (node is JsonArray array) foreach (var item in array) count += CheckExpressions(item, owner);
            return count;
        }
    }

    private static IEnumerable<JsonObject> Objects(JsonNode? node, string label, List<string> errors)
    {
        if (node is not JsonArray array || array.Count == 0) { errors.Add($"Missing or empty {label}."); yield break; }
        foreach (var item in array)
            if (item is JsonObject obj) yield return obj; else errors.Add($"Invalid object in {label}.");
    }
    private static string? Key(JsonObject row)
        => Text(row["nodeId"]) is { Length: > 0 } id && Text(row["module"]) is { Length: > 0 } module && Text(row["sourceClass"]) is { Length: > 0 } type
            ? $"{module}:{type}:{id}" : null;
    private static bool IsTrue(JsonNode? node) => node is JsonValue value && value.TryGetValue<bool>(out var result) && result;
    private static int? Number(JsonNode? node) => node is JsonValue value && value.TryGetValue<int>(out var result) ? result : null;
    private static string? Text(JsonNode? node) => node is JsonValue value && value.TryGetValue<string>(out var result) ? result : null;
}
