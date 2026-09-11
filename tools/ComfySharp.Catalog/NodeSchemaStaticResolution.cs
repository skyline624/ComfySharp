using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace ComfySharp.Catalog;

/// <summary>A reviewed source-evidence patch, not a Python evaluator or a runtime node implementation.</summary>
public static class NodeSchemaStaticResolution
{
    public const string PlanFileName = "node-schema-resolutions.json";
    public const string BackendCommit = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a";
    public const string PlanSha256 = "ab1da52ef9eb4129f2d2cefa5858abc08a1fd360603e4d40c9fcfaea9ad8e045";
    private static readonly JsonSerializerOptions CanonicalOptions = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    // Property order is part of this deliberately narrow plan identity; whitespace and LF/CRLF are not.
    public static string PlanDigest(JsonObject plan) => Convert.ToHexStringLower(SHA256.HashData(
        Encoding.UTF8.GetBytes(plan.ToJsonString(CanonicalOptions))));

    /// <summary>Returns a new document. A stale, missing or already applied occurrence fails without changing either input.</summary>
    public static JsonObject Apply(JsonObject schemas, JsonObject plan)
    {
        RequirePlan(schemas, plan);
        var errors = ValidateAccounting(schemas);
        if (errors.Count != 0) throw new InvalidOperationException(string.Join("; ", errors));
        if (schemas.ContainsKey("staticResolution") || Records(schemas).Any(r => r.ContainsKey("resolvedFields")))
            throw new InvalidOperationException("Static resolution evidence is already present.");
        var copy = schemas.DeepClone().AsObject();
        var records = Records(copy).ToDictionary(Identity, StringComparer.Ordinal);
        var rules = Rules(plan);
        CheckSharedEnums(copy, rules.Values);
        foreach (var occurrence in Occurrences(plan))
        {
            if (!records.TryGetValue(Identity(occurrence), out var record))
                throw new InvalidOperationException("Missing static resolution schema identity.");
            string path = RequiredText(occurrence, "path");
            var expected = Original(occurrence);
            if (!JsonNode.DeepEquals(At(record, path), expected))
                throw new InvalidOperationException($"Stale static resolution target: {Identity(record)}/{path}");
            Set(record, path, rules[RequiredText(occurrence, "rule")]["value"]!.DeepClone());
            var fields = record["unresolvedFields"]!.AsArray();
            var mirror = fields.Single(f => f?["path"]?.GetValue<string>() == path);
            fields.Remove(mirror);
            if (record["resolvedFields"] is null) record["resolvedFields"] = new JsonArray();
            record["resolvedFields"]!.AsArray().Add(Resolved(occurrence));
        }
        Recount(copy);
        copy["staticResolution"] = Marker();
        var afterErrors = ValidateApplied(copy, plan);
        if (afterErrors.Count != 0) throw new InvalidOperationException(string.Join("; ", afterErrors));
        return copy;
    }

    /// <summary>Checks the applied values and provenance against the exact reviewed plan; does not fetch source blobs.</summary>
    public static IReadOnlyList<string> ValidateApplied(JsonObject schemas, JsonObject plan)
    {
        var errors = new List<string>();
        try
        {
            RequirePlan(schemas, plan);
            errors.AddRange(ValidateAccounting(schemas));
            if (!JsonNode.DeepEquals(schemas["staticResolution"], Marker())) errors.Add("Static resolution marker mismatch.");
            var records = Records(schemas).ToDictionary(Identity, StringComparer.Ordinal);
            var rules = Rules(plan);
            CheckSharedEnums(schemas, rules.Values);
            int total = 0;
            foreach (var record in records.Values)
            {
                if (record["resolvedFields"] is JsonArray fields) total += fields.Count;
                else if (record.ContainsKey("resolvedFields")) errors.Add("Invalid resolved fields array.");
            }
            if (total != 55) errors.Add("Static resolution evidence must contain exactly 55 occurrences.");
            foreach (var occurrence in Occurrences(plan))
            {
                if (!records.TryGetValue(Identity(occurrence), out var record))
                { errors.Add("Missing resolved schema identity."); continue; }
                string path = RequiredText(occurrence, "path");
                if (!JsonNode.DeepEquals(At(record, path), rules[RequiredText(occurrence, "rule")]["value"]))
                    errors.Add($"Resolved value differs from the reviewed declaration: {Identity(record)}/{path}");
                var evidence = (record["resolvedFields"] as JsonArray)?.Where(f => f?["path"]?.GetValue<string>() == path).ToArray();
                if (evidence is not { Length: 1 } || !JsonNode.DeepEquals(evidence[0], Resolved(occurrence)))
                    errors.Add($"Missing or altered static resolution provenance: {Identity(record)}/{path}");
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or KeyNotFoundException or IndexOutOfRangeException or FormatException or OverflowException)
        { errors.Add($"Invalid static resolution evidence: {ex.Message}"); }
        return errors;
    }

    /// <summary>Counts canonical input/output/metadata locations once and verifies their summaries.</summary>
    public static IReadOnlyList<string> ValidateAccounting(JsonObject schemas)
    {
        var errors = new List<string>();
        try
        {
            var records = Records(schemas).ToArray();
            var identities = new HashSet<string>(StringComparer.Ordinal);
            var statuses = new Dictionary<string, int>(StringComparer.Ordinal);
            var summaries = new JsonArray();
            int total = 0;
            foreach (var record in records)
            {
                string identity = Identity(record);
                if (!identities.Add(identity)) errors.Add("Duplicate schema identity in accounting.");
                var actual = Unresolved(record);
                total += actual.Count;
                if (record["unresolvedFields"] is not JsonArray mirrors || mirrors.Count != actual.Count)
                    errors.Add($"Unresolved field count differs from canonical values: {identity}");
                else
                {
                    var paths = new HashSet<string>(StringComparer.Ordinal);
                    foreach (var item in mirrors)
                    {
                        if (item is not JsonObject mirror || Text(mirror["path"]) is not { } path || !paths.Add(path) ||
                            !actual.TryGetValue(path, out var value))
                        { errors.Add($"Missing, duplicate or stale unresolved field path: {identity}"); continue; }
                        var expected = value.DeepClone().AsObject(); expected["path"] = path;
                        if (!JsonNode.DeepEquals(mirror, expected)) errors.Add($"Unresolved field mirror differs: {identity}/{path}");
                    }
                }
                string status = RequiredText(record, "schemaStatus");
                if (status is not ("resolved-static" or "partial-static" or "unresolved"))
                    errors.Add($"Invalid schema status in accounting: {identity}");
                if (actual.Count == 0 && status != "resolved-static" || actual.Count > 0 && status == "resolved-static")
                    errors.Add($"Schema status differs from canonical unresolved values: {identity}");
                statuses[status] = statuses.GetValueOrDefault(status) + 1;
                if (actual.Count > 0) summaries.Add(Summary(record, actual.Count));
                if (record["parameterSources"] is JsonArray parameters)
                    foreach (var item in parameters)
                    {
                        if (item is not JsonObject parameter || Text(parameter["path"]) is not { } path)
                        { errors.Add($"Invalid parameter source path: {identity}"); continue; }
                        var value = At(record, path);
                        if (value is null) { errors.Add($"Missing parameter source target: {identity}/{path}"); continue; }
                        string expected = HasUnresolved(value) ? "partial-static" : "resolved-static";
                        if (Text(parameter["resolution"]) != expected) errors.Add($"Parameter resolution differs from its value: {identity}/{path}");
                    }
                else errors.Add($"Missing or invalid parameterSources array: {identity}");
            }
            if (!JsonNode.DeepEquals(schemas["unresolvedCases"], summaries)) errors.Add("Schema unresolvedCases summary is stale.");
            if (total > 0 && schemas["schemaComplete"] is JsonValue complete && complete.TryGetValue<bool>(out bool claimed) && claimed)
                errors.Add("Schema completeness conflicts with canonical unresolved values.");
            var coverage = schemas["coverage"] as JsonObject;
            if (coverage?["localNodeRecords"]?.GetValue<int>() != records.Length ||
                coverage?["unresolvedExpressionCount"]?.GetValue<int>() != total)
                errors.Add("Schema coverage count is stale.");
            var expectedStatuses = new JsonObject();
            foreach (var pair in statuses) expectedStatuses[pair.Key] = pair.Value;
            if (!JsonNode.DeepEquals(coverage?["statuses"], expectedStatuses)) errors.Add("Schema coverage statuses are stale.");
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or KeyNotFoundException or IndexOutOfRangeException or FormatException or OverflowException)
        { errors.Add($"Invalid schema accounting: {ex.Message}"); }
        return errors;
    }

    private static void RequirePlan(JsonObject schemas, JsonObject plan)
    {
        if (Text(schemas["backendCommit"]) != BackendCommit || Text(plan["backendCommit"]) != BackendCommit || PlanDigest(plan) != PlanSha256)
            throw new InvalidOperationException("Static resolution plan identity or backend revision mismatch.");
        if (schemas["schemaVersion"]?.GetValue<int>() != 1 ||
            schemas["schemaComplete"] is not JsonValue schemaComplete || !schemaComplete.TryGetValue<bool>(out bool schemaClaim) || schemaClaim ||
            schemas["catalogueComplete"] is not JsonValue catalogueComplete || !catalogueComplete.TryGetValue<bool>(out bool catalogueClaim) || catalogueClaim)
            throw new InvalidOperationException("The reviewed static resolution slice requires an incomplete schema and catalogue.");
    }

    private static void CheckSharedEnums(JsonObject schemas, IEnumerable<JsonObject> rules)
    {
        foreach (var rule in rules)
        {
            if (Text(rule["sharedContract"]) is not { } name) continue;
            var definitions = schemas["sharedContract"]?["definitions"] as JsonArray
                ?? throw new InvalidOperationException("Missing shared IO contract.");
            var definition = definitions.OfType<JsonObject>().Single(d => Text(d["name"]) == name);
            if (!JsonNode.DeepEquals(definition["source"], rule["definitionSource"]) ||
                !JsonNode.DeepEquals(definition["enumValues"]?[RequiredText(rule, "member")], rule["value"]))
                throw new InvalidOperationException("Shared IO enum differs from the reviewed declaration.");
        }
    }

    private static Dictionary<string, JsonObject> Rules(JsonObject plan) => plan["rules"]!.AsArray().OfType<JsonObject>()
        .ToDictionary(r => RequiredText(r, "id"), StringComparer.Ordinal);
    private static IEnumerable<JsonObject> Occurrences(JsonObject plan) => plan["occurrences"]!.AsArray().OfType<JsonObject>();
    private static IEnumerable<JsonObject> Records(JsonObject schemas)
    {
        var records = schemas["records"] as JsonArray ?? throw new InvalidOperationException("Missing schema records.");
        foreach (var record in records) yield return record as JsonObject ?? throw new InvalidOperationException("Invalid schema record.");
    }
    private static JsonObject Marker() => new() { ["planId"] = "source-static-55-v1", ["planSha256"] = PlanSha256, ["occurrences"] = 55 };
    private static string Identity(JsonObject record) => string.Join(":", new[] { "module", "sourceClass", "nodeId" }.Select(k => RequiredText(record, k)));
    private static string RequiredText(JsonObject obj, string key) => Text(obj[key]) ?? throw new InvalidOperationException($"Missing {key}.");
    private static string? Text(JsonNode? value) => value is JsonValue scalar && scalar.TryGetValue<string>(out var text) ? text : null;
    private static JsonObject Original(JsonObject occurrence) => new()
    {
        ["resolution"] = "unresolved", ["expression"] = occurrence["expression"]!.DeepClone(),
        ["source"] = occurrence["source"]!.DeepClone(), ["reason"] = occurrence["reason"]!.DeepClone()
    };
    private static JsonObject Resolved(JsonObject occurrence) => new()
    {
        ["path"] = occurrence["path"]!.DeepClone(), ["resolution"] = "resolved-static",
        ["expression"] = occurrence["expression"]!.DeepClone(), ["source"] = occurrence["source"]!.DeepClone(),
        ["reason"] = occurrence["reason"]!.DeepClone(), ["rule"] = occurrence["rule"]!.DeepClone()
    };
    private static JsonObject Summary(JsonObject record, int count) => new()
    {
        ["nodeId"] = record["nodeId"]!.DeepClone(), ["module"] = record["module"]!.DeepClone(),
        ["fields"] = count, ["status"] = record["schemaStatus"]!.DeepClone()
    };
    private static JsonNode? At(JsonObject record, string path)
    {
        JsonNode? node = record;
        foreach (var part in path.Split('/')) node = node switch
        {
            JsonObject obj => obj[part],
            JsonArray array => array[int.Parse(part, CultureInfo.InvariantCulture)],
            _ => throw new InvalidOperationException("Invalid schema path.")
        };
        return node;
    }
    private static void Set(JsonObject record, string path, JsonNode value)
    {
        int split = path.LastIndexOf('/');
        if (split < 0) throw new InvalidOperationException("Expected a nested schema path.");
        var parent = At(record, path[..split]); string name = path[(split + 1)..];
        if (parent is JsonObject obj) obj[name] = value;
        else if (parent is JsonArray array) array[int.Parse(name, CultureInfo.InvariantCulture)] = value;
        else throw new InvalidOperationException("Invalid schema path parent.");
    }
    private static bool HasUnresolved(JsonNode? node) => node switch
    {
        JsonObject obj => Text(obj["resolution"]) == "unresolved" || obj.Any(p => HasUnresolved(p.Value)),
        JsonArray array => array.Any(HasUnresolved), _ => false
    };
    private static Dictionary<string, JsonObject> Unresolved(JsonObject record)
    {
        var result = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
        foreach (string field in new[] { "inputs", "outputs", "metadata" }) Walk(record[field], field);
        return result;
        void Walk(JsonNode? node, string path)
        {
            if (node is JsonObject obj)
            {
                if (Text(obj["resolution"]) == "unresolved") result.Add(path, obj);
                foreach (var pair in obj) Walk(pair.Value, $"{path}/{pair.Key}");
            }
            else if (node is JsonArray array)
                for (int i = 0; i < array.Count; i++) Walk(array[i], $"{path}/{i}");
        }
    }
    private static void Recount(JsonObject schemas)
    {
        var statuses = new JsonObject(); var summaries = new JsonArray(); int count = 0, total = 0;
        foreach (var record in Records(schemas))
        {
            count++; int pending = Unresolved(record).Count; total += pending;
            string status = pending == 0 ? "resolved-static" : "partial-static"; record["schemaStatus"] = status;
            statuses[status] = (statuses[status]?.GetValue<int>() ?? 0) + 1;
            if (pending > 0) summaries.Add(Summary(record, pending));
            foreach (var parameter in record["parameterSources"]!.AsArray().OfType<JsonObject>())
                parameter["resolution"] = HasUnresolved(At(record, RequiredText(parameter, "path"))) ? "partial-static" : "resolved-static";
        }
        schemas["coverage"] = new JsonObject { ["localNodeRecords"] = count, ["statuses"] = statuses, ["unresolvedExpressionCount"] = total };
        schemas["unresolvedCases"] = summaries;
    }
}
