using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Nodes;
using ComfySharp.Nodes.Tensor;
using ComfySharp.Catalog;
using ComfySharp.Core;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: ComfySharp.Catalog <manifest.json> [--release] [--node-evidence <registrations.json> <schemas.json>]");
    return 2;
}
try
{
    bool release = false;
    string? registrationPath = null, schemaPath = null;
    for (int index = 1; index < args.Length; index++)
    {
        if (args[index] == "--release" && !release) release = true;
        else if (args[index] == "--node-evidence" && registrationPath is null && index + 2 < args.Length)
        { registrationPath = args[++index]; schemaPath = args[++index]; }
        else throw new InvalidOperationException("Invalid or repeated catalogue command argument.");
    }
    var manifest = JsonNode.Parse(File.ReadAllText(args[0]))!.AsObject();
    var capabilities = manifest["capabilities"]!.AsArray().Select(n => n!.AsObject()).ToArray();
    var errors = new List<string>();
    var ids = new HashSet<string>(StringComparer.Ordinal);
    var localNodes = new HashSet<string>(StringComparer.Ordinal);
    // The Host configures its local image store. Describe these service-backed nodes without opening files in this audit tool.
    var registry = NodeRegistry.Describe(TensorNodes.CreateRegistry().Nodes.Select(n => n.Schema)
        .Concat(ImageFileNodes.Schemas).Concat(Sd15Nodes.Schemas([])).Concat(LoraNodes.Schemas([])).Append(ImageInputNodes.Describe([])).Append(SaveLoraNode.Description));
    foreach (var row in capabilities)
    {
        var id = row["id"]!.GetValue<string>();
        if (!ids.Add(id)) errors.Add($"Duplicate capability id: {id}");
        if (!Uri.TryCreate(row["source"]!.GetValue<string>(), UriKind.Absolute, out var source) || source.Scheme != "https" ||
            source.Host != "github.com" || !source.AbsolutePath.Contains(manifest["backendCommit"]!.GetValue<string>(), StringComparison.Ordinal))
            errors.Add($"Source is not pinned to the backend revision: {id}");
        if (string.IsNullOrWhiteSpace(row["scenario"]?.GetValue<string>())) errors.Add($"Missing scenario: {id}");
        if (row["scope"]!.GetValue<string>() != "local") continue;
        if (row["kind"]!.GetValue<string>() == "node")
        {
            var name = row["name"]!.GetValue<string>();
            localNodes.Add(name);
            if (row["implementation"]!.GetValue<string>() == "implemented" && !registry.ContainsKey(name))
                errors.Add($"Node claimed implemented but absent from runtime registry: {name}");
        }
    }
    foreach (var (name, _) in registry)
        if (!localNodes.Contains(name)) errors.Add($"Runtime node missing from manifest: {name}");
    if (registrationPath is not null)
    {
        var schemas = JsonNode.Parse(File.ReadAllText(schemaPath!))!.AsObject();
        JsonObject? staticResolutions = null;
        if (schemas.ContainsKey("staticResolution") ||
            (schemas["records"] as JsonArray)?.OfType<JsonObject>().Any(r => r.ContainsKey("resolvedFields")) == true)
            staticResolutions = JsonNode.Parse(File.ReadAllText(Path.Combine(
                Path.GetDirectoryName(Path.GetFullPath(schemaPath!))!, NodeSchemaStaticResolution.PlanFileName)))!.AsObject();
        errors.AddRange(NodeCatalogueEvidence.Validate(manifest,
            JsonNode.Parse(File.ReadAllText(registrationPath))!.AsObject(), schemas, staticResolutions));
    }
    if (release) errors.AddRange(ReleaseQualification.Validate(manifest));
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        rows = capabilities.Length,
        local_node_declarations = capabilities.Count(c => c["scope"]!.GetValue<string>() == "local" && c["kind"]!.GetValue<string>() == "node"),
        registered_nodes = registry.Count,
        catalogue_complete = manifest["catalogueComplete"]!.GetValue<bool>(),
        node_evidence_checked = registrationPath is not null,
        release_ready = release && errors.Count == 0,
        error_count = errors.Count,
        errors = errors.Take(30)
    }, new JsonSerializerOptions { WriteIndented = true }));
    return errors.Count == 0 ? 0 : 1;
}
catch (Exception exception) when (exception is IOException or JsonException or InvalidOperationException or NullReferenceException)
{
    Console.Error.WriteLine($"Invalid manifest: {exception.Message}");
    return 2;
}
