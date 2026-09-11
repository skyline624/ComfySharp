using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Nodes;
using ComfySharp.Catalog;

if (args.Length is < 1 or > 2 || args.Length == 2 && args[1] != "--release")
{
    Console.Error.WriteLine("Usage: ComfySharp.Catalog <manifest.json> [--release]");
    return 2;
}
try
{
    var manifest = JsonNode.Parse(File.ReadAllText(args[0]))!.AsObject();
    var capabilities = manifest["capabilities"]!.AsArray().Select(n => n!.AsObject()).ToArray();
    var errors = new List<string>();
    var ids = new HashSet<string>(StringComparer.Ordinal);
    var localNodes = new HashSet<string>(StringComparer.Ordinal);
    var registry = BuiltInNodes.CreateRegistry().ToObjectInfo();
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
    if (args.Length == 2) errors.AddRange(ReleaseQualification.Validate(manifest));
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        rows = capabilities.Length,
        local_node_declarations = capabilities.Count(c => c["scope"]!.GetValue<string>() == "local" && c["kind"]!.GetValue<string>() == "node"),
        registered_nodes = registry.Count,
        catalogue_complete = manifest["catalogueComplete"]!.GetValue<bool>(),
        release_ready = args.Length == 2 && errors.Count == 0,
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
