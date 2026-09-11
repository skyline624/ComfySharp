using System.Text.Json.Nodes;

namespace ComfySharp.Contracts;

public sealed record InputSchema(string Name, string Type, bool Required = true, JsonObject? Options = null, bool Lazy = false);
public sealed record OutputSchema(string Type, string? Name = null, bool IsList = false, string? MatchTemplate = null);
public sealed record NodeSchema(string ClassType, string DisplayName, string Category,
    IReadOnlyList<InputSchema> Inputs, IReadOnlyList<OutputSchema> Outputs, bool OutputNode = false, bool InputIsList = false, bool Experimental = false);
public sealed record EngineDiagnostic(string Code, string Message, string? NodeId = null, string? InputName = null, string? TargetId = null);
public sealed record ValidationResult(IReadOnlyList<string> ValidTargets, IReadOnlyList<EngineDiagnostic> Diagnostics)
{
    public bool IsValid => ValidTargets.Count != 0;
}
public sealed record EngineEvent(string Type, string? NodeId = null, string? Message = null);
public sealed record ExecutionResult(string Status,
    Dictionary<string, IReadOnlyList<IReadOnlyList<JsonNode?>>> Outputs, IReadOnlyList<EngineDiagnostic> Diagnostics);

/// <summary>Each output slot is one JSON value; slots declared IsList must hold a JsonArray.</summary>
public interface INode
{
    NodeSchema Schema { get; }
    ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> inputs, CancellationToken cancellationToken);
    /// <summary>Return lazy names needed by any mapped invocation. Resolved values are execution lists, not literal arrays.</summary>
    IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<JsonNode?>> resolvedInputs) => [];
}
