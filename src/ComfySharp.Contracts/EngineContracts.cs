using System.Text.Json.Nodes;

namespace ComfySharp.Contracts;

public sealed record InputSchema(string Name, string Type, bool Required = true, JsonObject? Options = null, bool Lazy = false);
public sealed record OutputSchema(string Type, string? Name = null, bool IsList = false, string? MatchTemplate = null);
public sealed record NodeSchema(string ClassType, string DisplayName, string Category,
    IReadOnlyList<InputSchema> Inputs, IReadOnlyList<OutputSchema> Outputs, bool OutputNode = false, bool InputIsList = false, bool Experimental = false,
    string? Description = null, IReadOnlyList<string>? SearchAliases = null, string? PythonModule = null);
public sealed record EngineDiagnostic(string Code, string Message, string? NodeId = null, string? InputName = null, string? TargetId = null);
public sealed record ValidationResult(IReadOnlyList<string> ValidTargets, IReadOnlyList<EngineDiagnostic> Diagnostics)
{
    public bool IsValid => ValidTargets.Count != 0;
}
public sealed record NodeExecutionIdentity(string NodeId, string DisplayNodeId, string? ParentNodeId = null);
public sealed record EngineEvent(string Type, string? NodeId = null, string? Message = null,
    JsonObject? Output = null, NodeExecutionIdentity? Identity = null);
public sealed record ExecutionResult(string Status,
    Dictionary<string, IReadOnlyList<IReadOnlyList<JsonNode?>>> Outputs, IReadOnlyList<EngineDiagnostic> Diagnostics);
/// <summary>Serializable UI snapshots from every executed node, independently of the selected native result slots.</summary>
public sealed record UiExecutionResult(string Status, Dictionary<string, JsonObject> Outputs,
    Dictionary<string, NodeExecutionIdentity> Meta, IReadOnlyList<EngineDiagnostic> Diagnostics);
/// <summary>Borrowed execution slots and an explicit UI document. The engine copies the UI on receipt.</summary>
public sealed record NodeExecutionOutput(IReadOnlyList<RuntimeValue> Result, JsonObject? Ui = null);

/// <summary>Inputs and returned values are borrowed for the invocation. Allocate resources through the supplied context.</summary>
public interface IRuntimeNode
{
    NodeSchema Schema { get; }
    ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
        IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken);
    /// <summary>Resolved values are execution lists, not literal arrays. All values are borrowed for this call.</summary>
    IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> resolvedInputs) => [];
}

/// <summary>JSON node adapter for the typed runtime. Each slot is a JSON value; IsList slots must hold a JsonArray.</summary>
public interface INode : IRuntimeNode
{
    ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> inputs, CancellationToken cancellationToken);
    /// <summary>Return lazy names needed by any mapped invocation. Resolved values are execution lists, not literal arrays.</summary>
    IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<JsonNode?>> resolvedInputs) => [];

    async ValueTask<NodeExecutionOutput> IRuntimeNode.ExecuteAsync(RuntimeNodeContext context,
        IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
    {
        var snapshot = inputs.ToDictionary(p => p.Key, p => p.Value.ToJson(), StringComparer.Ordinal);
        var returned = await ExecuteAsync(snapshot, cancellationToken);
        return new(returned.Select((value, index) => index < Schema.Outputs.Count && Schema.Outputs[index].IsList && value is JsonArray list
            ? context.List(list.Select(context.Json)) : context.Json(value)).ToArray());
    }
    IReadOnlyCollection<string> IRuntimeNode.GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> resolvedInputs) =>
        GetRequiredLazyInputs(resolvedInputs.ToDictionary(p => p.Key,
            p => (IReadOnlyList<JsonNode?>)p.Value.Select(v => v.ToJson()).ToArray(), StringComparer.Ordinal));
}
