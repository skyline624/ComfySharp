using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Nodes;

/// <summary>nodes_toolkit.py CreateList: concatenate execution lists once, retaining their items.</summary>
public sealed class CreateListNode : IRuntimeNode
{
    public NodeSchema Schema { get; } = new("CreateList", "Create List", "utilities",
        Array.AsReadOnly(new[] { new InputSchema("inputs", "COMFY_AUTOGROW_V3", Autogrow: new(
            new("input", "COMFY_MATCHTYPE_V3", Options: new JsonObject { ["template"] = new JsonObject
            { ["template_id"] = "type", ["allowed_types"] = "*" } }), "input")) }),
        Array.AsReadOnly(new[] { new OutputSchema("COMFY_MATCHTYPE_V3", "list", IsList: true, MatchTemplate: "type") }),
        InputIsList: true, Description: "", SearchAliases: Array.AsReadOnly(new[] { "Image Iterator", "Text Iterator", "Iterator" }),
        PythonModule: "comfy_extras.nodes_toolkit", V3ObjectInfo: true);

    public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
        IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!inputs.TryGetValue("inputs", out var group) || group.Kind != RuntimeValueKind.Map)
            throw new ArgumentException("CreateList requires the finalized inputs map.");
        IEnumerable<RuntimeValue> Items()
        {
            foreach (var value in group.Properties.Values)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (value.Kind != RuntimeValueKind.List) throw new ArgumentException("CreateList inputs must be execution lists.");
                foreach (var item in value.Items) { cancellationToken.ThrowIfCancellationRequested(); yield return item; }
            }
        }
        // Context.List retains each child and rolls back already retained children if enumeration fails.
        var result = context.List(Items());
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(new NodeExecutionOutput([result]));
    }
}
