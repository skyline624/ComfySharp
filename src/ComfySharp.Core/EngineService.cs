using System.Globalization;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Core;

/// <summary>JSON-only DAG executor. Per-run memoization never owns native handles or outlives a job.</summary>
public sealed class EngineService(NodeRegistry registry)
{
    public NodeRegistry Registry { get; } = registry;

    public ValidationResult Validate(JsonObject prompt, IReadOnlyCollection<string>? targets = null)
    {
        var diagnostics = new List<EngineDiagnostic>();
        var valid = new List<string>();
        var selected = targets ?? prompt.Where(p => p.Value is JsonObject n &&
            ReadString(n["class_type"]) is { } type && Registry.TryGet(type, out var impl) && impl.Schema.OutputNode)
            .Select(p => p.Key).ToArray();
        if (selected.Count == 0) diagnostics.Add(new("no_outputs", "No output nodes selected. Supply explicit targets for scalar nodes."));
        foreach (var target in selected.Distinct(StringComparer.Ordinal))
        {
            var visited = new HashSet<string>(StringComparer.Ordinal);
            var active = new HashSet<string>(StringComparer.Ordinal);
            var count = diagnostics.Count;
            void Error(string code, string message, string id, string? input = null) => diagnostics.Add(new(code, message, id, input, target));
            void Visit(string id)
            {
                if (active.Contains(id)) { Error("cycle", "Dependency cycle detected.", id); return; }
                if (!visited.Add(id)) return;
                if (!prompt.TryGetPropertyValue(id, out var raw)) { Error("missing_node", "Referenced node does not exist.", id); return; }
                if (raw is not JsonObject n || ReadString(n["class_type"]) is not { } type)
                { Error("invalid_node", "Node must contain a string class_type.", id); return; }
                if (!Registry.TryGet(type, out var node)) { Error("unknown_node", $"Node type '{type}' is not implemented.", id); return; }
                if (n["inputs"] is not JsonObject inputs) { Error("invalid_inputs", "Node inputs must be an object.", id); return; }
                active.Add(id);
                foreach (var input in node.Schema.Inputs)
                {
                    if (!inputs.TryGetPropertyValue(input.Name, out var value))
                    {
                        if (input.Required) Error("required_input_missing", "Required input is missing.", id, input.Name);
                        continue;
                    }
                    if (value is JsonArray)
                    {
                        if (!TryLink(value, out var source, out var index)) { Error("invalid_link", "Link must be [string nodeId, non-negative integer outputIndex]; wrap literal arrays in __value__.", id, input.Name); continue; }
                        Visit(source);
                        if (prompt[source] is JsonObject upstream && ReadString(upstream["class_type"]) is { } upstreamType && Registry.TryGet(upstreamType, out var producer))
                        {
                            if (index >= producer.Schema.Outputs.Count) Error("invalid_output_index", "Source output index is out of range.", id, input.Name);
                            else if (!Compatible(producer.Schema.Outputs[index].Type, input.Type)) Error("type_mismatch", $"Cannot connect {producer.Schema.Outputs[index].Type} to {input.Type}.", id, input.Name);
                        }
                    }
                    else
                    {
                        try { Normalize(Unwrap(value), input); }
                        catch (Exception e) when (e is FormatException or InvalidOperationException or OverflowException)
                        { Error("invalid_input_type", e.Message, id, input.Name); }
                    }
                }
                active.Remove(id);
            }
            Visit(target);
            if (diagnostics.Count == count) valid.Add(target);
        }
        return new(valid, diagnostics);
    }

    public async Task<ExecutionResult> ExecuteAsync(JsonObject prompt, IReadOnlyCollection<string>? targets = null,
        Func<EngineEvent, ValueTask>? onEvent = null, CancellationToken cancellationToken = default)
    {
        // Freeze caller-owned JSON before the first await, so validation and execution see the same graph.
        prompt = prompt.DeepClone().AsObject();
        var validation = Validate(prompt, targets);
        var diagnostics = validation.Diagnostics.ToList();
        var results = new Dictionary<string, IReadOnlyList<IReadOnlyList<JsonNode?>>>(StringComparer.Ordinal);
        var memo = new Dictionary<string, IReadOnlyList<IReadOnlyList<JsonNode?>>>(StringComparer.Ordinal);
        var failed = new HashSet<string>(StringComparer.Ordinal);
        async ValueTask Emit(string type, string? id = null, string? message = null)
        { if (onEvent is not null) await onEvent(new(type, id, message)); }
        async Task<IReadOnlyList<IReadOnlyList<JsonNode?>>> Run(string id)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (memo.TryGetValue(id, out var previous)) return previous;
            if (failed.Contains(id)) throw new NodeFailure(id, "Dependency previously failed.");
            var data = prompt[id]!.AsObject();
            Registry.TryGet(data["class_type"]!.GetValue<string>(), out var node);
            var rawInputs = data["inputs"]!.AsObject();
            var resolved = new Dictionary<string, IReadOnlyList<JsonNode?>>(StringComparer.Ordinal);
            async Task Resolve(InputSchema input)
            {
                if (!rawInputs.TryGetPropertyValue(input.Name, out var raw)) return;
                if (TryLink(raw, out var source, out var index)) resolved[input.Name] = (await Run(source))[index];
                else resolved[input.Name] = [Normalize(Unwrap(raw), input)];
            }
            try
            {
                foreach (var input in node.Schema.Inputs.Where(i => !i.Lazy)) await Resolve(input);
                // Read-only interfaces do not make their JSON values immutable. Hooks receive
                // private containers and deep values, just like ordinary node invocations.
                var lazySnapshot = resolved.ToDictionary(p => p.Key,
                    p => (IReadOnlyList<JsonNode?>)p.Value.Select(v => v?.DeepClone()).ToArray(), StringComparer.Ordinal);
                foreach (var name in node.GetRequiredLazyInputs(lazySnapshot))
                {
                    var input = node.Schema.Inputs.SingleOrDefault(i => i.Name == name && i.Lazy)
                        ?? throw new InvalidOperationException($"Node requested undeclared lazy input '{name}'.");
                    await Resolve(input);
                }
                await Emit("executing", id);
                var output = node.Schema.Outputs.Select(_ => new List<JsonNode?>()).ToArray();
                var count = node.Schema.InputIsList ? 1 : resolved.Count == 0 ? 1 : resolved.Values.Max(v => v.Count);
                // Upstream invokes once with no arguments when every execution list is empty.
                if (count == 0) count = 1;
                for (var index = 0; index < count; index++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var invocation = new Dictionary<string, JsonNode?>(StringComparer.Ordinal);
                    foreach (var (name, values) in resolved)
                    {
                        if (node.Schema.InputIsList) invocation[name] = new JsonArray(values.Select(v => v?.DeepClone()).ToArray());
                        else if (values.Count != 0) invocation[name] = values[Math.Min(index, values.Count - 1)]?.DeepClone();
                        else if (resolved.Values.Any(v => v.Count != 0)) throw new InvalidOperationException("Cannot repeat the last item of an empty execution list.");
                    }
                    var returned = await node.ExecuteAsync(invocation, cancellationToken);
                    cancellationToken.ThrowIfCancellationRequested();
                    if (returned.Count != output.Length) throw new InvalidOperationException("Node returned an incorrect output count.");
                    for (var slot = 0; slot < output.Length; slot++)
                    {
                        if (node.Schema.Outputs[slot].IsList)
                        {
                            if (returned[slot] is not JsonArray list) throw new InvalidOperationException("List output must be a JSON array.");
                            output[slot].AddRange(list.Select(v => v?.DeepClone()));
                        }
                        else output[slot].Add(returned[slot]?.DeepClone());
                    }
                }
                var completed = output.Cast<IReadOnlyList<JsonNode?>>().ToArray();
                memo[id] = completed;
                await Emit("executed", id);
                return completed;
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { throw; }
            catch (NodeFailure) { failed.Add(id); throw; }
            catch (Exception e) { failed.Add(id); throw new NodeFailure(id, e.Message); }
        }
        if (!validation.IsValid) return new("error", results, diagnostics);
        await Emit("execution_start");
        try
        {
            foreach (var target in validation.ValidTargets)
            {
                try
                {
                    var output = await Run(target);
                    // Never expose memo entries to an event consumer or another result owner.
                    results[target] = output.Select(slot => (IReadOnlyList<JsonNode?>)slot.Select(v => v?.DeepClone()).ToArray()).ToArray();
                }
                catch (NodeFailure e)
                {
                    diagnostics.Add(new("execution_error", e.Message, e.NodeId, TargetId: target));
                    await Emit("execution_error", e.NodeId, e.Message);
                }
            }
            var status = diagnostics.Any(d => d.Code == "execution_error") ? "error" : "success";
            await Emit(status == "success" ? "execution_success" : "execution_failed");
            return new(status, results, diagnostics);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            await Emit("execution_interrupted");
            return new("cancelled", results, diagnostics);
        }
    }

    private sealed class NodeFailure(string nodeId, string message) : Exception(message) { public string NodeId { get; } = nodeId; }
    private static string? ReadString(JsonNode? node) => node is JsonValue v && v.TryGetValue<string>(out var text) ? text : null;
    private static JsonNode? Unwrap(JsonNode? value) => value is JsonObject obj && obj.TryGetPropertyValue("__value__", out var wrapped) ? wrapped : value;
    private static bool TryLink(JsonNode? value, out string source, out int index)
    {
        source = ""; index = -1;
        if (value is not JsonArray { Count: 2 } array || ReadString(array[0]) is not { } id ||
            array[1] is not JsonValue number || !number.TryGetValue<int>(out index) || index < 0) return false;
        source = id; return true;
    }
    private static bool Compatible(string output, string input)
    {
        if (output == "COMFY_MATCHTYPE_V3" || input == "COMFY_MATCHTYPE_V3") return true;
        var produced = output.Split(',', StringSplitOptions.TrimEntries);
        var accepted = input.Split(',', StringSplitOptions.TrimEntries);
        return produced.Contains("*") || accepted.Contains("*") || produced.Intersect(accepted).Any();
    }
    internal static JsonNode? Normalize(JsonNode? value, InputSchema input)
    {
        JsonNode? normalized;
        try
        {
            normalized = input.Type switch
            {
                "*" or "COMFY_MATCHTYPE_V3" => value?.DeepClone(),
                "STRING" => JsonValue.Create(PythonValues.String(value)),
                "BOOLEAN" => JsonValue.Create(PythonValues.Truth(value)),
                "INT" => JsonValue.Create(PythonValues.Integer(value)),
                "FLOAT" => JsonValue.Create(PythonValues.Float(value)),
                "ARRAY" when value is JsonArray => value.DeepClone(),
                "DICT" when value is JsonObject => value.DeepClone(),
                "COMBO" when value is JsonValue v && v.TryGetValue<string>(out _) => value.DeepClone(),
                _ => throw new FormatException($"Expected {input.Type} literal; opaque types require links.")
            };
            if (input.Options?["options"] is JsonArray choices && !choices.Any(c => JsonNode.DeepEquals(c, normalized)))
                throw new FormatException("Value is not an allowed choice.");
            if (input.Type == "INT")
            {
                var number = PythonValues.Integer(normalized);
                if (input.Options?["min"] is { } min && number < PythonValues.Integer(min)) throw new FormatException("Value is below minimum.");
                if (input.Options?["max"] is { } max && number > PythonValues.Integer(max)) throw new FormatException("Value is above maximum.");
            }
            if (input.Type == "FLOAT")
            {
                var number = PythonValues.Float(normalized);
                if (input.Options?["min"] is { } min && number < PythonValues.Float(min)) throw new FormatException("Value is below minimum.");
                if (input.Options?["max"] is { } max && number > PythonValues.Float(max)) throw new FormatException("Value is above maximum.");
            }
            return normalized;
        }
        catch (Exception e) when (e is InvalidCastException or ArgumentException) { throw new FormatException(e.Message, e); }
    }
}
