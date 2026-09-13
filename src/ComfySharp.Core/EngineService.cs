using System.Globalization;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Core;

/// <summary>Typed DAG executor. Each job owns its memo; returned values have independent disposable leases.</summary>
public sealed class EngineService(NodeRegistry registry) : IDisposable
{
    public NodeRegistry Registry { get; } = registry;
    private readonly NodeObjectCache objects = new();
    public void Dispose() => objects.Dispose();

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
                ExpandedNodeInputs expanded;
                try { expanded = NodeInputExpansion.Expand(node.Schema, inputs.Select(p => p.Key)); }
                catch (NodeInputExpansionException e) { Error(e.Code, e.Message, id, e.InputName); return; }
                active.Add(id);
                foreach (var input in expanded.Inputs)
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
        Func<EngineEvent, ValueTask>? onEvent = null, CancellationToken cancellationToken = default, JsonObject? extraData = null)
    {
        // Project only at the explicit JSON boundary. Native disposal is part of successful completion.
        EngineEvent? terminal = null;
        async ValueTask Forward(EngineEvent e)
        {
            if (IsTerminal(e)) terminal = e;
            else if (onEvent is not null) await onEvent(e);
        }
        using var owned = await ExecuteValuesAsync(prompt, targets, Forward, cancellationToken, extraData);
        var diagnostics = owned.Diagnostics.ToList();
        var outputs = new Dictionary<string, IReadOnlyList<IReadOnlyList<JsonNode?>>>(StringComparer.Ordinal);
        var status = owned.Status;
        foreach (var (target, slots) in owned.Outputs)
        {
            try { outputs.Add(target, slots.Select(slot => (IReadOnlyList<JsonNode?>)slot.Select(v => v.ToJson()).ToArray()).ToArray()); }
            catch (RuntimeValueProjectionException e)
            {
                diagnostics.Add(new("runtime_value_not_json", e.Message, target, TargetId: target));
                if (status != "cancelled") status = "error";
                if (onEvent is not null) await onEvent(new("execution_error", target, e.Message, Identity: new(target, target)));
            }
        }
        owned.Dispose();
        if (onEvent is not null && terminal is not null)
            await onEvent(status == "cancelled" ? terminal : new(status == "success" ? "execution_success" : "execution_failed"));
        return new(status, outputs, diagnostics);
    }

    /// <summary>Returns only explicit UI documents. All native slots are released before the terminal event.</summary>
    public async Task<UiExecutionResult> ExecuteUiAsync(JsonObject prompt, IReadOnlyCollection<string>? targets = null,
        Func<EngineEvent, ValueTask>? onEvent = null, CancellationToken cancellationToken = default, JsonObject? extraData = null)
    {
        EngineEvent? terminal = null;
        async ValueTask Forward(EngineEvent e)
        {
            if (IsTerminal(e)) terminal = e;
            else if (onEvent is not null) await onEvent(e);
        }
        using var owned = await ExecuteValuesAsync(prompt, targets, Forward, cancellationToken, extraData);
        var result = new UiExecutionResult(owned.Status,
            owned.UiOutputs.ToDictionary(p => p.Key, p => UiDocument.Snapshot(p.Value), StringComparer.Ordinal),
            owned.Meta.ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal), owned.Diagnostics);
        owned.Dispose();
        if (onEvent is not null && terminal is not null) await onEvent(terminal);
        return result;
    }

    private static bool IsTerminal(EngineEvent e) => e.Type is "execution_success" or "execution_failed" or "execution_interrupted";

    public async Task<OwnedExecutionResult> ExecuteValuesAsync(JsonObject prompt, IReadOnlyCollection<string>? targets = null,
        Func<EngineEvent, ValueTask>? onEvent = null, CancellationToken cancellationToken = default, JsonObject? extraData = null)
    {
        // Freeze caller-owned JSON before the first await, so validation and execution see the same graph.
        prompt = prompt.DeepClone().AsObject();
        extraData = extraData?.DeepClone().AsObject();
        objects.CheckAvailable();
        var validation = Validate(prompt, targets);
        using var objectScope = validation.IsValid ? objects.BeginPrompt(prompt) : null;
        var diagnostics = validation.Diagnostics.ToList();
        var results = new Dictionary<string, IReadOnlyList<IReadOnlyList<RuntimeValue>>>(StringComparer.Ordinal);
        var memo = new Dictionary<string, IReadOnlyList<IReadOnlyList<RuntimeValue>>>(StringComparer.Ordinal);
        var uiOutputs = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
        var meta = new Dictionary<string, NodeExecutionIdentity>(StringComparer.Ordinal);
        using var job = new RuntimeNodeContext();
        OwnedExecutionResult Complete(string status)
        {
            var result = new OwnedExecutionResult(status, results, diagnostics, uiOutputs, meta);
            try { job.Dispose(); objectScope?.Dispose(); return result; }
            catch { result.Dispose(); throw; }
        }
        async Task<OwnedExecutionResult> CompleteWithEvent(string status, string type)
        {
            var result = Complete(status);
            try { await Emit(type); return result; }
            catch { result.Dispose(); throw; }
        }
        var failed = new HashSet<string>(StringComparer.Ordinal);
        async ValueTask Emit(string type, string? id = null, string? message = null, JsonObject? output = null)
        {
            if (onEvent is not null) await onEvent(new(type, id, message, output is null ? null : UiDocument.Snapshot(output),
                id is null ? null : new NodeExecutionIdentity(id, id)));
        }
        async Task<IReadOnlyList<IReadOnlyList<RuntimeValue>>> Run(string id)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (memo.TryGetValue(id, out var previous)) return previous;
            if (failed.Contains(id)) throw new NodeFailure(id, "Dependency previously failed.");
            var data = prompt[id]!.AsObject();
            Registry.TryGet(data["class_type"]!.GetValue<string>(), out var node);
            var rawInputs = data["inputs"]!.AsObject();
            var promptInputOrder = rawInputs.Select(p => p.Key).ToArray();
            var expanded = NodeInputExpansion.Expand(node.Schema, promptInputOrder);
            // Supplied hidden names retain their prompt position; new names append in schema order.
            // Their values always come from job metadata, never from caller-provided links/literals.
            var argumentOrder = promptInputOrder.Concat(node.Schema.HiddenInputs?.Select(i => i.Name) ?? [])
                .Distinct(StringComparer.Ordinal).ToArray();
            var declaredInputs = expanded.Inputs;
            using var nodeScope = new RuntimeNodeContext();
            var resolved = new Dictionary<string, IReadOnlyList<RuntimeValue>>(StringComparer.Ordinal);
            async Task Resolve(InputSchema input)
            {
                if (!rawInputs.TryGetPropertyValue(input.Name, out var raw)) return;
                if (TryLink(raw, out var source, out var index)) resolved[input.Name] = (await Run(source))[index];
                else resolved[input.Name] = [nodeScope.Json(Normalize(Unwrap(raw), input))];
            }
            try
            {
                foreach (var input in declaredInputs.Where(i => !i.Lazy)) await Resolve(input);
                foreach (var input in node.Schema.HiddenInputs ?? [])
                    resolved[input.Name] = [nodeScope.Json(LegacyHiddenInputs.Resolve(input.Type, prompt, extraData, id))];
                cancellationToken.ThrowIfCancellationRequested();
                var instance = objectScope!.Get(id, node);
                IReadOnlyCollection<string> lazyNames;
                using (var lazyScope = new RuntimeNodeContext())
                {
                    var lazySnapshot = LazySnapshot(lazyScope, resolved, node.Schema.InputIsList);
                    lazyNames = lazySnapshot is null ? [] : instance.GetRequiredLazyInputs(lazySnapshot).ToArray();
                }
                foreach (var name in lazyNames.Distinct(StringComparer.Ordinal))
                {
                    var input = declaredInputs.SingleOrDefault(i => i.Name == name && i.Lazy)
                        ?? throw new InvalidOperationException($"Node requested undeclared lazy input '{name}'.");
                    await Resolve(input);
                }
                await Emit("executing", id);
                var output = node.Schema.Outputs.Select(_ => new List<RuntimeValue>()).ToArray();
                var invocationUis = new List<JsonObject>();
                var count = node.Schema.InputIsList ? 1 : resolved.Count == 0 ? 1 : resolved.Values.Max(v => v.Count);
                // Upstream invokes once with no arguments when every execution list is empty.
                if (count == 0) count = 1;
                for (var index = 0; index < count; index++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    using var invocationScope = new RuntimeNodeContext(LegacyHiddenInputs.V3Context(node.Schema,prompt,extraData,id));
                    var invocation = new Dictionary<string, RuntimeValue>(StringComparer.Ordinal);
                    foreach (var (name, values) in resolved)
                    {
                        if (node.Schema.InputIsList) invocation[name] = invocationScope.List(values);
                        else if (values.Count != 0) invocation[name] = invocationScope.Retain(values[Math.Min(index, values.Count - 1)]);
                        else if (resolved.Values.Any(v => v.Count != 0)) throw new InvalidOperationException("Cannot repeat the last item of an empty execution list.");
                    }
                    var block = FirstBlocker(invocation, argumentOrder, node.Schema.InputIsList);
                    NodeExecutionOutput returned;
                    if (block is not null)
                    {
                        if (block.Message is not null)
                        {
                            string message = $"Execution Blocked: {block.Message}";
                            diagnostics.Add(new("execution_blocked", message, id));
                            await Emit("execution_error", id, message);
                        }
                        cancellationToken.ThrowIfCancellationRequested();
                        // Consume a message at this invocation, never mutate the memoized producer.
                        returned = NodeExecutionOutput.Blocked();
                    }
                    else
                    {
                        // Source scans flat prompt-order values for blockers before constructing V3 dictionaries.
                        // Acquisition preserves prompt order for ordinary arguments; the V3 binder then
                        // appends groups in schema order. Reorder here without changing producer resolution.
                        var flatArguments = argumentOrder.Where(invocation.ContainsKey)
                            .ToDictionary(name => name, name => invocation[name], StringComparer.Ordinal);
                        var arguments = expanded.BindArguments(invocationScope, flatArguments, cancellationToken);
                        returned = await instance.ExecuteAsync(invocationScope, arguments, cancellationToken);
                    }
                    if (returned.Ui is not null) invocationUis.Add(UiDocument.Snapshot(returned.Ui));
                    cancellationToken.ThrowIfCancellationRequested();
                    var returnedValues = returned.BlockExecution is { } wholeCall
                        ? Enumerable.Range(0, output.Length).Select(_ => invocationScope.Blocker(wholeCall.Message)).ToArray()
                        : returned.Result;
                    if (returnedValues.Count != output.Length) throw new InvalidOperationException("Node returned an incorrect output count.");
                    for (var slot = 0; slot < output.Length; slot++)
                    {
                        if (node.Schema.Outputs[slot].IsList && returnedValues[slot].Kind != RuntimeValueKind.Blocker)
                        {
                            if (returnedValues[slot].Kind != RuntimeValueKind.List) throw new InvalidOperationException("List output must be an execution list.");
                            foreach (var value in returnedValues[slot].Items) output[slot].Add(nodeScope.Retain(value));
                        }
                        else output[slot].Add(nodeScope.Retain(returnedValues[slot]));
                    }
                }
                var ui = MergeUi(invocationUis);
                var completed = output.Select(slot => (IReadOnlyList<RuntimeValue>)slot.Select(job.Retain).ToArray()).ToArray();
                nodeScope.Dispose();
                if (ui.Count != 0)
                {
                    uiOutputs[id] = ui;
                    meta[id] = new(id, id);
                    await Emit("executed", id, output: ui);
                }
                memo[id] = completed;
                return completed;
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { throw; }
            catch (NodeFailure) { failed.Add(id); throw; }
            catch (Exception e) { failed.Add(id); throw new NodeFailure(id, e.Message); }
        }
        if (!validation.IsValid) return Complete("error");
        string status;
        try
        {
            await Emit("execution_start");
            foreach (var target in validation.ValidTargets)
            {
                try
                {
                    var output = await Run(target);
                    results[target] = output;
                }
                catch (NodeFailure e)
                {
                    diagnostics.Add(new("execution_error", e.Message, e.NodeId, TargetId: target));
                    await Emit("execution_error", e.NodeId, e.Message);
                }
            }
            status = diagnostics.Any(d => d.Code == "execution_error") ? "error" : "success";
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            status = "cancelled";
        }
        // Completion is outside the execution cancellation handler: a terminal transport/disposal exception
        // cannot attempt a second handoff from leases that were already transferred and released.
        return await CompleteWithEvent(status, status == "cancelled" ? "execution_interrupted"
            : status == "success" ? "execution_success" : "execution_failed");
    }

    private static ExecutionBlocker? FirstBlocker(IReadOnlyDictionary<string, RuntimeValue> inputs,
        IReadOnlyList<string> promptOrder, bool inputIsList)
    {
        foreach (var name in promptOrder)
        {
            if (!inputs.TryGetValue(name, out var value)) continue;
            if (value.Kind == RuntimeValueKind.Blocker) return value.Blocker;
            if (inputIsList && value.Kind == RuntimeValueKind.List)
                foreach (var item in value.Items)
                    if (item.Kind == RuntimeValueKind.Blocker) return item.Blocker;
        }
        return null;
    }

    private static IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>>? LazySnapshot(RuntimeNodeContext scope,
        IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> resolved, bool inputIsList)
    {
        bool hasBlocker = resolved.Values.Any(values => values.Any(v => v.Kind == RuntimeValueKind.Blocker));
        if (!hasBlocker)
            return resolved.ToDictionary(p => p.Key,
                p => (IReadOnlyList<RuntimeValue>)Array.AsReadOnly(p.Value.Select(scope.Retain).ToArray()), StringComparer.Ordinal);
        if (inputIsList) return null;
        int count = resolved.Values.Max(values => values.Count);
        if (resolved.Values.Any(values => values.Count == 0))
            throw new InvalidOperationException("Cannot repeat the last item of an empty execution list.");
        // Slice all inputs together before removing blocked rows. Independent list
        // filtering would shift pairings and could request an unrelated lazy branch.
        var activeRows = Enumerable.Range(0, count).Where(index => resolved.Values.All(values =>
            values[Math.Min(index, values.Count - 1)].Kind != RuntimeValueKind.Blocker)).ToArray();
        if (activeRows.Length == 0) return null;
        return resolved.ToDictionary(p => p.Key, p => (IReadOnlyList<RuntimeValue>)Array.AsReadOnly(activeRows
            .Select(index => scope.Retain(p.Value[Math.Min(index, p.Value.Count - 1)])).ToArray()), StringComparer.Ordinal);
    }

    private static JsonObject MergeUi(IReadOnlyList<JsonObject> invocations)
    {
        var merged = new JsonObject();
        if (invocations.Count == 0) return merged;
        // execution.py: the first returned UI determines keys; subsequent lists concatenate.
        // A later extra key is ignored, while a missing key cannot be silently supplied as an empty list.
        foreach (var (key, _) in invocations[0])
        {
            var items = new JsonArray();
            foreach (var invocation in invocations)
            {
                if (invocation[key] is not JsonArray values)
                    throw new InvalidOperationException($"UI output '{key}' must contain a JSON array in every UI return.");
                foreach (var value in values) items.Add(value?.DeepClone());
            }
            merged[key] = items;
        }
        return merged;
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
            bool unsigned = input.Type == "INT" && input.Options?["max"] is JsonValue limit &&
                limit.TryGetValue<ulong>(out var upper) && upper > long.MaxValue &&
                input.Options?["min"] is { } lower && PythonValues.Float(lower) >= 0;
            normalized = input.Type switch
            {
                "*" or "COMFY_MATCHTYPE_V3" => value?.DeepClone(),
                "STRING" => JsonValue.Create(PythonValues.String(value)),
                "BOOLEAN" => JsonValue.Create(PythonValues.Truth(value)),
                "INT" => unsigned ? JsonValue.Create(PythonValues.UnsignedInteger(value)) : JsonValue.Create(PythonValues.Integer(value)),
                "FLOAT" => JsonValue.Create(PythonValues.Float(value)),
                "ARRAY" when value is JsonArray => value.DeepClone(),
                "DICT" when value is JsonObject => value.DeepClone(),
                "COMBO" when value is JsonValue v && v.TryGetValue<string>(out _) => value.DeepClone(),
                _ => throw new FormatException($"Expected {input.Type} literal; opaque types require links.")
            };
            if (input.Options?["options"] is JsonArray choices && !choices.Any(c => JsonNode.DeepEquals(c, normalized)))
                throw new FormatException("Value is not an allowed choice.");
            if (unsigned)
            {
                var number = PythonValues.UnsignedInteger(normalized);
                if (input.Options?["min"] is { } min && number < PythonValues.UnsignedInteger(min)) throw new FormatException("Value is below minimum.");
                if (input.Options?["max"] is { } max && number > PythonValues.UnsignedInteger(max)) throw new FormatException("Value is above maximum.");
            }
            else if (input.Type == "INT")
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
