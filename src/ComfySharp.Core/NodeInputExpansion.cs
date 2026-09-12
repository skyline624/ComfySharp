using System.Collections.ObjectModel;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Core;

public sealed class NodeInputExpansionException(string code, string message, string? inputName = null) : ArgumentException(message)
{
    public string Code { get; } = code;
    public string? InputName { get; } = inputName;
}

/// <summary>Finalized flat inputs and ordered V3 paths. Bind only after the flat blocker check.</summary>
public sealed class ExpandedNodeInputs
{
    private readonly IReadOnlyList<InputSchema> inputs;
    private readonly IReadOnlyList<(string Name, IReadOnlyList<(string Flat, string Member)> Paths)> groups;
    public IReadOnlyList<InputSchema> Inputs => groups.Count == 0 ? inputs : Array.AsReadOnly(inputs.Select(Snapshot).ToArray());
    internal ExpandedNodeInputs(IReadOnlyList<InputSchema> inputs,
        IReadOnlyList<(string Name, IReadOnlyList<(string Flat, string Member)> Paths)> groups)
    { this.inputs = inputs; this.groups = groups; }

    public IReadOnlyDictionary<string, RuntimeValue> BindArguments(RuntimeNodeContext context,
        IReadOnlyDictionary<string, RuntimeValue> flat, CancellationToken cancellationToken = default)
    {
        if (groups.Count == 0) return flat;
        cancellationToken.ThrowIfCancellationRequested();
        var dynamicNames = groups.SelectMany(g => g.Paths.Select(p => p.Flat)).ToHashSet(StringComparer.Ordinal);
        var result = flat.Where(p => !dynamicNames.Contains(p.Key)).ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal);
        foreach (var group in groups)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var members = new Dictionary<string, RuntimeValue>(StringComparer.Ordinal);
            foreach (var path in group.Paths)
            {
                cancellationToken.ThrowIfCancellationRequested();
                // Source build_nested_inputs uses None when a finalized live path has no mapped value.
                members.Add(path.Member, flat.TryGetValue(path.Flat, out var value) ? value : context.Json(null));
            }
            result.Add(group.Name, context.Map(members));
        }
        cancellationToken.ThrowIfCancellationRequested();
        return new ReadOnlyDictionary<string, RuntimeValue>(result);
    }
    private static InputSchema Snapshot(InputSchema input) => input with { Options = input.Options?.DeepClone().AsObject() };
}

public static class NodeInputExpansion
{
    public static ExpandedNodeInputs Expand(NodeSchema schema, IEnumerable<string> liveNames)
    {
        if (!schema.Inputs.Any(i => i.Autogrow is not null || i.Type == "COMFY_AUTOGROW_V3"))
            return new(schema.Inputs, []);
        if (schema.Inputs.FirstOrDefault(i => i.Lazy) is { } lazy)
            throw new NodeInputExpansionException("unsupported_dynamic_template",
                "Mixing Autogrow and lazy inputs is not supported by this slice.", lazy.Name);
        var live = liveNames.ToHashSet(StringComparer.Ordinal);
        var leaves = new List<InputSchema>();
        var groups = new List<(string Name, IReadOnlyList<(string Flat, string Member)> Paths)>();
        var allNames = new HashSet<string>(StringComparer.Ordinal);
        var rootNames = new HashSet<string>(StringComparer.Ordinal);
        foreach (var input in schema.Inputs)
        {
            if (!rootNames.Add(input.Name)) throw new NodeInputExpansionException("invalid_dynamic_template", "Duplicate input/group name.", input.Name);
            if (input.Autogrow is not { } template)
            {
                if (input.Type == "COMFY_AUTOGROW_V3") throw new NodeInputExpansionException("invalid_dynamic_template", "Missing typed Autogrow template.", input.Name);
                Add(input with { Options = input.Options?.DeepClone().AsObject() }); continue;
            }
            if (input.Type != "COMFY_AUTOGROW_V3" || input.Lazy || !input.Required ||
                string.IsNullOrEmpty(input.Name) || input.Name.Contains('.'))
                throw new NodeInputExpansionException("unsupported_dynamic_template", "Only required, non-lazy, top-level prefix groups are supported.", input.Name);
            try { AutogrowPrefixTemplate.ValidateOptions(input.Options); }
            catch (ArgumentException ex) { throw new NodeInputExpansionException("unsupported_dynamic_template", ex.Message, input.Name); }
            var prototype = template.Input;
            var paths = new List<(string Flat, string Member)>();
            for (int i = 0; i < template.Max; i++)
            {
                string member = template.Prefix + i.ToString(System.Globalization.CultureInfo.InvariantCulture);
                string name = input.Name + "." + member;
                Add(prototype with { Name = name, Required = prototype.Required && i < template.Min,
                    Options = prototype.Options?.DeepClone().AsObject() });
                if (live.Contains(name)) paths.Add((name, member));
            }
            groups.Add((input.Name, paths.AsReadOnly()));
        }
        foreach (var name in live)
            if (!allNames.Contains(name)) throw new NodeInputExpansionException("unsupported_dynamic_input",
                "Input is outside the supported flat Autogrow contract. Extra inputs are not silently ignored.", name);
        return new(leaves.AsReadOnly(), groups.AsReadOnly());

        void Add(InputSchema leaf)
        {
            if (!allNames.Add(leaf.Name)) throw new NodeInputExpansionException("invalid_dynamic_template", "Expanded input names collide.", leaf.Name);
            leaves.Add(leaf);
        }
    }

    public static JsonObject TemplateOptions(InputSchema group)
    {
        // Validate the same definition used by execution, without any live prompt input.
        _ = Expand(new("template", "template", "", [group], []), []);
        var template = group.Autogrow!; var input = template.Input;
        var prototypeOptions = input.Options?.DeepClone().AsObject() ?? new JsonObject();
        var prototype = new JsonObject { [input.Required ? "required" : "optional"] = new JsonObject
        { [input.Name] = new JsonArray(input.Type, prototypeOptions) } };
        if (!input.Required) prototype.Insert(0, "required", new JsonObject());
        var options = group.Options?.DeepClone().AsObject() ?? new JsonObject();
        options["template"] = new JsonObject { ["input"] = prototype, ["prefix"] = template.Prefix,
            ["min"] = template.Min, ["max"] = template.Max };
        return options;
    }
}
