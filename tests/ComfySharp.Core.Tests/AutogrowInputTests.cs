using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>Managed behavioral checks from the source contract; numerical/source fixtures are collected separately.</summary>
public sealed class AutogrowInputTests
{
    private static NodeSchema Schema(AutogrowPrefixTemplate template) => new("Test", "Test", "test",
        [new("values", "COMFY_AUTOGROW_V3", Autogrow: template)], []);

    [Fact]
    public void TemplateAndExpandedOptionsAreIndependentSnapshots()
    {
        var options = new JsonObject { ["tooltip"] = "original", ["advanced"] = true };
        var template = new AutogrowPrefixTemplate(new("value", "*", Options: options), "value");
        options["tooltip"] = "caller change";
        template.Input.Options!["tooltip"] = "getter change";
        var expanded = NodeInputExpansion.Expand(Schema(template), ["values.value2", "values.value0"]);
        expanded.Inputs[0].Options!["tooltip"] = "leaf change";
        Assert.Equal("original", template.Input.Options!["tooltip"]!.GetValue<string>());
        Assert.Equal("original", expanded.Inputs[0].Options!["tooltip"]!.GetValue<string>());
        Assert.Equal("original", expanded.Inputs[1].Options!["tooltip"]!.GetValue<string>());
        Assert.Equal(10, expanded.Inputs.Count);
        Assert.True(expanded.Inputs[0].Required);
        Assert.All(expanded.Inputs.Skip(1), input => Assert.False(input.Required));
    }

    [Theory]
    [InlineData(-1, 10)]
    [InlineData(0, 0)]
    [InlineData(1, 101)]
    public void SourcePrefixBoundsAreEnforced(int min, int max) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new AutogrowPrefixTemplate(new("value", "*"), "value", min, max));

    [Fact]
    public void MinimumAboveMaximumKeepsAllGeneratedNamesRequired()
    {
        var expanded = NodeInputExpansion.Expand(Schema(new(new("value", "*"), "value", 12, 10)), []);
        Assert.Equal(10, expanded.Inputs.Count);
        Assert.All(expanded.Inputs, input => Assert.True(input.Required));
        var hundred = NodeInputExpansion.Expand(Schema(new(new("value", "*"), "value", 0, 100)), []);
        Assert.Equal("values.value99", hundred.Inputs[^1].Name);
    }

    [Fact]
    public void OptionalPrototypeIgnoresMinimumAndNoLivePathsBindAnEmptyMap()
    {
        var template = new AutogrowPrefixTemplate(new("value", "*", Required: false), "value", 8, 10);
        var expanded = NodeInputExpansion.Expand(Schema(template), []);
        Assert.All(expanded.Inputs, input => Assert.False(input.Required));
        using var context = new RuntimeNodeContext();
        var bound = expanded.BindArguments(context, new Dictionary<string, RuntimeValue>());
        Assert.Empty(bound["values"].Properties);
        var options = NodeInputExpansion.TemplateOptions(Schema(template).Inputs[0]);
        Assert.Empty(options["template"]!["input"]!["required"]!.AsObject());
        Assert.NotNull(options["template"]!["input"]!["optional"]!["value"]);
    }

    [Theory]
    [InlineData("rawLink")]
    [InlineData("force_input")]
    [InlineData("default")]
    [InlineData("unreviewed")]
    public void UnqualifiedSemanticOptionsAreExplicitlyRejected(string option) =>
        Assert.Throws<ArgumentException>(() => new AutogrowPrefixTemplate(new("value", "*", Options: new() { [option] = true }), "value"));

    [Fact]
    public void DynamicWidgetAndLazyPrototypesAreNotSilentlyTreatedAsPlainInputs()
    {
        Assert.Throws<ArgumentException>(() => new AutogrowPrefixTemplate(new("value", "STRING"), "value"));
        Assert.Throws<ArgumentException>(() => new AutogrowPrefixTemplate(new("value", "*", Lazy: true), "value"));
        var nested = new AutogrowPrefixTemplate(new("value", "*"), "value");
        Assert.Throws<ArgumentException>(() => new AutogrowPrefixTemplate(new("nested", "COMFY_AUTOGROW_V3", Autogrow: nested), "value"));
        Assert.Throws<ArgumentException>(() => new AutogrowPrefixTemplate(new("value", "COMFY_MATCHTYPE_V3"), "value"));
    }

    [Theory]
    [InlineData("values.value10")]
    [InlineData("values.value01")]
    [InlineData("values")]
    [InlineData("other.value0")]
    public void UnknownNamesReceiveTheDocumentedEarlyDiagnostic(string name)
    {
        var error = Assert.Throws<NodeInputExpansionException>(() => NodeInputExpansion.Expand(Schema(new(new("value", "*"), "value")), [name]));
        Assert.Equal("unsupported_dynamic_input", error.Code); Assert.Equal(name, error.InputName);
    }

    [Fact]
    public void BindingUsesTemplateOrderAndRetainsBorrowedLists()
    {
        var expanded = NodeInputExpansion.Expand(Schema(new(new("value", "*"), "value")), ["values.value2", "values.value0"]);
        using var caller = new RuntimeNodeContext(); using var invocation = new RuntimeNodeContext();
        var first = caller.List([caller.Json(JsonValue.Create("a"))]);
        var second = caller.List([caller.Json(JsonValue.Create("b"))]);
        var flat = new Dictionary<string, RuntimeValue> { ["values.value2"] = second, ["values.value0"] = first };
        var bound = expanded.BindArguments(invocation, flat);
        Assert.Equal(new[] { "value0", "value2" }, bound["values"].Properties.Keys);
        using var kept = bound["values"].Retain();
        invocation.Dispose(); caller.Dispose();
        Assert.Equal("a", kept.Properties["value0"].Items[0].ToJson()!.GetValue<string>());
        Assert.Equal("b", kept.Properties["value2"].Items[0].ToJson()!.GetValue<string>());
    }

    [Fact]
    public void OrdinaryNodesKeepTheirFlatBindingPath()
    {
        var schema = new NodeSchema("Plain", "Plain", "test", [new("value", "*")], []);
        var expanded = NodeInputExpansion.Expand(schema, ["value"]);
        Assert.Same(schema.Inputs, expanded.Inputs);
        using var context = new RuntimeNodeContext();
        var values = new Dictionary<string, RuntimeValue> { ["value"] = context.Json(null) };
        Assert.Same(values, expanded.BindArguments(context, values));
    }

    [Fact]
    public void AStaticLazyInputCannotBypassTheUnsupportedDynamicLazyContract()
    {
        var schema = new NodeSchema("Mixed", "Mixed", "test",
            [new("values", "COMFY_AUTOGROW_V3", Autogrow: new AutogrowPrefixTemplate(new("value", "*"), "value")), new("other", "*", Lazy: true)], []);
        var error = Assert.Throws<NodeInputExpansionException>(() => NodeInputExpansion.Expand(schema, ["values.value0"]));
        Assert.Equal("unsupported_dynamic_template", error.Code); Assert.Equal("other", error.InputName);
    }

    [Fact]
    public void ObjectInfoDescribesTheOriginalTemplateRatherThanExpandedPorts()
    {
        var info = BuiltInNodes.CreateRegistry().ToObjectInfo(); var node = info["CreateList"]!;
        Assert.Equal("COMFY_AUTOGROW_V3", node["input"]!["required"]!["inputs"]![0]!.GetValue<string>());
        var template = node["input"]!["required"]!["inputs"]![1]!["template"]!;
        Assert.Equal("input", template["prefix"]!.GetValue<string>());
        Assert.Equal(1, template["min"]!.GetValue<int>()); Assert.Equal(10, template["max"]!.GetValue<int>());
        Assert.Equal("COMFY_MATCHTYPE_V3", template["input"]!["required"]!["input"]![0]!.GetValue<string>());
        Assert.Equal("type", template["input"]!["required"]!["input"]![1]!["template"]!["template_id"]!.GetValue<string>());
        Assert.Equal("*", template["input"]!["required"]!["input"]![1]!["template"]!["allowed_types"]!.GetValue<string>());
        Assert.Equal("inputs", Assert.Single(node["input_order"]!["required"]!.AsArray())!.GetValue<string>());
        Assert.False(node["input"]!.AsObject().ContainsKey("optional"));
        Assert.True(node["is_input_list"]!.GetValue<bool>());
        Assert.True(node["output_is_list"]![0]!.GetValue<bool>());
        Assert.Equal("list", node["output_name"]![0]!.GetValue<string>());
        Assert.Equal("type", node["output_matchtypes"]![0]!.GetValue<string>());
        Assert.Null(node["output_tooltips"]![0]);
        Assert.Equal("comfy_extras.nodes_toolkit", node["python_module"]!.GetValue<string>());
        Assert.False(node["output_node"]!.GetValue<bool>());
        Assert.False(node["experimental"]!.GetValue<bool>());
        Assert.False(info["PrimitiveString"]!.AsObject().ContainsKey("output_tooltips"));
        template["min"] = 5;
        Assert.Equal(1, BuiltInNodes.CreateRegistry().ToObjectInfo()["CreateList"]!["input"]!["required"]!["inputs"]![1]!["template"]!["min"]!.GetValue<int>());
    }
}
