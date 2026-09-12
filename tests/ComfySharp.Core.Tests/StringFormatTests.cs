using System.Globalization;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>Managed profile/resource contracts; this file does not construct a source oracle.</summary>
public sealed class StringFormatTests
{
    private static string Render(JsonNode? value, string format = "{a}", PythonTextFormat.Limits? limits = null)
    {
        using var context = new RuntimeNodeContext();
        return PythonTextFormat.Format(format, new Dictionary<string, RuntimeValue> { ["a"] = context.Json(value) }, limits: limits);
    }

    [Fact]
    public void EveryAdmittedClrIntegerKindPreservesExactValue()
    {
        JsonNode?[] values = [JsonValue.Create((sbyte)-1), JsonValue.Create((byte)1), JsonValue.Create((short)-2),
            JsonValue.Create((ushort)2), JsonValue.Create(-3), JsonValue.Create((uint)3), JsonValue.Create(long.MinValue),
            JsonValue.Create((ulong)long.MaxValue), JsonNode.Parse("9223372036854775807")];
        string[] expected = ["-1", "1", "-2", "2", "-3", "3", "-9223372036854775808", "9223372036854775807", "9223372036854775807"];
        Assert.Equal(expected, values.Select(value => Render(value)));
    }

    [Fact]
    public void FloatAndDecimalRepresentationsNeverBecomeIntegerTokens()
    {
        JsonNode?[] values = [JsonValue.Create(1f), JsonValue.Create(1d), JsonValue.Create(1m), JsonValue.Create(ulong.MaxValue),
            JsonNode.Parse("1.0"), JsonNode.Parse("1e0"), JsonNode.Parse("9223372036854775808"), JsonValue.Create(double.NaN),
            JsonValue.Create(double.PositiveInfinity)];
        foreach (var value in values)
            Assert.Equal("unsupported_format_value", Assert.Throws<PythonTextFormat.Error>(() => Render(value)).Category);
    }

    [Fact]
    public void BooleanNullAndIntegersRemainDistinctAndCultureIndependent()
    {
        var previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("fr-FR");
            Assert.Equal("True", Render(JsonValue.Create(true)));
            Assert.Equal("False", Render(JsonValue.Create(false)));
            Assert.Equal("None", Render(null));
            Assert.Equal("1234567", Render(JsonValue.Create(1234567)));
        }
        finally { CultureInfo.CurrentCulture = previous; }
    }

    [Theory]
    [InlineData("{a!r}", "unsupported_format_feature")]
    [InlineData("{a!a}", "unsupported_format_feature")]
    [InlineData("{a.real}", "unsupported_format_feature")]
    [InlineData("{a[x]}", "unsupported_format_feature")]
    [InlineData("{a:>{b}}", "unsupported_format_feature")]
    [InlineData("{a:05}", "unsupported_format_feature")]
    [InlineData("{a:=6}", "unsupported_format_feature")]
    [InlineData("{a:+6}", "unsupported_format_feature")]
    [InlineData("{a:#6}", "unsupported_format_feature")]
    [InlineData("{a:,}", "unsupported_format_feature")]
    [InlineData("{a:.2f}", "unsupported_format_feature")]
    [InlineData("{a!q}", "format_syntax")]
    [InlineData("{a!qq}", "format_syntax")]
    [InlineData("{a:.}", "format_syntax")]
    public void ExcludedFeaturesFailWithAnExplicitProfileCategory(string format, string category) =>
        Assert.Equal(category, Assert.Throws<PythonTextFormat.Error>(() => Render(JsonValue.Create("text"), format)).Category);

    [Theory]
    [InlineData("{b} {", "format_missing_field")]
    [InlineData("{b!q}", "format_missing_field")]
    [InlineData("{b:04x}", "format_missing_field")]
    [InlineData("{b!qq}", "format_syntax")]
    [InlineData("{b", "format_syntax")]
    [InlineData("{a!q:{b}}", "format_syntax")]
    [InlineData("{}", "format_missing_field")]
    [InlineData("{0}", "format_missing_field")]
    public void StructuralParseLookupAndConversionHaveSeparateOrderedFailureBoundaries(string format, string category) =>
        Assert.Equal(category, Assert.Throws<PythonTextFormat.Error>(() => Render(JsonValue.Create("x"), format)).Category);

    [Fact]
    public void ScalarSpecificationRequiresExplicitConversion()
    {
        Assert.Equal("unsupported_format_feature", Assert.Throws<PythonTextFormat.Error>(() => Render(JsonValue.Create(12), "{a:6}")).Category);
        Assert.Equal("12    ", Render(JsonValue.Create(12), "{a!s:6}"));
        Assert.Equal("None", Render(null, "{a:}"));
    }

    [Fact]
    public void OrdinalLookupDoesNotInheritCallerComparer()
    {
        using var context = new RuntimeNodeContext();
        var values = new Dictionary<string, RuntimeValue>(StringComparer.OrdinalIgnoreCase) { ["A"] = context.Json(JsonValue.Create("upper")) };
        Assert.Equal("upper", PythonTextFormat.Format("{A}", values));
        Assert.Equal("format_missing_field", Assert.Throws<PythonTextFormat.Error>(() => PythonTextFormat.Format("{a}", values)).Category);
    }

    [Fact]
    public void CodePointBudgetsCountAstralTextAndPrecisionIndependentlyOfUtf16()
    {
        var limits = new PythonTextFormat.Limits(32, 6, 4);
        Assert.Equal("😀😀😀😀", Render(JsonValue.Create("😀😀😀😀x"), "{a:.4}", limits));
        Assert.Equal("😀xx😀", Render(JsonValue.Create("xx"), "{a:😀^4}", limits));
        Assert.Equal("", Render(JsonValue.Create("😀x"), "{a:.0}", limits));
        Assert.Equal("format_limit", Assert.Throws<PythonTextFormat.Error>(() => Render(JsonValue.Create("😀😀😀😀x"), limits: limits)).Category);
        Assert.Equal("format_limit", Assert.Throws<PythonTextFormat.Error>(() => Render(JsonValue.Create("x"), "{a:5}", limits)).Category);
        Assert.Equal("format_limit", Assert.Throws<PythonTextFormat.Error>(() => Render(JsonValue.Create("x"), "{a:.7}", limits)).Category);
    }

    [Fact]
    public void LiteralAndCumulativeOutputBudgetsAreCheckedBeforePadding()
    {
        using var context = new RuntimeNodeContext();
        var empty = new Dictionary<string, RuntimeValue>();
        Assert.Equal("😀😀", PythonTextFormat.Format("😀😀", empty, limits: new(2, 1, 2)));
        Assert.Equal("format_limit", Assert.Throws<PythonTextFormat.Error>(() => PythonTextFormat.Format("😀😀x", empty, limits: new(2, 1, 4))).Category);
        Assert.Equal("format_limit", Assert.Throws<PythonTextFormat.Error>(() => Render(JsonValue.Create("ab"), "{a}{a}{a}", new(32, 3, 5))).Category);
        Assert.Equal("format_limit", Assert.Throws<PythonTextFormat.Error>(() => Render(JsonValue.Create("a"), "{a:99999999999999999999}")).Category);
        Assert.Throws<ArgumentOutOfRangeException>(() => PythonTextFormat.Format("", empty, limits: new(-1)));
    }

    [Fact]
    public void InvalidUtf16IsRejectedEvenInPrecisionDiscardedText()
    {
        foreach (var invalid in new[] { "\ud800", "\udc00", "x\ud800y" })
        {
            Assert.Equal("unsupported_format_value", Assert.Throws<PythonTextFormat.Error>(() => Render(JsonValue.Create(invalid), "{a:.0}")).Category);
            Assert.Equal("unsupported_format_value", Assert.Throws<PythonTextFormat.Error>(() => Render(JsonValue.Create("a"), invalid)).Category);
        }
    }

    [Fact]
    public void CancellationPrecedesReadingAnyBorrowedValue()
    {
        using var context = new RuntimeNodeContext();
        var dead = context.Json(null); dead.Dispose();
        using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
        var error = Assert.ThrowsAny<OperationCanceledException>(() => PythonTextFormat.Format("{a}",
            new Dictionary<string, RuntimeValue> { ["a"] = dead }, cancellation.Token));
        Assert.Equal(cancellation.Token, error.CancellationToken);
    }

    [Fact]
    public void UnusedTypedValuesAreNotProjectedAndFailuresDoNotStealTheirOwner()
    {
        var resource = new Resource();
        using var context = new RuntimeNodeContext();
        var native = context.Own(resource);
        var values = new Dictionary<string, RuntimeValue> { ["a"] = native,
            ["b"] = context.Map(new Dictionary<string, RuntimeValue> { ["inner"] = context.Blocker("nested") }),
            ["c"] = context.List([native]), ["d"] = context.Json(new JsonArray(1, 2)) };
        Assert.Equal("constant", PythonTextFormat.Format("constant", values));
        foreach (var key in values.Keys)
            Assert.Equal("unsupported_format_value", Assert.Throws<PythonTextFormat.Error>(() => PythonTextFormat.Format("{" + key + "}", values)).Category);
        Assert.Equal(0, resource.Disposals); Assert.Same(resource, native.GetNative<Resource>());
        context.Dispose(); Assert.Equal(1, resource.Disposals);
    }

    [Fact]
    public async Task RealNodeResultSurvivesCallerScopeWithoutKeepingUnusedNativeAlive()
    {
        var resource = new Resource();
        using var caller = new RuntimeNodeContext(); using var outputScope = new RuntimeNodeContext();
        var inputs = new Dictionary<string, RuntimeValue>
        {
            ["values"] = caller.Map(new Dictionary<string, RuntimeValue> { ["a"] = caller.Own(resource) }),
            ["f_string"] = caller.Json(JsonValue.Create("constant"))
        };
        var output = await new StringFormatNode().ExecuteAsync(outputScope, inputs, CancellationToken.None);
        using var retained = Assert.Single(output.Result).Retain();
        caller.Dispose(); outputScope.Dispose();
        Assert.Equal(1, resource.Disposals); Assert.Equal("constant", retained.ToJson()!.GetValue<string>());
    }

    [Fact]
    public async Task RealEngineMapsValuesAndFormatUsingRepeatLast()
    {
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new Producer("Formats", c => c.List([c.Json(JsonValue.Create("{a}:{b}")), c.Json(JsonValue.Create("[{a}]"))]), true));
        registry.Register(new Producer("Words", c => c.List([c.Json(JsonValue.Create("one")), c.Json(JsonValue.Create("two")), c.Json(JsonValue.Create("three"))]), true));
        var prompt = JsonNode.Parse("""
            {"f":{"class_type":"Formats","inputs":{}},"w":{"class_type":"Words","inputs":{}},
             "p":{"class_type":"StringFormat","inputs":{"values.a":["w",0],"f_string":["f",0],"values.b":"B"}}}
            """)!.AsObject();
        using var result = await new EngineService(registry).ExecuteValuesAsync(prompt, ["p"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        Assert.Equal(new[] { "one:B", "[two]", "[three]" }, result.Outputs["p"][0].Select(v => v.ToJson()!.GetValue<string>()));
    }

    [Fact]
    public async Task UnusedDirectBlockerStillPreemptsRealFormatting()
    {
        var registry = BuiltInNodes.CreateRegistry();
        registry.Register(new Producer("Blocked", c => c.Blocker("stop")));
        using var result = await new EngineService(registry).ExecuteValuesAsync(JsonNode.Parse("""
            {"b":{"class_type":"Blocked","inputs":{}},"p":{"class_type":"StringFormat","inputs":{"values.a":["b",0],"f_string":"constant"}}}
            """)!.AsObject(), ["p"]);
        Assert.Equal("Execution Blocked: stop", Assert.Single(result.Diagnostics).Message);
        Assert.Equal(RuntimeValueKind.Blocker, Assert.Single(result.Outputs["p"][0]).Kind);
    }

    [Fact]
    public async Task CancellationAfterResolutionReleasesNativeProducerWithoutFormatting()
    {
        var resource = new Resource(); using var cancellation = new CancellationTokenSource();
        var registry = BuiltInNodes.CreateRegistry(); registry.Register(new Producer("Owned", c => c.Own(resource)));
        using var result = await new EngineService(registry).ExecuteValuesAsync(JsonNode.Parse("""
            {"s":{"class_type":"Owned","inputs":{}},"p":{"class_type":"StringFormat","inputs":{"values.a":["s",0],"f_string":"constant"}}}
            """)!.AsObject(), ["p"], e =>
            { if (e.Type == "executing" && e.NodeId == "p") cancellation.Cancel(); return ValueTask.CompletedTask; }, cancellation.Token);
        Assert.Equal("cancelled", result.Status); Assert.Empty(result.Outputs); Assert.Equal(1, resource.Disposals);
    }

    private sealed class Resource : IDisposable { public int Disposals { get; private set; } public void Dispose() => Disposals++; }
    private sealed class Producer(string name, Func<RuntimeNodeContext, RuntimeValue> produce, bool list = false) : IRuntimeNode
    {
        public NodeSchema Schema { get; } = new(name, name, "test", [], [new("*", IsList: list)]);
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) =>
            ValueTask.FromResult(new NodeExecutionOutput([produce(context)]));
    }
}
