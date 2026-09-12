using System.Globalization;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class TextComparisonNodeTests
{
    [Theory]
    [InlineData("Hello", "ell", true, true)]
    [InlineData("Hello", "ELL", true, false)]
    [InlineData("Hello", "ELL", false, true)]
    [InlineData("", "", true, true)]
    [InlineData("", "x", false, false)]
    [InlineData("😀É\0", "😀é", false, true)]
    [InlineData("İ", "i\u0307", false, true)]
    [InlineData("I", "ı", false, false)]
    [InlineData("Straße", "STRASSE", false, false)]
    [InlineData("é", "e\u0301", false, false)]
    [InlineData("ΟΣ", "ος", false, true)]
    [InlineData("ΟΣ", "οσ", false, false)]
    [InlineData("Σ", "σ", false, true)]
    [InlineData("Α\u0345Σ", "α\u0345ς", false, true)]
    public async Task ContainsUsesWholeStringPythonLowerWithoutNormalizationOrCaseFolding(
        string text, string substring, bool sensitive, bool expected) =>
        Assert.Equal(expected, await Run("StringContains", new()
        { ["string"] = text, ["substring"] = substring, ["case_sensitive"] = sensitive }));

    [Theory]
    [InlineData("Starts With", "İstanbul", "i\u0307", false, true)]
    [InlineData("Starts With", "İstanbul", "i", true, false)]
    [InlineData("Ends With", "ΑΣ", "ς", false, true)]
    [InlineData("Ends With", "ΑΣ", "Σ", false, false)]
    [InlineData("Equal", "ΟΣ", "ος", false, true)]
    [InlineData("Equal", "Οσ", "ος", false, false)]
    [InlineData("Equal", "Straße", "STRASSE", false, false)]
    [InlineData("Equal", "é", "e\u0301", false, false)]
    [InlineData("Equal", "😀", "😀", true, true)]
    [InlineData("Starts With", "", "", false, true)]
    [InlineData("Ends With", "", "", true, true)]
    [InlineData("Equal", "", "", true, true)]
    public async Task CompareModesKeepOrdinalSemanticsAfterIndependentWholeStringLower(
        string mode, string a, string b, bool sensitive, bool expected) =>
        Assert.Equal(expected, await Run("StringCompare", new()
        { ["string_a"] = a, ["string_b"] = b, ["mode"] = mode, ["case_sensitive"] = sensitive }));

    [Fact]
    public async Task CasingDoesNotDependOnCurrentCulture()
    {
        var original = CultureInfo.CurrentCulture;
        try
        {
            foreach (string name in new[] { "en-US", "tr-TR", "el-GR" })
            {
                CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo(name);
                Assert.True(await Run("StringCompare", new()
                { ["string_a"] = "IİΟΣ", ["string_b"] = "ii\u0307ος", ["mode"] = "Equal", ["case_sensitive"] = false }));
            }
        }
        finally { CultureInfo.CurrentCulture = original; }
    }

    [Fact]
    public async Task EngineMapsTwoRealCreateListsAndRepeatsTheShorterColumn()
    {
        var prompt = JsonNode.Parse("""
            {"a":{"class_type":"CreateList","inputs":{"inputs.input0":"ΟΣ","inputs.input1":"abc","inputs.input2":"ABC"}},
             "b":{"class_type":"CreateList","inputs":{"inputs.input0":"ος","inputs.input1":"abc"}},
             "p":{"class_type":"StringCompare","inputs":{"string_a":["a",0],"string_b":["b",0],"mode":"Equal","case_sensitive":false}}}
            """)!.AsObject();
        using var result = await new EngineService(BuiltInNodes.CreateRegistry()).ExecuteValuesAsync(prompt, ["p"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        Assert.Equal(new[] { true, true, true }, result.Outputs["p"][0].Select(v => v.ToJson()!.GetValue<bool>()));
    }

    [Fact]
    public async Task InvalidModeIsRejectedBeforeExecution()
    {
        using var result = await new EngineService(BuiltInNodes.CreateRegistry()).ExecuteValuesAsync(Prompt("StringCompare", new()
        { ["string_a"] = "a", ["string_b"] = "a", ["mode"] = "equal", ["case_sensitive"] = true }), ["p"]);
        Assert.NotEqual("success", result.Status); Assert.Empty(result.Outputs);
        Assert.Contains(result.Diagnostics, d => d.InputName == "mode");
    }

    [Theory]
    [InlineData("StringContains", true)]
    [InlineData("StringContains", false)]
    [InlineData("StringCompare", true)]
    [InlineData("StringCompare", false)]
    public async Task BothCasePathsRejectInvalidUtf16RatherThanMatchingHalfOfAScalar(string type, bool sensitive)
    {
        using var context = new RuntimeNodeContext();
        bool contains = type == "StringContains";
        var inputs = new Dictionary<string, RuntimeValue>
        {
            [contains ? "string" : "string_a"] = context.Json(JsonValue.Create(((char)0xd800).ToString())),
            [contains ? "substring" : "string_b"] = context.Json(JsonValue.Create("")),
            ["case_sensitive"] = context.Json(JsonValue.Create(sensitive)),
            ["mode"] = context.Json(JsonValue.Create("Starts With"))
        };
        IRuntimeNode node = contains ? new StringContainsNode() : new StringCompareNode();
        var error = await Assert.ThrowsAsync<ArgumentException>(() => node.ExecuteAsync(context, inputs, CancellationToken.None).AsTask());
        Assert.Contains("unsupported_unicode_value", error.Message);
    }

    [Theory]
    [InlineData("StringContains")]
    [InlineData("StringCompare")]
    public async Task CancellationPrecedesReadingBorrowedArguments(string type)
    {
        using var borrowed = new RuntimeNodeContext(); using var output = new RuntimeNodeContext();
        var dead = borrowed.Json(JsonValue.Create("dead")); borrowed.Dispose();
        using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
        IRuntimeNode node = type == "StringContains" ? new StringContainsNode() : new StringCompareNode();
        var inputs = new Dictionary<string, RuntimeValue>
        { ["string"] = dead, ["substring"] = dead, ["string_a"] = dead, ["string_b"] = dead, ["case_sensitive"] = dead };
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => node.ExecuteAsync(output, inputs, cancellation.Token).AsTask());
        Assert.Equal(cancellation.Token, error.CancellationToken);
    }

    private static JsonObject Prompt(string type, JsonObject inputs) => new()
    { ["p"] = new JsonObject { ["class_type"] = type, ["inputs"] = inputs } };

    private static async Task<bool> Run(string type, JsonObject inputs)
    {
        using var result = await new EngineService(BuiltInNodes.CreateRegistry()).ExecuteValuesAsync(Prompt(type, inputs), ["p"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        return Assert.Single(result.Outputs["p"][0]).ToJson()!.GetValue<bool>();
    }
}
