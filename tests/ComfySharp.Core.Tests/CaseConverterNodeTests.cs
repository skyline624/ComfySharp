using System.Globalization;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class CaseConverterNodeTests
{
    [Theory]
    [InlineData("UPPERCASE", "", "")]
    [InlineData("lowercase", "", "")]
    [InlineData("Capitalize", "", "")]
    [InlineData("Title Case", "", "")]
    [InlineData("UPPERCASE", "Straße ﬃ ǳ", "STRASSE FFI Ǳ")]
    [InlineData("UPPERCASE", "iıİ😀\0", "IIİ😀\0")]
    [InlineData("lowercase", "İΟΣ", "i\u0307ος")]
    [InlineData("lowercase", "𐐀😀\0", "𐐨😀\0")]
    [InlineData("Capitalize", "ǳABC", "ǲabc")]
    [InlineData("Capitalize", "ßABC", "Ssabc")]
    [InlineData("Capitalize", "ﬃABC", "Ffiabc")]
    [InlineData("Capitalize", "#ABC", "#abc")]
    [InlineData("Capitalize", "AΣ", "Aς")]
    [InlineData("Capitalize", "𐐨ABC", "𐐀abc")]
    [InlineData("Title Case", "they're 12abc", "They'Re 12Abc")]
    [InlineData("Title Case", "ǳABC ßABC", "ǲabc Ssabc")]
    [InlineData("Title Case", "ΟΣ", "Ος")]
    [InlineData("Title Case", "A\u0345Σ", "A\u0345ς")]
    [InlineData("Title Case", "A\u0301Σ", "A\u0301Σ")]
    [InlineData("Title Case", "😀ABC\0DEF", "😀Abc\0Def")]
    public async Task ModesPreserveFullUnicodeMappingsAndOriginalContext(string mode, string text, string expected) =>
        Assert.Equal(expected, await Run(text, mode));

    [Fact]
    public async Task CasingIsIndependentOfProcessCulture()
    {
        var original = CultureInfo.CurrentCulture;
        try
        {
            foreach (string name in new[] { "en-US", "tr-TR", "el-GR" })
            {
                CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo(name);
                Assert.Equal("IIİ", await Run("iıİ", "UPPERCASE"));
                Assert.Equal("Iς", await Run("iΣ", "Capitalize"));
            }
        }
        finally { CultureInfo.CurrentCulture = original; }
    }

    [Fact]
    public async Task TwoRealCreateListsMapModesAndRepeatTheLastText()
    {
        var prompt = JsonNode.Parse("""
            {"texts":{"class_type":"CreateList","inputs":{"inputs.input0":"ßABC","inputs.input1":"AΣ"}},
             "modes":{"class_type":"CreateList","inputs":{"inputs.input0":"UPPERCASE","inputs.input1":"lowercase","inputs.input2":"Capitalize","inputs.input3":"Title Case"}},
             "p":{"class_type":"CaseConverter","inputs":{"string":["texts",0],"mode":["modes",0]}}}
            """)!.AsObject();
        string before = prompt.ToJsonString();
        using var result = await new EngineService(BuiltInNodes.CreateRegistry()).ExecuteValuesAsync(prompt, ["p"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        Assert.Equal(new[] { "SSABC", "aς", "Aς", "Aς" }, result.Outputs["p"][0].Select(v => v.ToJson()!.GetValue<string>()));
        Assert.Equal(before, prompt.ToJsonString());
    }

    [Fact]
    public async Task InvalidComboIsDiagnosedBeforeAnyNodeOutput()
    {
        using var result = await new EngineService(BuiltInNodes.CreateRegistry()).ExecuteValuesAsync(Prompt("ABC", "Uppercase"), ["p"]);
        Assert.NotEqual("success", result.Status); Assert.Empty(result.Outputs);
        Assert.Contains(result.Diagnostics, d => d.InputName == "mode");
    }

    private static async Task<string> Run(string text, string mode)
    {
        var prompt = Prompt(text, mode); string before = prompt.ToJsonString();
        using var result = await new EngineService(BuiltInNodes.CreateRegistry()).ExecuteValuesAsync(prompt, ["p"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        Assert.Equal(before, prompt.ToJsonString());
        return Assert.Single(Assert.Single(result.Outputs["p"])).ToJson()!.GetValue<string>();
    }

    private static JsonObject Prompt(string text, string mode) => new()
    { ["p"] = new JsonObject { ["class_type"] = "CaseConverter", ["inputs"] = new JsonObject { ["string"] = text, ["mode"] = mode } } };

    [Theory]
    [InlineData("UPPERCASE")]
    [InlineData("lowercase")]
    [InlineData("Capitalize")]
    [InlineData("Title Case")]
    public async Task EveryModeRejectsIsolatedUtf16(string mode)
    {
        using var context = new RuntimeNodeContext();
        var inputs = new Dictionary<string, RuntimeValue>
        { ["string"] = context.Json(JsonValue.Create("A" + (char)0xd800)), ["mode"] = context.Json(JsonValue.Create(mode)) };
        var error = await Assert.ThrowsAsync<ArgumentException>(() => new CaseConverterNode().ExecuteAsync(context, inputs, CancellationToken.None).AsTask());
        Assert.Contains("unsupported_unicode_value", error.Message);
    }

    [Fact]
    public async Task CancellationPrecedesBorrowedValueAccess()
    {
        using var borrowed = new RuntimeNodeContext(); using var output = new RuntimeNodeContext();
        var value = borrowed.Json(JsonValue.Create("A")); borrowed.Dispose();
        var inputs = new Dictionary<string, RuntimeValue> { ["string"] = value, ["mode"] = value };
        using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => new CaseConverterNode().ExecuteAsync(output, inputs, cancellation.Token).AsTask());
    }
}
