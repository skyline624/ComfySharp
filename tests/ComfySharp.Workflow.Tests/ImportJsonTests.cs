using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class ImportJsonTests
{
    [Theory]
    [InlineData("NaN", "null")]
    [InlineData("Infinity", "null")]
    [InlineData("-Infinity", "null")]
    [InlineData("[NaN,Infinity,-Infinity]", "[null,null,null]")]
    [InlineData("{\"nested\":[{\"value\":NaN}]}", "{\"nested\":[{\"value\":null}]}")]
    [InlineData("{\"NaN\":Infinity,\"Infinity\":-Infinity}", "{\"NaN\":null,\"Infinity\":null}")]
    [InlineData("{\"text\":\"NaN Infinity -Infinity\",\"value\":NaN}", "{\"text\":\"NaN Infinity -Infinity\",\"value\":null}")]
    [InlineData("{\"text\":\"\\\"NaN\\\"\",\"value\":Infinity}", "{\"text\":\"\\\"NaN\\\"\",\"value\":null}")]
    [InlineData("[\"ends\\\\\",NaN]", "[\"ends\\\\\",null]")]
    [InlineData("[\"\\u0022Infinity\\u0022\",NaN]", "[\"\\u0022Infinity\\u0022\",null]")]
    [InlineData("[\n NaN,\tInfinity,\r-Infinity ]", "[null,null,null]")]
    public void Replaces_only_bare_tokens_and_reports_once(string json, string expected)
    {
        var warnings = new List<string>(); var parsed = ImportJson.Parse(json, warnings.Add);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(expected), parsed));
        Assert.Equal(ImportJson.NonFiniteWarning, Assert.Single(warnings));
    }

    [Theory]
    [InlineData("{\"NaN\":\"Infinity\",\"n\":1e10,\"b\":true}")]
    [InlineData("[null,false,{},[],\"-Infinity\"]")]
    [InlineData("\"escaped\\nNaN\\\\Infinity\"")]
    [InlineData("1e400")]
    public void Strict_json_is_not_coerced_or_warned(string json)
    {
        var warnings = new List<string>(); var result = ImportJson.Parse(json, warnings.Add);
        Assert.Empty(warnings); Assert.True(JsonNode.DeepEquals(JsonNode.Parse(json), result));
    }

    [Theory]
    [InlineData("NaN123")]
    [InlineData("InfinitySuffix")]
    [InlineData("-Infinity0")]
    [InlineData("NaNNaN")]
    [InlineData("1-Infinity")]
    [InlineData("1.5-Infinity")]
    [InlineData("+Infinity")]
    [InlineData("-NaN")]
    [InlineData("NaN.2")]
    [InlineData("Infinity.123")]
    [InlineData("_NaN")]
    [InlineData("NaN_")]
    [InlineData("éNaN")]
    [InlineData("NaNé")]
    [InlineData("fooNaNbar")]
    [InlineData("{\"x\":NaN,}")]
    [InlineData("[NaN] garbage")]
    [InlineData("{\"x\":\"unfinished\\")]
    [InlineData("{\"x\":\"NaN\nInfinity\"}")]
    public void Other_syntax_errors_remain_errors(string json) => Assert.ThrowsAny<JsonException>(() => ImportJson.Parse(json));

    [Fact]
    public void Warnings_are_per_call_and_fallback_does_not_mask_later_errors()
    {
        var warnings = new List<string>();
        ImportJson.Parse("[NaN,Infinity]", warnings.Add); ImportJson.Parse("-Infinity", warnings.Add);
        Assert.Equal(2, warnings.Count);
        Assert.ThrowsAny<JsonException>(() => ImportJson.Parse("[NaN,]", warnings.Add)); Assert.Equal(3, warnings.Count);
        Assert.ThrowsAny<JsonException>(() => ImportJson.Parse("{invalid}", warnings.Add)); Assert.Equal(3, warnings.Count);
    }

    [Fact]
    public void Long_arrays_and_escaped_strings_preserve_order_without_regex_backtracking()
    {
        string json = "[" + string.Join(",", Enumerable.Range(0, 10000).Select(i => i % 2 == 0 ? "NaN" : "\"NaN\\\"Infinity\"")) + "]";
        var result = ImportJson.Parse(json)!.AsArray(); Assert.Equal(10000, result.Count);
        for (int i = 0; i < result.Count; i++)
            if (i % 2 == 0) Assert.Null(result[i]); else Assert.Equal("NaN\"Infinity", result[i]!.GetValue<string>());
    }

    [Theory]
    [InlineData("not JSON")]
    [InlineData("[]")]
    [InlineData("{\"version\":\"wrong\",\"nodes\":[]}")]
    [InlineData("{\"version\":2,\"nodes\":[]}")]
    public void Invalid_png_graph_falls_back_to_api_with_diagnostics(string workflow)
    {
        var warnings = new List<string>();
        var metadata = new PngMetadata(new Dictionary<string,string> { ["workflow"] = workflow,
            ["prompt"] = """{"p":{"class_type":"PreviewAny","inputs":{"source":NaN}}}""" }, []);
        var document = PngWorkflowImport.ReadWorkflow(metadata, reportWarning: warnings.Add);
        Assert.Equal("PreviewAny", Assert.Single(document.Nodes).Type);
        Assert.Equal(2, warnings.Count); Assert.Contains("trying API prompt", warnings[0]);
        Assert.Equal(ImportJson.NonFiniteWarning, warnings[1]);
        var compiled = PromptCompiler.Compile(document); Assert.True(compiled.Success);
        Assert.True(compiled.Prompt!["p"]!["inputs"]!.AsObject().ContainsKey("source")); Assert.Null(compiled.Prompt["p"]!["inputs"]!["source"]);
        Assert.Equal(workflow, metadata.Text["workflow"]);
    }

    [Fact]
    public void Valid_graph_precedes_prompt_and_document_parser_remains_strict()
    {
        var warnings = new List<string>(); const string graph = """{"version":0.4,"nodes":[],"extension":NaN}""";
        var document = PngWorkflowImport.ReadWorkflow(new(new Dictionary<string,string> { ["workflow"] = graph, ["prompt"] = "not parsed" }, []), reportWarning: warnings.Add);
        Assert.Empty(document.Nodes); Assert.Null(document.Snapshot()["extension"]); Assert.Single(warnings);
        Assert.ThrowsAny<JsonException>(() => WorkflowDocument.Parse(graph));
    }
}
