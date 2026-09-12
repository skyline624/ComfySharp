using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Nodes;
using ComfySharp.Host;
using Microsoft.AspNetCore.Mvc.Testing;

namespace ComfySharp.Host.Tests;

/// <summary>HTTP materialization must preserve case-sensitive prompt dictionary keys.
/// These are product regression tests through the real Host, not a new source oracle.</summary>
public sealed class HttpPromptCaseSensitivityTests
{
    [Fact]
    public async Task CaseDistinctNodeIdsExecuteBothRealProducers()
    {
        var prompt = JsonNode.Parse("""
            {"item":{"class_type":"PrimitiveString","inputs":{"value":"lower"}},
             "ITEM":{"class_type":"PrimitiveString","inputs":{"value":"upper"}},
             "join":{"class_type":"StringConcatenate","inputs":{"string_a":["item",0],"string_b":["ITEM",0],"delimiter":"/"}},
             "preview":{"class_type":"PreviewAny","inputs":{"source":["join",0]}}}
            """)!.AsObject();
        Assert.Equal("lower/upper", await ExecutePreview(prompt));
    }

    [Fact]
    public async Task WrongCaseInputDoesNotSatisfyRequiredSourceOrEnterTheQueue()
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var response = await client.PostAsJsonAsync("/prompt", Request(Preview(new JsonObject
        {
            ["SOURCE"] = "wrong case"
        })), timeout.Token);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var error = await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token);
        Assert.Contains("required_input_missing", error!.ToJsonString());
        var jobs = await client.GetFromJsonAsync<JsonObject>("/api/jobs", timeout.Token);
        Assert.Empty(jobs!["jobs"]!.AsArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ExactInputWinsIndependentlyOfCaseDistinctSiblingOrder(bool uppercaseFirst)
    {
        var inputs = new JsonObject();
        if (uppercaseFirst) inputs["SOURCE"] = "uppercase sibling";
        inputs["source"] = "exact source";
        if (!uppercaseFirst) inputs["SOURCE"] = "uppercase sibling";
        Assert.Equal("exact source", await ExecutePreview(Preview(inputs)));
    }

    [Theory]
    [InlineData("{\"__value__\":1,\"__VALUE__\":2}")]
    [InlineData("{\"__VALUE__\":2,\"__value__\":1}")]
    public async Task OnlyExactReservedKeyIsUnwrappedWhenBothSpellingsExist(string literal)
    {
        var prompt = Preview(new JsonObject { ["source"] = JsonNode.Parse(literal) });
        Assert.Equal("1", await ExecutePreview(prompt));
    }

    [Fact]
    public async Task OrdinaryNestedDictionaryRetainsCaseDistinctKeysWithoutRecursiveUnwrap()
    {
        const string literal = """
            {"Field":1,"field":2,"nested":{"__VALUE__":3,"__value__":4}}
            """;
        var expected = JsonNode.Parse(literal);
        string preview = await ExecutePreview(Preview(new JsonObject { ["source"] = expected!.DeepClone() }));
        Assert.True(JsonNode.DeepEquals(expected, JsonNode.Parse(preview)), preview);
    }

    private static JsonObject Preview(JsonObject inputs) => new()
    {
        ["preview"] = new JsonObject { ["class_type"] = "PreviewAny", ["inputs"] = inputs }
    };

    private static JsonObject Request(JsonObject prompt) => new()
    {
        ["prompt"] = prompt.DeepClone(), ["partial_execution_targets"] = new JsonArray("preview")
    };

    private static async Task<string> ExecutePreview(JsonObject prompt)
    {
        string before = prompt.ToJsonString();
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var response = await client.PostAsJsonAsync("/prompt", Request(prompt), timeout.Token);
        response.EnsureSuccessStatusCode();
        string id = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
        while (true)
        {
            var job = await client.GetFromJsonAsync<JsonObject>($"/api/jobs/{id}", timeout.Token);
            string state = job!["status"]!.GetValue<string>();
            if (state == "completed") break;
            Assert.DoesNotContain(state, new[] { "failed", "cancelled" });
            await Task.Delay(10, timeout.Token);
        }
        var history = await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token);
        var outputs = history![id]!["outputs"]!.AsObject();
        Assert.Equal("preview", Assert.Single(outputs).Key);
        Assert.Equal(before, prompt.ToJsonString());
        return Assert.Single(outputs["preview"]!["text"]!.AsArray())!.GetValue<string>();
    }
}
