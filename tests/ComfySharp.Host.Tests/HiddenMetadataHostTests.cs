using System.Net.Http.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Host;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;

namespace ComfySharp.Host.Tests;

public sealed class HiddenMetadataHostTests
{
    [Theory]
    [InlineData("/prompt", false)]
    [InlineData("/prompt", true)]
    [InlineData("/api/prompt", false)]
    [InlineData("/api/prompt", true)]
    public async Task Submitted_metadata_reaches_node_and_history_without_unrelated_extra_data(string endpoint, bool metadata)
    {
        await using var factory = Factory();
        using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var hub = factory.Services.GetRequiredService<EventHub>();
        var subscriber = hub.Subscribe("metadata-client");
        try
        {
            var prompt = Prompt();
            var png = JsonNode.Parse("""{"workflow":{"version":1,"unknown":{"preserve":[17,"é",null]}},"note":"test"}""");
            var body = new JsonObject { ["prompt"] = prompt.DeepClone(), ["client_id"] = "metadata-client" };
            if (metadata) body["extra_data"] = new JsonObject { ["extra_pnginfo"] = png!.DeepClone(), ["unrelated"] = "not passed" };
            using var submitted = await client.PostAsJsonAsync(endpoint, body, timeout.Token);
            submitted.EnsureSuccessStatusCode();
            string id = (await submitted.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
            JsonObject? output = null;
            while (true)
            {
                var message = JsonNode.Parse(await subscriber.Reader.ReadAsync(timeout.Token))!;
                string type = message["type"]!.GetValue<string>();
                if (type == "executed") output = message["data"]!["output"]!.DeepClone().AsObject();
                Assert.NotEqual("execution_error", type);
                if (type == "execution_success") break;
            }
            Assert.NotNull(output);
            var capture = output["captures"]![0]!;
            Assert.True(JsonNode.DeepEquals(prompt, capture["prompt"]));
            Assert.True(JsonNode.DeepEquals(metadata ? png : null, capture["extra_pnginfo"]));
            Assert.Equal("nested:9", capture["unique_id"]!.GetValue<string>());
            Assert.False(capture.AsObject().ContainsKey("unrelated"));
            var history = (await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token))![id]!;
            Assert.True(history["status"]!["completed"]!.GetValue<bool>());
            Assert.True(JsonNode.DeepEquals(output, history["outputs"]!["nested:9"]));
            Assert.True(JsonNode.DeepEquals(metadata ? png : null, history["prompt"]![3]!["extra_pnginfo"]));
            if (metadata) Assert.Equal("not passed", history["prompt"]![3]!["unrelated"]!.GetValue<string>());
        }
        finally { hub.Remove(subscriber.Id); }
    }

    [Fact]
    public async Task Queued_jobs_capture_metadata_before_caller_changes_it()
    {
        var registry = new NodeRegistry(); registry.Register(new MetadataNode());
        var hub = new EventHub();
        using var queue = new JobQueue(new EngineService(registry), hub);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var subscription = hub.Subscribe("queued");
        var first = Prompt(); var extra = new JsonObject { ["extra_pnginfo"] = new JsonObject { ["tag"] = "first" } };
        queue.Enqueue(first, ["nested:9"], "queued", null, false, "first-job", extra);
        first["nested:9"]!["_meta"]!["title"] = "second";
        extra["extra_pnginfo"]!["tag"] = "second";
        queue.Enqueue(first, ["nested:9"], "queued", null, false, "second-job", extra);
        first.Clear(); extra.Clear();
        await queue.StartAsync(timeout.Token);
        try
        {
            var results = new Dictionary<string, JsonNode?>();
            while (results.Count < 2)
            {
                var message = JsonNode.Parse(await subscription.Reader.ReadAsync(timeout.Token))!;
                Assert.NotEqual("execution_error", message["type"]!.GetValue<string>());
                if (message["type"]!.GetValue<string>() == "executed")
                    results.Add(message["data"]!["prompt_id"]!.GetValue<string>(), message["data"]!["output"]!["captures"]![0]);
            }
            Assert.Equal("first", results["first-job"]!["extra_pnginfo"]!["tag"]!.GetValue<string>());
            Assert.Equal("second", results["second-job"]!["extra_pnginfo"]!["tag"]!.GetValue<string>());
            Assert.Equal("original", results["first-job"]!["prompt"]!["nested:9"]!["_meta"]!["title"]!.GetValue<string>());
            Assert.Equal("second", results["second-job"]!["prompt"]!["nested:9"]!["_meta"]!["title"]!.GetValue<string>());
        }
        finally { await queue.StopAsync(timeout.Token); hub.Remove(subscription.Id); }
    }

    private static WebApplicationFactory<Program> Factory()
    {
        var registry = new NodeRegistry(); registry.Register(new MetadataNode());
        return new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            builder.ConfigureServices(services => services.AddSingleton(new EngineService(registry))));
    }
    private static JsonObject Prompt() => JsonNode.Parse("""
        {"nested:9":{"class_type":"TestMetadata","inputs":{},"_meta":{"title":"original"}}}
        """)!.AsObject();
    private sealed class MetadataNode : IRuntimeNode
    {
        public NodeSchema Schema { get; } = new("TestMetadata", "TestMetadata", "test", [], [], OutputNode: true,
            HiddenInputs: [new("prompt", "PROMPT"), new("extra_pnginfo", "EXTRA_PNGINFO"), new("unique_id", "UNIQUE_ID")]);
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) => ValueTask.FromResult(
                new NodeExecutionOutput([], new JsonObject { ["captures"] = new JsonArray(
                    new JsonObject(inputs.Select(p => new KeyValuePair<string, JsonNode?>(p.Key, p.Value.ToJson())))) }));
    }
}
