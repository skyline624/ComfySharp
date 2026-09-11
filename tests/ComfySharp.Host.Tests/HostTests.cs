using System.Net;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Host;
using ComfySharp.Nodes;
using ComfySharp.Storage;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;

namespace ComfySharp.Host.Tests;

public sealed class HostTests
{
    [Theory]
    [InlineData("/interrupt")]
    [InlineData("/api/interrupt")]
    public async Task ChunkedInterruptPreservesCompletedJobIdentityAndBodylessCancelsActive(string endpoint)
    {
        var registry = BuiltInNodes.CreateRegistry();
        var block = new BlockingNode();
        registry.Register(block);
        await using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            builder.ConfigureServices(services => services.AddSingleton(new EngineService(registry))));
        using var client = factory.CreateClient();
        var queue = factory.Services.GetRequiredService<JobQueue>();
        var first = queue.Enqueue(JsonNode.Parse("""{"1":{"class_type":"PrimitiveInt","inputs":{"value":1}}}""")!.AsObject(), ["1"], null, null, false, null, null).Id;
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        while (queue.JobSnapshot(first)!["status"]!.GetValue<string>() != "completed") await Task.Delay(10, timeout.Token);
        var second = queue.Enqueue(JsonNode.Parse("""{"1":{"class_type":"TestBlocking","inputs":{}}}""")!.AsObject(), ["1"], null, null, false, null, null).Id;
        await block.FirstStarted.Task.WaitAsync(timeout.Token);
        using var request = new HttpRequestMessage(HttpMethod.Post, endpoint)
        {
            Content = new UnknownLengthContent(new JsonObject { ["prompt_id"] = first }.ToJsonString())
        };
        request.Headers.TransferEncodingChunked = true;
        Assert.Null(request.Content.Headers.ContentLength);
        using var response = await client.SendAsync(request, timeout.Token);
        response.EnsureSuccessStatusCode();
        Assert.False((await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["cancelled"]!.GetValue<bool>());
        foreach (var invalid in new[] { "{\"prompt_id\":null}", "{\"prompt_id\":3}", "{\"prompt_id\":[]}", "[1]", "{broken" })
        {
            using var bad = await client.PostAsync(endpoint, new UnknownLengthContent(invalid), timeout.Token);
            Assert.Equal(HttpStatusCode.BadRequest, bad.StatusCode);
            Assert.Equal("invalid_request", (await bad.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["error"]!["type"]!.GetValue<string>());
        }
        Assert.Equal("in_progress", queue.JobSnapshot(second)!["status"]!.GetValue<string>());
        using var bodyless = await client.PostAsync(endpoint, null, timeout.Token);
        Assert.True((await bodyless.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["cancelled"]!.GetValue<bool>());
        while (queue.JobSnapshot(second)!["status"]!.GetValue<string>() != "cancelled") await Task.Delay(10, timeout.Token);
    }

    public static IEnumerable<object[]> InvalidFields()
    {
        foreach (var field in new[] { "partial_execution_targets", "prompt_id", "client_id", "number", "front", "extra_data" })
        foreach (var value in new[] { "null", "[]", "{}" })
            if (!((field == "partial_execution_targets" && value == "[]") || (field == "extra_data" && value == "{}")))
                yield return ["/prompt", field, value];
        yield return ["/prompt", "partial_execution_targets", "[null]"];
        yield return ["/prompt", "partial_execution_targets", "[1]"];
        yield return ["/prompt", "number", "\"1\""];
        yield return ["/prompt", "front", "1"];
        foreach (var endpoint in new[] { "/queue", "/history" })
        {
            foreach (var value in new[] { "null", "1", "[]", "\"true\"" }) yield return [endpoint, "clear", value];
            foreach (var value in new[] { "null", "1", "{}", "[null]", "[1]" }) yield return [endpoint, "delete", value];
        }
    }

    [Theory]
    [MemberData(nameof(InvalidFields))]
    public async Task InvalidFieldShapesReturnStructured400WithoutEnqueue(string endpoint, string field, string json)
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        var request = JsonNode.Parse("""{"prompt":{"1":{"class_type":"PrimitiveInt","inputs":{"value":1}}},"partial_execution_targets":["1"]}""")!.AsObject();
        request[field] = JsonNode.Parse(json);
        using var response = await client.PostAsJsonAsync(endpoint, request);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var error = await response.Content.ReadFromJsonAsync<JsonObject>();
        Assert.Equal("invalid_request", error!["error"]!["type"]!.GetValue<string>());
        Assert.Contains(field, error["error"]!["message"]!.GetValue<string>());
        Assert.Empty(factory.Services.GetRequiredService<JobQueue>().Jobs());
    }

    private sealed class UnknownLengthContent(string json) : HttpContent
    {
        protected override bool TryComputeLength(out long length) { length = 0; return false; }
        protected override Task SerializeToStreamAsync(Stream stream, TransportContext? context) =>
            stream.WriteAsync(Encoding.UTF8.GetBytes(json)).AsTask();
    }

    [Fact]
    public async Task IndependentHostValidatesBeforeQueueThenRunsRealTextGraph()
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        var health = await client.GetFromJsonAsync<JsonObject>("/health");
        Assert.Equal("ok", health!["status"]!.GetValue<string>());
        var bad = await client.PostAsJsonAsync("/prompt", JsonNode.Parse("""{"prompt":{"1":{"class_type":"MissingPythonNode","inputs":{}}},"partial_execution_targets":["1"]}"""));
        Assert.Equal(HttpStatusCode.BadRequest, bad.StatusCode);
        var before = await client.GetFromJsonAsync<JsonObject>("/queue");
        Assert.Empty(before!["queue_pending"]!.AsArray());
        var response = await client.PostAsJsonAsync("/api/prompt", JsonNode.Parse("""
            {"prompt":{"1":{"class_type":"PrimitiveString","inputs":{"value":"Bonjour 🌍"}},
                       "2":{"class_type":"StringLength","inputs":{"string":["1",0]}}},
             "partial_execution_targets":["2"]}
            """));
        response.EnsureSuccessStatusCode();
        var accepted = await response.Content.ReadFromJsonAsync<JsonObject>();
        var id = accepted!["prompt_id"]!.GetValue<string>();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        JsonObject? job;
        do
        {
            job = await client.GetFromJsonAsync<JsonObject>($"/api/jobs/{id}", timeout.Token);
            if (job!["status"]!.GetValue<string>() == "completed") break;
            await Task.Delay(10, timeout.Token);
        } while (true);
        var history = await client.GetFromJsonAsync<JsonObject>($"/history/{id}");
        Assert.Equal(9, history![id]!["outputs"]!["2"]![0]![0]!.GetValue<int>());
        var cancel = await client.PostAsync($"/api/jobs/{id}/cancel", null);
        Assert.False((await cancel.Content.ReadFromJsonAsync<JsonObject>())!["cancelled"]!.GetValue<bool>());
    }

    [Fact]
    public async Task BrowserFromAnotherOriginCannotSubmitToLocalHost()
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        client.DefaultRequestHeaders.Add("Origin", "https://unrelated.example");
        Assert.Equal(HttpStatusCode.Forbidden, (await client.GetAsync("/health")).StatusCode);
    }

    [Fact]
    public async Task WebSocketStartsWithQueueStatusAndSessionId()
    {
        await using var factory = new WebApplicationFactory<Program>();
        var wsClient = factory.Server.CreateWebSocketClient();
        using var socket = await wsClient.ConnectAsync(new Uri("ws://localhost/ws?clientId=test-session"), CancellationToken.None);
        var bytes = new byte[8192];
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        var received = await socket.ReceiveAsync(bytes, timeout.Token);
        var message = JsonNode.Parse(Encoding.UTF8.GetString(bytes, 0, received.Count));
        Assert.Equal("status", message!["type"]!.GetValue<string>());
        Assert.Equal("test-session", message["data"]!["sid"]!.GetValue<string>());
        Assert.Equal(0, message["data"]!["status"]!["exec_info"]!["queue_remaining"]!.GetValue<int>());
        await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", timeout.Token);
    }

    [Fact]
    public async Task CancellingPreviousJobCannotInterruptSuccessor()
    {
        var registry = BuiltInNodes.CreateRegistry();
        var block = new BlockingNode();
        registry.Register(block);
        using var queue = new JobQueue(new EngineService(registry), new EventHub());
        var prompt = JsonNode.Parse("""{"1":{"class_type":"TestBlocking","inputs":{}}}""")!.AsObject();
        var first = queue.Enqueue(prompt, ["1"], null, null, false, null, null).Id;
        var second = queue.Enqueue(prompt, ["1"], null, null, false, null, null).Id;
        await queue.StartAsync(CancellationToken.None);
        await block.FirstStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(queue.Cancel(first));
        await block.SecondStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False(queue.Cancel(first));
        Assert.Equal("in_progress", queue.JobSnapshot(second)!["status"]!.GetValue<string>());
        block.FinishSecond.TrySetResult();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        while (queue.JobSnapshot(second)!["status"]!.GetValue<string>() != "completed") await Task.Delay(10, timeout.Token);
        Assert.Equal("cancelled", queue.JobSnapshot(first)!["status"]!.GetValue<string>());
        await queue.StopAsync(timeout.Token);
    }

    [Fact]
    public async Task PriorityAndCancelledPendingEntriesAreRespected()
    {
        using var queue = new JobQueue(new EngineService(BuiltInNodes.CreateRegistry()), new EventHub());
        var prompt = JsonNode.Parse("""{"1":{"class_type":"PrimitiveInt","inputs":{"value":3}}}""")!.AsObject();
        var later = queue.Enqueue(prompt, ["1"], null, 10, false, null, null).Id;
        var first = queue.Enqueue(prompt, ["1"], null, 2, true, null, null).Id;
        Assert.Equal(first, queue.QueueSnapshot()["queue_pending"]![0]![1]!.GetValue<string>());
        Assert.True(queue.Cancel(later));
        Assert.Single(queue.QueueSnapshot()["queue_pending"]!.AsArray());
        await queue.StartAsync(CancellationToken.None);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        while (queue.JobSnapshot(first)!["status"]!.GetValue<string>() != "completed") await Task.Delay(10, timeout.Token);
        await queue.StopAsync(timeout.Token);
    }

    [Fact]
    public void PreviewFramesUseUpstreamBigEndianHeaders()
    {
        Assert.Equal(new byte[] { 0,0,0,1,0,0,0,2,42 }, PreviewCodec.Image([42], png: true));
        var frame = PreviewCodec.ImageWithMetadata([17,18], new JsonObject { ["node_id"] = "1:2" });
        Assert.Equal(4u, System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(frame));
        var length = (int)System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(frame.AsSpan(4));
        Assert.Equal("1:2", JsonNode.Parse(frame.AsSpan(8, length))!["node_id"]!.GetValue<string>());
        Assert.Equal(new byte[] { 17,18 }, frame[(8 + length)..]);
    }

    private sealed class BlockingNode : INode
    {
        private int calls;
        public TaskCompletionSource FirstStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource SecondStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource FinishSecond { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public NodeSchema Schema { get; } = new("TestBlocking", "Blocking test node", "tests", [], [new("INT")]);
        public async ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> inputs, CancellationToken cancellationToken)
        {
            if (Interlocked.Increment(ref calls) == 1)
            {
                FirstStarted.TrySetResult();
                await Task.Delay(Timeout.Infinite, cancellationToken);
            }
            else
            {
                SecondStarted.TrySetResult();
                await FinishSecond.Task.WaitAsync(cancellationToken);
            }
            return [JsonValue.Create(1)];
        }
    }
}

public sealed class StorageTests : IDisposable
{
    private readonly string directory = Path.Combine(Path.GetTempPath(), "ComfySharpTests-" + Guid.NewGuid());

    [Fact]
    public void SettingsRoundtripAndPruneNeverDeletesPhysicalFiles()
    {
        var store = new LocalStore(directory);
        store.SetSettings(new JsonObject { ["theme"] = "dark", ["extension"] = new JsonObject { ["unknown"] = 12 } });
        Assert.Equal(12, new LocalStore(directory).Settings()["extension"]!["unknown"]!.GetValue<int>());
        File.WriteAllText(Path.Combine(directory, "keep.txt"), "keep");
        File.WriteAllText(Path.Combine(directory, "missing.txt"), "missing");
        store.RegisterAsset("keep.txt", "text/plain");
        store.RegisterAsset("missing.txt", "text/plain");
        File.Delete(Path.Combine(directory, "missing.txt"));
        Assert.Equal(1, store.PruneMissing());
        Assert.Equal(0, store.PruneMissing());
        Assert.Equal("keep", File.ReadAllText(Path.Combine(directory, "keep.txt")));
        Assert.Equal(2, store.Assets().Count);
        Assert.Single(store.Assets(), a => a!["missing"]!.GetValue<bool>());
    }

    [Theory]
    [InlineData("../outside.txt")]
    [InlineData("x/../../outside.txt")]
    [InlineData("C:/outside.txt")]
    [InlineData("/outside.txt")]
    public void DataPathCannotEscapeOwnDirectory(string path)
    {
        Assert.Throws<ArgumentException>(() => new LocalStore(directory).ResolvePath(path));
    }

    [Fact]
    public void SchemaMigrationCanBeReversedAndReapplied()
    {
        var store = new LocalStore(directory);
        store.Migrate(0);
        store.Migrate(1);
        store.SetSettings(new JsonObject { ["version"] = 1 });
        Assert.Equal(1, store.Settings()["version"]!.GetValue<int>());
    }

    public void Dispose() { if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true); }
}
