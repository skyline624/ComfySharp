using System.Net;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Host;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;

namespace ComfySharp.Host.Tests;

public sealed class HostUiTests
{
    [Fact]
    public async Task Anonymous_interruption_is_broadcast_without_other_execution_details_and_after_history_commit()
    {
        var interruptedNode = new InterruptibleOutputNode();
        var registry = new NodeRegistry(); registry.Register(interruptedNode);
        await using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            builder.ConfigureServices(services => services.AddSingleton(new EngineService(registry))));
        using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var hub = factory.Services.GetRequiredService<EventHub>();
        var first = hub.Subscribe("first-observer");
        var second = hub.Subscribe("second-observer");
        try
        {
            using var submitted = await client.PostAsJsonAsync("/prompt", JsonNode.Parse("""
                {"prompt":{"wait":{"class_type":"TestInterruptibleOutput","inputs":{}}}}
                """), timeout.Token);
            submitted.EnsureSuccessStatusCode();
            var id = (await submitted.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
            await interruptedNode.Entered.Task.WaitAsync(timeout.Token);
            using var cancelled = await client.PostAsJsonAsync("/interrupt", new JsonObject { ["prompt_id"] = id }, timeout.Token);
            cancelled.EnsureSuccessStatusCode();
            Assert.True((await cancelled.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["cancelled"]!.GetValue<bool>());

            foreach (var subscriber in new[] { first, second })
            {
                var messages = new List<JsonObject>();
                while (true)
                {
                    var message = JsonNode.Parse(await subscriber.Reader.ReadAsync(timeout.Token))!.AsObject();
                    messages.Add(message);
                    if (message["type"]!.GetValue<string>() != "execution_interrupted") continue;
                    Assert.Equal(id, message["data"]!["prompt_id"]!.GetValue<string>());
                    // This single request observes the state as soon as the interruption terminal arrives.
                    var history = (await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token))![id]!;
                    Assert.Equal("cancelled", history["status"]!["status_str"]!.GetValue<string>());
                    Assert.False(history["status"]!["completed"]!.GetValue<bool>());
                    Assert.Empty(history["outputs"]!.AsObject());
                    break;
                }
                messages.AddRange(await UntilIdle(subscriber.Reader, timeout.Token));
                Assert.Single(messages, message => message["type"]!.GetValue<string>() == "execution_interrupted");
                Assert.All(messages, message => Assert.Contains(message["type"]!.GetValue<string>(), new[] { "status", "execution_interrupted" }));
            }
        }
        finally { hub.Remove(first.Id); hub.Remove(second.Id); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Anonymous_execution_does_not_broadcast_success_or_failure(bool fail)
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var subscriber = factory.Services.GetRequiredService<EventHub>().Subscribe("observer");
        var prompt = fail ? JsonNode.Parse("""
            {"1":{"class_type":"KarrasScheduler","inputs":{"steps":3,"sigma_max":3.0,"sigma_min":1.0,"rho":0.0}},
             "2":{"class_type":"PreviewAny","inputs":{"source":["1",0]}}}
            """)!.AsObject() : TextPrompt();
        using var response = await client.PostAsJsonAsync("/prompt", new JsonObject { ["prompt"] = prompt }, timeout.Token);
        response.EnsureSuccessStatusCode();
        var id = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
        var messages = await UntilIdle(subscriber.Reader, timeout.Token);
        Assert.All(messages, m => Assert.Equal("status", m["type"]!.GetValue<string>()));
        var history = (await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token))![id]!;
        Assert.Equal(!fail, history["status"]!["completed"]!.GetValue<bool>());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Extra_data_client_routes_execution_and_root_client_overrides_it(bool rootOverride)
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var hub = factory.Services.GetRequiredService<EventHub>();
        var nested = hub.Subscribe("nested"); var root = hub.Subscribe("root");
        var body = new JsonObject { ["prompt"] = TextPrompt(), ["extra_data"] = new JsonObject { ["client_id"] = "nested", ["keep"] = "value" } };
        if (rootOverride) body["client_id"] = "root";
        using var response = await client.PostAsJsonAsync("/prompt", body, timeout.Token);
        response.EnsureSuccessStatusCode();
        var id = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
        var intended = await UntilIdle((rootOverride ? root : nested).Reader, timeout.Token);
        var other = await UntilIdle((rootOverride ? nested : root).Reader, timeout.Token);
        Assert.Contains(intended, m => m["type"]!.GetValue<string>() == "executed");
        Assert.All(other, m => Assert.Equal("status", m["type"]!.GetValue<string>()));
        var history = await CompletedHistory(client, id, timeout.Token);
        Assert.Equal(rootOverride ? "root" : "nested", history["prompt"]![3]!["client_id"]!.GetValue<string>());
        Assert.Equal("value", history["prompt"]![3]!["keep"]!.GetValue<string>());
    }

    [Fact]
    public async Task Reconnecting_sid_replaces_previous_subscription_and_old_cleanup_keeps_new_one()
    {
        var hub = new EventHub(); var previous = hub.Subscribe("same"); var current = hub.Subscribe("same");
        await previous.Reader.Completion.WaitAsync(TimeSpan.FromSeconds(2));
        hub.Remove(previous.Id);
        hub.Publish("executed", new JsonObject { ["value"] = 7 }, "same");
        Assert.False(previous.Reader.TryRead(out _));
        Assert.True(current.Reader.TryRead(out var message));
        Assert.Equal(7, JsonNode.Parse(message)!["data"]!["value"]!.GetValue<int>());
        hub.Remove(current.Id);
    }

    [Fact]
    public async Task Empty_sid_gets_an_independent_generated_session()
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var socket = await factory.Server.CreateWebSocketClient().ConnectAsync(new Uri("ws://localhost/ws?clientId="), timeout.Token);
        var message = await ReadMessage(socket, timeout.Token);
        Assert.True(Guid.TryParseExact(message["data"]!["sid"]!.GetValue<string>(), "D", out _));
        await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", timeout.Token);
    }

    private static async Task<List<JsonObject>> UntilIdle(System.Threading.Channels.ChannelReader<string> reader, CancellationToken token)
    {
        var messages = new List<JsonObject>();
        while (true)
        {
            var message = JsonNode.Parse(await reader.ReadAsync(token))!.AsObject(); messages.Add(message);
            if (message["type"]!.GetValue<string>() == "status" && message["data"]!["status"]!["exec_info"]!["queue_remaining"]!.GetValue<int>() == 0) return messages;
        }
    }

    private static JsonObject TextPrompt() => JsonNode.Parse("""
        {"1":{"class_type":"PrimitiveString","inputs":{"value":"first"}},
         "2":{"class_type":"PreviewAny","inputs":{"source":["1",0]}},
         "3":{"class_type":"PreviewAny","inputs":{"source":"second"}}}
        """)!.AsObject();

    [Theory]
    [InlineData("/prompt")]
    [InlineData("/api/prompt")]
    public async Task A_prompt_without_an_output_node_is_rejected_before_enqueue(string endpoint)
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var response = await client.PostAsJsonAsync(endpoint, JsonNode.Parse("""
            {"prompt":{"1":{"class_type":"PrimitiveString","inputs":{"value":"internal value"}}}}
            """));
        await AssertRejected(response, "prompt_no_outputs");
        Assert.Empty(factory.Services.GetRequiredService<JobQueue>().Jobs());
    }

    [Theory]
    [InlineData("[\"1\"]")]
    [InlineData("[]")]
    [InlineData("[\"missing\"]")]
    public async Task Partial_targets_cannot_turn_internal_or_absent_nodes_into_outputs(string targets)
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var response = await client.PostAsJsonAsync("/prompt", new JsonObject
        {
            ["prompt"] = TextPrompt(), ["partial_execution_targets"] = JsonNode.Parse(targets)
        });
        await AssertRejected(response, "prompt_no_outputs");
        Assert.Empty(factory.Services.GetRequiredService<JobQueue>().Jobs());
    }

    [Fact]
    public async Task Mixed_partial_targets_select_only_requested_output_nodes_and_preserve_graph_inputs()
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var response = await client.PostAsJsonAsync("/api/prompt", new JsonObject
        {
            ["prompt"] = TextPrompt(), ["partial_execution_targets"] = new JsonArray("1", "2", "2", "missing")
        }, timeout.Token);
        response.EnsureSuccessStatusCode();
        var id = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
        var history = await CompletedHistory(client, id, timeout.Token);
        Assert.Equal("2", Assert.Single(history["outputs"]!.AsObject()).Key);
        Assert.Equal("first", history["outputs"]!["2"]!["text"]![0]!.GetValue<string>());
        Assert.Equal("2", Assert.Single(history["prompt"]![4]!.AsArray())!.GetValue<string>());
        Assert.Equal("1", history["prompt"]![2]!["2"]!["inputs"]!["source"]![0]!.GetValue<string>());
    }

    [Theory]
    [InlineData("{\"inputs\":{},\"_meta\":{\"title\":\"Lost node\"}}", null)]
    [InlineData("{\"class_type\":\"MissingPythonNode\",\"inputs\":{},\"_meta\":{\"title\":\"Lost node\"}}", "MissingPythonNode")]
    [InlineData("{\"class_type\":12,\"inputs\":{},\"_meta\":{\"title\":\"Lost node\"}}", null)]
    public async Task Missing_node_types_are_rejected_even_outside_selected_dependencies(string unknown, string? type)
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        var prompt = TextPrompt(); prompt["unreachable"] = JsonNode.Parse(unknown);
        using var response = await client.PostAsJsonAsync("/prompt", new JsonObject
        {
            ["prompt"] = prompt, ["partial_execution_targets"] = new JsonArray("2")
        });
        var error = await AssertRejected(response, "missing_node_type");
        Assert.Equal("unreachable", error["extra_info"]!["node_id"]!.GetValue<string>());
        Assert.Equal(type, error["extra_info"]!["class_type"]?.GetValue<string>());
        Assert.Equal("Lost node", error["extra_info"]!["node_title"]!.GetValue<string>());
        Assert.Empty(factory.Services.GetRequiredService<JobQueue>().Jobs());
    }

    [Fact]
    public async Task Invalid_independent_output_reports_its_errors_while_valid_output_executes()
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var prompt = TextPrompt(); prompt["3"]!["inputs"] = new JsonObject();
        using var response = await client.PostAsJsonAsync("/prompt", new JsonObject { ["prompt"] = prompt }, timeout.Token);
        response.EnsureSuccessStatusCode();
        var accepted = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!;
        Assert.Equal("3", Assert.Single(accepted["node_errors"]!.AsObject()).Key);
        Assert.Equal("required_input_missing", accepted["node_errors"]!["3"]!["errors"]![0]!["type"]!.GetValue<string>());
        Assert.Equal("source", accepted["node_errors"]!["3"]!["errors"]![0]!["extra_info"]!["input_name"]!.GetValue<string>());
        Assert.Equal("3", accepted["node_errors"]!["3"]!["dependent_outputs"]![0]!.GetValue<string>());
        var history = await CompletedHistory(client, accepted["prompt_id"]!.GetValue<string>(), timeout.Token);
        Assert.Equal("2", Assert.Single(history["outputs"]!.AsObject()).Key);
    }

    [Fact]
    public async Task Real_sigma_graph_sends_only_ui_to_its_session_and_commits_history_before_terminal()
    {
        await using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var wsClient = factory.Server.CreateWebSocketClient();
        using var owner = await wsClient.ConnectAsync(new Uri("ws://localhost/ws?clientId=sigma-owner"), timeout.Token);
        using var observer = await wsClient.ConnectAsync(new Uri("ws://localhost/ws?clientId=other-client"), timeout.Token);
        Assert.Equal("sigma-owner", (await ReadMessage(owner, timeout.Token))["data"]!["sid"]!.GetValue<string>());
        Assert.Equal("other-client", (await ReadMessage(observer, timeout.Token))["data"]!["sid"]!.GetValue<string>());

        using var response = await client.PostAsJsonAsync("/prompt", JsonNode.Parse("""
            {"client_id":"sigma-owner","prompt":{
                "1":{"class_type":"KarrasScheduler","inputs":{"steps":3,"sigma_max":3.0,"sigma_min":1.0,"rho":1.0}},
                "2":{"class_type":"SplitSigmas","inputs":{"sigmas":["1",0],"step":1}},
                "3":{"class_type":"PreviewAny","inputs":{"source":["2",0]}},
                "4":{"class_type":"PreviewAny","inputs":{"source":["2",1]}}}}
            """), timeout.Token);
        response.EnsureSuccessStatusCode();
        var id = (await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
        var executed = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
        while (true)
        {
            var message = await ReadMessage(owner, timeout.Token);
            var kind = message["type"]!.GetValue<string>();
            if (kind == "status") continue;
            Assert.Equal(id, message["data"]!["prompt_id"]!.GetValue<string>());
            Assert.DoesNotContain(kind, new[] { "execution_error", "execution_failed", "execution_interrupted" });
            if (kind == "executed")
            {
                var data = message["data"]!.AsObject();
                var node = data["node"]!.GetValue<string>();
                Assert.Equal(node, data["display_node"]!.GetValue<string>());
                Assert.Equal("text", Assert.Single(data["output"]!.AsObject()).Key);
                Assert.False(data.ContainsKey("result"));
                executed.Add(node, data["output"]!.DeepClone().AsObject());
            }
            if (kind == "execution_success") break;
        }
        Assert.Equal(new[] { "3", "4" }, executed.Keys.Order(StringComparer.Ordinal));
        Assert.Equal("tensor([3., 2.])", executed["3"]["text"]![0]!.GetValue<string>());
        Assert.Equal("tensor([2., 1., 0.])", executed["4"]["text"]![0]!.GetValue<string>());

        // This is deliberately a single immediate request after the terminal event, not eventual polling.
        var historyEnvelope = (await client.GetFromJsonAsync<JsonObject>($"/history/{id}", timeout.Token))!;
        var history = Assert.IsType<JsonObject>(historyEnvelope[id]);
        Assert.True(history["status"]!["completed"]!.GetValue<bool>());
        Assert.Equal("success", history["status"]!["status_str"]!.GetValue<string>());
        Assert.Equal(new[] { "3", "4" }, history["outputs"]!.AsObject().Select(p => p.Key).Order(StringComparer.Ordinal));
        Assert.Equal(new[] { "3", "4" }, history["meta"]!.AsObject().Select(p => p.Key).Order(StringComparer.Ordinal));
        foreach (var node in executed.Keys)
        {
            Assert.True(JsonNode.DeepEquals(executed[node], history["outputs"]![node]));
            Assert.Equal(node, history["meta"]![node]!["node_id"]!.GetValue<string>());
            Assert.Equal(node, history["meta"]![node]!["real_node_id"]!.GetValue<string>());
            Assert.Null(history["meta"]![node]!["parent_node"]);
        }
        // Queue status is broadcast after every execution event, establishing a deterministic observation boundary.
        while (true)
        {
            var message = await ReadMessage(observer, timeout.Token);
            Assert.Equal("status", message["type"]!.GetValue<string>());
            Assert.False(message["data"]!.AsObject().ContainsKey("prompt_id"));
            if (message["data"]!["status"]!["exec_info"]!["queue_remaining"]!.GetValue<int>() == 0) break;
        }
        await owner.CloseAsync(WebSocketCloseStatus.NormalClosure, "", timeout.Token);
        await observer.CloseAsync(WebSocketCloseStatus.NormalClosure, "", timeout.Token);
    }

    [Fact]
    public void Event_hub_routes_session_events_and_freezes_payloads_before_broadcast()
    {
        var hub = new EventHub();
        var owner = hub.Subscribe("owner");
        var other = hub.Subscribe("other");
        var anonymous = hub.Subscribe();
        var payload = new JsonObject { ["output"] = new JsonObject { ["text"] = new JsonArray("before") } };
        hub.Publish("executed", payload, "owner");
        payload["output"]!["text"]![0] = "after";
        hub.Publish("status", new JsonObject { ["queue_remaining"] = 0 });
        Assert.True(owner.Reader.TryRead(out var executed));
        Assert.Equal("before", JsonNode.Parse(executed)!["data"]!["output"]!["text"]![0]!.GetValue<string>());
        foreach (var subscriber in new[] { owner, other, anonymous })
        {
            Assert.True(subscriber.Reader.TryRead(out var status));
            Assert.Equal("status", JsonNode.Parse(status)!["type"]!.GetValue<string>());
            Assert.False(subscriber.Reader.TryRead(out _));
            hub.Remove(subscriber.Id);
        }
    }

    private static async Task<JsonObject> AssertRejected(HttpResponseMessage response, string expectedType)
    {
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var document = (await response.Content.ReadFromJsonAsync<JsonObject>())!;
        Assert.Equal(expectedType, document["error"]!["type"]!.GetValue<string>());
        Assert.IsType<JsonObject>(document["node_errors"]);
        return document["error"]!.AsObject();
    }

    private static async Task<JsonObject> CompletedHistory(HttpClient client, string id, CancellationToken token)
    {
        while (true)
        {
            var document = (await client.GetFromJsonAsync<JsonObject>($"/history/{id}", token))!;
            if (document[id] is JsonObject history)
            {
                Assert.True(history["status"]!["completed"]!.GetValue<bool>(), history.ToJsonString());
                return history;
            }
            await Task.Delay(10, token);
        }
    }

    private static async Task<JsonObject> ReadMessage(WebSocket socket, CancellationToken token)
    {
        var buffer = new byte[4096];
        using var stream = new MemoryStream();
        while (true)
        {
            var received = await socket.ReceiveAsync(buffer, token);
            Assert.Equal(WebSocketMessageType.Text, received.MessageType);
            stream.Write(buffer, 0, received.Count);
            Assert.True(stream.Length <= 65536, "Unexpectedly large UI protocol message.");
            if (received.EndOfMessage) return JsonNode.Parse(stream.ToArray())!.AsObject();
        }
    }

    // Test-only asynchronous terminal: no synthetic node is registered in the product catalogue.
    private sealed class InterruptibleOutputNode : INode
    {
        public TaskCompletionSource Entered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public NodeSchema Schema { get; } = new("TestInterruptibleOutput", "Interruptible test output", "tests", [], [], OutputNode: true);
        public async ValueTask<IReadOnlyList<JsonNode?>> ExecuteAsync(IReadOnlyDictionary<string, JsonNode?> inputs,
            CancellationToken cancellationToken)
        {
            Entered.TrySetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            throw new InvalidOperationException("The blocking test node should only exit through cancellation.");
        }
    }
}
