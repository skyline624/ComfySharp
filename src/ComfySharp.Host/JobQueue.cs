using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;

namespace ComfySharp.Host;

// Queue state belongs to the host, independently of graph execution. Jobs are intentionally in memory.
public sealed class JobQueue(EngineService engine, EventHub events) : BackgroundService
{
    private readonly object gate = new();
    private readonly Dictionary<string, Job> jobs = new(StringComparer.Ordinal);
    private readonly SemaphoreSlim available = new(0);
    private long sequence;
    private Job? active;

    public sealed class Job(string id, double priority, long order, JsonObject prompt,
        string[] targets, string? clientId, JsonObject extraData)
    {
        public string Id { get; } = id;
        public double Priority { get; } = priority;
        public long Order { get; } = order;
        public JsonObject Prompt { get; } = prompt;
        public string[] Targets { get; } = targets;
        public string? ClientId { get; } = clientId;
        public JsonObject ExtraData { get; } = extraData;
        public string Status { get; set; } = "pending";
        public JsonNode? Outputs { get; set; }
        public JsonNode? Meta { get; set; }
        public JsonNode? Diagnostics { get; set; }
        public CancellationTokenSource Cancellation { get; } = new();
    }

    public (string Id, double Number) Enqueue(JsonObject prompt, string[] targets,
        string? clientId, double? number, bool front, string? requestedId, JsonObject? extraData)
    {
        lock (gate)
        {
            var id = requestedId ?? Guid.NewGuid().ToString();
            if (jobs.ContainsKey(id)) throw new ArgumentException("prompt_id already exists.");
            var order = sequence++;
            var priority = number ?? order;
            if (front) priority = -priority;
            var retainedExtra = (JsonObject?)extraData?.DeepClone() ?? [];
            if (clientId is not null) retainedExtra["client_id"] = clientId;
            jobs.Add(id, new Job(id, priority, order, (JsonObject)prompt.DeepClone(), targets,
                clientId, retainedExtra));
            available.Release();
            PublishStatus();
            return (id, priority);
        }
    }

    public bool Cancel(string id)
    {
        lock (gate)
        {
            if (!jobs.TryGetValue(id, out var job)) return false;
            if (job.Status == "pending")
            {
                job.Status = "cancelled";
                PublishStatus();
                return true;
            }
            if (job != active || job.Status != "in_progress") return false;
            // Capture and cancel this job's token while identity is protected. Never use a global interrupt bit.
            job.Cancellation.Cancel();
            return true;
        }
    }

    public bool CancelActive()
    {
        lock (gate) return active is not null && Cancel(active.Id);
    }

    public void ClearPending(IEnumerable<string>? ids = null)
    {
        lock (gate)
        {
            var selected = ids?.ToHashSet(StringComparer.Ordinal);
            foreach (var job in jobs.Values.Where(j => j.Status == "pending" && (selected is null || selected.Contains(j.Id))))
                job.Status = "cancelled";
            PublishStatus();
        }
    }

    public void ClearHistory(IEnumerable<string>? ids = null)
    {
        lock (gate)
        {
            var selected = ids?.ToHashSet(StringComparer.Ordinal);
            foreach (var job in jobs.Values.Where(j => IsTerminal(j.Status) && (selected is null || selected.Contains(j.Id))).ToArray())
            {
                jobs.Remove(job.Id);
                job.Cancellation.Dispose();
            }
        }
    }

    public JsonObject QueueSnapshot()
    {
        lock (gate) return new JsonObject
        {
            ["queue_running"] = new JsonArray(jobs.Values.Where(j => j.Status == "in_progress").Select(QueueEntry).ToArray()),
            ["queue_pending"] = new JsonArray(jobs.Values.Where(j => j.Status == "pending")
                .OrderBy(j => j.Priority).ThenBy(j => j.Order).Select(QueueEntry).ToArray())
        };
    }

    public JsonObject StatusSnapshot()
    {
        lock (gate) return new JsonObject { ["exec_info"] = new JsonObject
        { ["queue_remaining"] = jobs.Values.Count(j => j.Status is "pending" or "in_progress") } };
    }

    public JsonObject History(string? id = null, int? maxItems = null)
    {
        lock (gate)
        {
            var result = new JsonObject();
            foreach (var job in jobs.Values.Where(j => IsTerminal(j.Status) && (id is null || j.Id == id))
                         .OrderByDescending(j => j.Order).Take(maxItems ?? int.MaxValue))
                result[job.Id] = new JsonObject
                {
                    ["prompt"] = QueueEntry(job), ["outputs"] = job.Outputs?.DeepClone() ?? new JsonObject(),
                    ["meta"] = job.Meta?.DeepClone() ?? new JsonObject(),
                    ["status"] = new JsonObject { ["status_str"] = job.Status == "completed" ? "success" : job.Status,
                        ["completed"] = job.Status == "completed", ["messages"] = job.Diagnostics?.DeepClone() ?? new JsonArray() }
                };
            return result;
        }
    }

    public JsonNode? JobSnapshot(string id)
    {
        lock (gate) return jobs.TryGetValue(id, out var job) ? Describe(job) : null;
    }

    public JsonArray Jobs()
    {
        lock (gate) return new JsonArray(jobs.Values.OrderBy(j => j.Order).Select(Describe).ToArray());
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                await available.WaitAsync(stoppingToken);
                Job? job;
                lock (gate)
                {
                    job = jobs.Values.Where(j => j.Status == "pending").OrderBy(j => j.Priority).ThenBy(j => j.Order).FirstOrDefault();
                    if (job is null) continue;
                    active = job;
                    job.Status = "in_progress";
                }
                using var linked = CancellationTokenSource.CreateLinkedTokenSource(stoppingToken, job.Cancellation.Token);
                try
                {
                    var terminals = new List<EngineEvent>();
                    var result = await engine.ExecuteUiAsync(job.Prompt, job.Targets,
                        e =>
                        {
                            if (e.Type is "execution_success" or "execution_failed" or "execution_interrupted") terminals.Add(e);
                            else PublishExecution(job, e);
                            return ValueTask.CompletedTask;
                        }, linked.Token, job.ExtraData);
                    lock (gate)
                    {
                        job.Status = result.Status switch { "success" => "completed", "cancelled" => "cancelled", _ => "failed" };
                        job.Outputs = System.Text.Json.JsonSerializer.SerializeToNode(result.Outputs);
                        job.Meta = new JsonObject();
                        foreach (var (nodeId, identity) in result.Meta)
                            job.Meta[nodeId] = new JsonObject { ["node_id"] = identity.NodeId, ["display_node"] = identity.DisplayNodeId,
                                ["parent_node"] = identity.ParentNodeId, ["real_node_id"] = identity.NodeId };
                        job.Diagnostics = System.Text.Json.JsonSerializer.SerializeToNode(result.Diagnostics);
                    }
                    // Clients observing terminal success can now read the completed managed history.
                    foreach (var terminal in terminals) PublishExecution(job, terminal);
                }
                catch (OperationCanceledException) when (linked.IsCancellationRequested)
                {
                    lock (gate) job.Status = "cancelled";
                }
                catch (Exception exception)
                {
                    lock (gate)
                    {
                        job.Status = "failed";
                        job.Diagnostics = new JsonArray(new JsonObject { ["type"] = exception.GetType().Name, ["message"] = exception.Message });
                    }
                    if (job.ClientId is not null)
                        events.Publish("execution_error", new JsonObject { ["prompt_id"] = job.Id, ["exception_message"] = exception.Message }, job.ClientId);
                }
                finally
                {
                    lock (gate) active = null;
                    if (job.ClientId is not null)
                        events.Publish("executing", new JsonObject { ["prompt_id"] = job.Id, ["node"] = null }, job.ClientId);
                    PublishStatus();
                }
            }
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested) { }
    }

    private static bool IsTerminal(string status) => status is "completed" or "failed" or "cancelled";
    private void PublishExecution(Job job, EngineEvent execution)
    {
        // PromptExecutor.add_message permits anonymous interruption broadcasts; other
        // execution data is delivered only to the submitting session.
        if (job.ClientId is null && execution.Type != "execution_interrupted") return;
        var payload = new JsonObject { ["prompt_id"] = job.Id };
        if (execution.NodeId is not null)
        {
            payload["node"] = execution.NodeId;
            payload["display_node"] = execution.Identity?.DisplayNodeId ?? execution.NodeId;
        }
        if (execution.Type == "executed") payload["output"] = execution.Output?.DeepClone();
        if (execution.Message is not null) payload["message"] = execution.Message;
        events.Publish(execution.Type, payload, job.ClientId);
    }
    private void PublishStatus() => events.Publish("status", new JsonObject { ["status"] = StatusSnapshot() });
    private static JsonNode Describe(Job job) => new JsonObject { ["id"] = job.Id, ["prompt_id"] = job.Id,
        ["status"] = job.Status, ["number"] = job.Priority };
    private static JsonNode QueueEntry(Job job) => new JsonArray(JsonValue.Create(job.Priority), JsonValue.Create(job.Id),
        job.Prompt.DeepClone(), job.ExtraData.DeepClone(), new JsonArray(job.Targets.Select(s => (JsonNode?)JsonValue.Create(s)).ToArray()));
}
