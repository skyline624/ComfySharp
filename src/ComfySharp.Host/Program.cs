using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Host;
using ComfySharp.Nodes.Tensor;
using ComfySharp.Storage;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);
// Prompt IDs, input names and literal dictionary keys are case-sensitive JSON data.
builder.Services.ConfigureHttpJsonOptions(options => options.SerializerOptions.PropertyNameCaseInsensitive = false);
if (builder.Configuration["urls"] is null) builder.WebHost.UseUrls("http://127.0.0.1:8189");
builder.Services.AddSingleton(sp =>
{
    var registry = TensorNodes.CreateRegistry();
    ImageFileNodes.Register(registry, sp.GetRequiredService<IImageFileStore>(),
        disableMetadata: builder.Configuration.GetValue<bool>("disable-metadata"));
    ImageInputNodes.Register(registry, sp.GetRequiredService<ImageInputService>());
    LoraNodes.Register(registry, builder.Configuration["models-dir"] ?? builder.Configuration["COMFYSHARP_MODELS_DIR"]);
    registry.Register(new SaveLoraNode(sp.GetRequiredService<ImageFileStore>()));
    Sd15Nodes.Register(registry, new CheckpointFiles(builder.Configuration["models-dir"] ?? builder.Configuration["COMFYSHARP_MODELS_DIR"]),
        cpuThreads: builder.Configuration.GetValue<int?>("cpu-threads") ?? Math.Min(16, Environment.ProcessorCount),
        inferenceDevice: builder.Configuration["inference-device"] ?? "cpu");
    return new EngineService(registry);
});
builder.Services.AddSingleton<EventHub>();
builder.Services.AddSingleton<JobQueue>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<JobQueue>());
builder.Services.AddSingleton(_ => new LocalStore(builder.Configuration["data-dir"] ??
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ComfySharp")));
builder.Services.AddSingleton(_ => new ImageFileStore(builder.Configuration["data-dir"] ??
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ComfySharp")));
builder.Services.AddSingleton<IImageFileStore>(sp => sp.GetRequiredService<ImageFileStore>());
builder.Services.AddSingleton<ImageInputService>();
var app = builder.Build();
app.Use(async (context, next) =>
{
    var origin = context.Request.Headers.Origin.ToString();
    var address = context.Connection.RemoteIpAddress;
    if ((address is not null && !IPAddress.IsLoopback(address)) ||
        (origin.Length > 0 && (!Uri.TryCreate(origin, UriKind.Absolute, out var uri) || !uri.IsLoopback ||
            !string.Equals(uri.Authority, context.Request.Host.Value, StringComparison.OrdinalIgnoreCase))))
    {
        context.Response.StatusCode = StatusCodes.Status403Forbidden;
        return;
    }
    try { await next(); }
    catch (JsonException exception) { await BadRequest(context, exception.Message); }
    catch (ArgumentException exception) { await BadRequest(context, exception.Message); }
    catch (InvalidDataException exception) { await BadRequest(context, exception.Message); }
});
app.UseWebSockets();
foreach (var prefix in new[] { "", "/api" })
{
    var routes = app.MapGroup(prefix);
    routes.MapGet("/health", () => new { status = "ok", product = "ComfySharp", version = "0.1.0-dev", inference_ready = false });
    routes.MapPost("/upload/image", async (HttpContext context, ImageInputService images) =>
    {
        if (!context.Request.HasFormContentType) return Results.BadRequest(new { error = "expected_multipart_image" });
        var form = await context.Request.ReadFormAsync(context.RequestAborted);
        var upload = form.Files.GetFile("image");
        if (upload is null) return Results.BadRequest(new { error = "missing_image" });
        if (form["type"].Count > 0 && form["type"].ToString() != "input") return Results.BadRequest(new { error = "only_input_uploads_supported" });
        await using var stream = upload.OpenReadStream();
        var file = await images.UploadAsync(upload.FileName, form["subfolder"].ToString(), stream, form["overwrite"].ToString() is "true" or "1", context.RequestAborted);
        return Results.Json(new { name = file.Filename, subfolder = file.Subfolder, type = file.Type });
    });
    routes.MapGet("/object_info", (EngineService engine) => engine.Registry.ToObjectInfo());
    routes.MapGet("/object_info/{nodeType}", (string nodeType, EngineService engine) =>
    {
        var info = engine.Registry.ToObjectInfo();
        return info[nodeType] is { } node ? Results.Json(new JsonObject { [nodeType] = node.DeepClone() }) : Results.NotFound();
    });
    routes.MapMethods("/view", ["GET", "HEAD"], (HttpContext context, ImageFileStore store) =>
    {
        var query = context.Request.Query;
        if (query.Keys.Any(key => key is not ("filename" or "subfolder" or "type")))
            return Results.BadRequest(new { error = "unsupported_view_options", message = "This route currently serves original input/output/temp PNG files. Image conversions and annotated asset paths remain to be ported." });
        string? filename = query["filename"].FirstOrDefault();
        if (string.IsNullOrEmpty(filename)) return Results.BadRequest(new { error = "missing_filename" });
        if (!filename.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            return Results.BadRequest(new { error = "unsupported_image_format" });
        var file = new ImageFileDescriptor(filename, query["subfolder"].FirstOrDefault() ?? "", query["type"].FirstOrDefault() ?? "output");
        try
        {
            var stream = store.OpenRead(file);
            context.Response.Headers.CacheControl = "no-store";
            return Results.File(stream, "image/png", enableRangeProcessing: true);
        }
        catch (FileNotFoundException) { return Results.NotFound(); }
        catch (DirectoryNotFoundException) { return Results.NotFound(); }
    });
    routes.MapPost("/prompt", (JsonObject request, EngineService engine, JobQueue queue) =>
    {
        if (request["prompt"] is not JsonObject prompt) return Results.BadRequest(new { error = new { type = "invalid_prompt", message = "prompt must be an object." }, node_errors = new { } });
        var targets = RequestFields.Strings(request, "partial_execution_targets");
        var id = RequestFields.Optional<string>(request, "prompt_id");
        var clientId = RequestFields.Optional<string>(request, "client_id");
        double? number = request.ContainsKey("number") ? RequestFields.Optional<double>(request, "number") : null;
        var front = RequestFields.Optional<bool>(request, "front");
        var extraData = RequestFields.Object(request, "extra_data");
        if (clientId is null && extraData is not null) clientId = RequestFields.Optional<string>(extraData, "client_id");
        var submission = PromptSubmission.Validate(engine, prompt, targets);
        var validation = submission.Validation;
        var nodeErrors = submission.NodeErrors;
        if (submission.Error is not null) return Results.BadRequest(new { error = submission.Error, node_errors = nodeErrors });
        if (id is not null && (!Guid.TryParseExact(id, "D", out var parsed) || parsed.ToString() != id))
            throw new InvalidDataException("prompt_id must be a canonical UUID.");
        if (number.HasValue && !double.IsFinite(number.Value)) throw new InvalidDataException("number must be finite.");
        var accepted = queue.Enqueue(prompt, validation.ValidTargets.ToArray(), clientId,
            number, front, id, extraData);
        return Results.Json(new { prompt_id = accepted.Id, number = accepted.Number, node_errors = nodeErrors });
    });
    routes.MapGet("/prompt", (JobQueue queue) => queue.StatusSnapshot());
    routes.MapGet("/queue", (JobQueue queue) => queue.QueueSnapshot());
    routes.MapPost("/queue", (JsonObject request, JobQueue queue) =>
    {
        var clear = RequestFields.Optional<bool>(request, "clear");
        var ids = RequestFields.Strings(request, "delete");
        if (clear) queue.ClearPending();
        if (ids is not null) queue.ClearPending(ids);
        return Results.Ok();
    });
    routes.MapGet("/history", (int? max_items, JobQueue queue) => queue.History(maxItems: Math.Clamp(max_items ?? 10000, 0, 10000)));
    routes.MapGet("/history/{id}", (string id, JobQueue queue) => queue.History(id));
    routes.MapPost("/history", (JsonObject request, JobQueue queue) =>
    {
        var clear = RequestFields.Optional<bool>(request, "clear");
        var ids = RequestFields.Strings(request, "delete");
        if (clear) queue.ClearHistory();
        if (ids is not null) queue.ClearHistory(ids);
        return Results.Ok();
    });
    routes.MapPost("/interrupt", async (HttpRequest request, JobQueue queue) =>
    {
        var canHaveBody = request.HttpContext.Features.Get<IHttpRequestBodyDetectionFeature>()?.CanHaveBody
            ?? (request.ContentLength is > 0 || request.Headers.ContainsKey("Transfer-Encoding"));
        if (!canHaveBody) return Results.Json(new { cancelled = queue.CancelActive() });
        var body = await JsonNode.ParseAsync(request.Body, cancellationToken: request.HttpContext.RequestAborted) as JsonObject
            ?? throw new InvalidDataException("Interrupt body must be an object.");
        var id = RequestFields.Optional<string>(body, "prompt_id");
        return Results.Json(new { cancelled = id is not null ? queue.Cancel(id) : queue.CancelActive() });
    });
    routes.MapGet("/settings", (LocalStore store) => store.Settings());
    routes.MapPost("/settings", (JsonObject settings, LocalStore store) => { store.SetSettings(settings); return Results.Ok(); });
    routes.MapGet("/features", () => new JsonObject());
}
app.MapGet("/api/jobs", (JobQueue queue) => new { jobs = queue.Jobs() });
app.MapGet("/api/jobs/{id}", (string id, JobQueue queue) => queue.JobSnapshot(id) is { } job ? Results.Json(job) : Results.NotFound());
app.MapPost("/api/jobs/{id}/cancel", (string id, JobQueue queue) => Results.Json(new { cancelled = queue.Cancel(id) }));
app.MapGet("/api/assets", (LocalStore store) => store.Assets());
app.MapPost("/api/assets/prune", (LocalStore store) => new { marked = store.PruneMissing() });
app.Map("/ws", async (HttpContext context, EventHub hub, JobQueue queue) =>
{
    if (!context.WebSockets.IsWebSocketRequest) { context.Response.StatusCode = 400; return; }
    using var socket = await context.WebSockets.AcceptWebSocketAsync();
    using var stop = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted);
    var sid = context.Request.Query["clientId"].FirstOrDefault();
    if (string.IsNullOrEmpty(sid)) sid = Guid.NewGuid().ToString();
    var subscription = hub.Subscribe(sid);
    try
    {
        await socket.SendAsync(Encoding.UTF8.GetBytes(new JsonObject { ["type"] = "status", ["data"] = new JsonObject
            { ["sid"] = sid, ["status"] = queue.StatusSnapshot() } }.ToJsonString()), WebSocketMessageType.Text, true, stop.Token);
        var receive = ReceiveUntilClosed(socket, stop.Token);
        var send = SendEvents(socket, subscription.Reader, stop.Token);
        await Task.WhenAny(receive, send);
        stop.Cancel();
        try { await Task.WhenAll(receive, send); } catch (OperationCanceledException) { }
        if (socket.State == WebSocketState.CloseReceived)
            await socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "", CancellationToken.None);
    }
    catch (WebSocketException) { }
    finally { hub.Remove(subscription.Id); }
});
app.Run();

static async Task SendEvents(WebSocket socket, System.Threading.Channels.ChannelReader<string> events, CancellationToken token)
{
    await foreach (var json in events.ReadAllAsync(token))
        await socket.SendAsync(Encoding.UTF8.GetBytes(json), WebSocketMessageType.Text, true, token);
}

static async Task ReceiveUntilClosed(WebSocket socket, CancellationToken token)
{
    var buffer = new byte[4096];
    var messageBytes = 0;
    while (!token.IsCancellationRequested)
    {
        var result = await socket.ReceiveAsync(buffer, token);
        if (result.MessageType == WebSocketMessageType.Close) return;
        messageBytes += result.Count;
        if (messageBytes > 65536) return;
        if (result.EndOfMessage) messageBytes = 0;
        // No optional feature is advertised until its negotiation and codec are implemented.
    }
}

static async Task BadRequest(HttpContext context, string message)
{
    if (context.Response.HasStarted) throw new InvalidOperationException(message);
    context.Response.StatusCode = StatusCodes.Status400BadRequest;
    await context.Response.WriteAsJsonAsync(new { error = new { type = "invalid_request", message } });
}

public partial class Program;
