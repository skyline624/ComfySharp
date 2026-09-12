using System.Diagnostics;
using System.Net;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace ComfySharp.Desktop;

/// <summary>Owns only a local child process; never owns or replays document jobs.</summary>
public sealed class HostSupervisor : IDisposable
{
    private Process? process;
    private readonly SemaphoreSlim gate = new(1, 1);
    private readonly HttpClient client = new() { Timeout = TimeSpan.FromSeconds(5) };
    private bool disposed;
    public string Status { get; private set; } = "Host stopped";
    public Uri? Address { get; private set; }
    public bool Ready { get; private set; }
    public event EventHandler? Changed;
    private void SetStatus(string text) { Status = text; Changed?.Invoke(this, EventArgs.Empty); }
    public static string? DiscoverHost(string? baseDirectory = null, string? configuredHostPath = null)
    {
        var configured = configuredHostPath ?? Environment.GetEnvironmentVariable("COMFYSHARP_HOST_PATH");
        if (!string.IsNullOrWhiteSpace(configured)) return Path.GetFullPath(configured);
        var directory = new DirectoryInfo(baseDirectory ?? AppContext.BaseDirectory);
        var names = OperatingSystem.IsWindows()
            ? new[] { "ComfySharp.Host.exe", "ComfySharp.Host.dll" }
            : new[] { "ComfySharp.Host", "ComfySharp.Host.dll" };
        // Published processes have separate dependency closures (including different
        // SkiaSharp versions). Always prefer host/ over a legacy flat layout.
        foreach (var folder in new[] { Path.Combine(directory.FullName, "host"), directory.FullName })
            foreach (var name in names)
            { var path = Path.Combine(folder, name); if (File.Exists(path)) return path; }
        var runtime = RuntimeInformation.OSArchitecture switch
        {
            Architecture.X64 when OperatingSystem.IsWindows() => "win-x64",
            Architecture.X64 when OperatingSystem.IsLinux() => "linux-x64",
            Architecture.Arm64 when OperatingSystem.IsMacOS() => "osx-arm64",
            _ => null
        };
        var configurations = new[] { "Debug", "Release" };
        for (var ancestor = directory; ancestor is not null; ancestor = ancestor.Parent)
            if (ancestor.Name is "Debug" or "Release")
            {
                configurations = ancestor.Name == "Release" ? ["Release", "Debug"] : ["Debug", "Release"];
                break;
            }
        for (var parent = directory; parent is not null; parent = parent.Parent)
            foreach (var buildRoot in runtime is null
                    ? new[] { Path.Combine(parent.FullName, "src", "ComfySharp.Host", "bin") }
                    : new[] {
                        Path.Combine(parent.FullName, "src", "ComfySharp.Host", "bin", "native", runtime, "cpu"),
                        Path.Combine(parent.FullName, "src", "ComfySharp.Host", "bin") })
                foreach (var configuration in configurations)
                    foreach (var name in names)
                    { var path = Path.Combine(buildRoot, configuration, "net10.0", name); if (File.Exists(path)) return path; }
        return null;
    }
    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        await gate.WaitAsync(cancellationToken);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            StopCore();
            var path = DiscoverHost() ?? throw new FileNotFoundException("Build ComfySharp.Host or set COMFYSHARP_HOST_PATH to its apphost or DLL.");
            using var listener = new TcpListener(IPAddress.Loopback, 0); listener.Start();
            var port = ((IPEndPoint)listener.LocalEndpoint).Port; listener.Stop();
            Address = new Uri($"http://127.0.0.1:{port}");
            var info = new ProcessStartInfo(path.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) ? "dotnet" : path)
            { UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = Path.GetDirectoryName(path)! };
            if (path.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)) info.ArgumentList.Add(path);
            info.ArgumentList.Add("--urls"); info.ArgumentList.Add(Address.AbsoluteUri);
            var child = new Process { StartInfo = info, EnableRaisingEvents = true };
            child.Exited += (_, _) => { if (ReferenceEquals(process, child)) { Ready = false; SetStatus("Host exited. Documents are retained; restart is available."); } };
            process = child;
            SetStatus("Starting Host…");
            child.Start();
            for (var attempt = 0; attempt < 100; attempt++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (child.HasExited) throw new InvalidOperationException($"Host exited with code {child.ExitCode}.");
                try
                {
                    var health = await client.GetFromJsonAsync<JsonObject>(new Uri(Address, "/health"), JsonSerializerOptions.Default, cancellationToken);
                    if (health?["status"]?.GetValue<string>() == "ok") { Ready = true; SetStatus($"Host ready at {Address}"); return; }
                }
                catch (HttpRequestException) { }
                catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested) { }
                await Task.Delay(100, cancellationToken);
            }
            throw new TimeoutException("The Host did not become healthy.");
        }
        catch (Exception error) { StopCore(); SetStatus(error.Message); throw; }
        finally { gate.Release(); }
    }
    public async Task<JsonObject> GetAsync(string route) => await client.GetFromJsonAsync<JsonObject>(Endpoint(route), JsonSerializerOptions.Default) ?? new JsonObject();
    public Task<JsonObject> SubmitAsync(JsonObject prompt, string clientId, IReadOnlyList<string>? targets = null)
    {
        var body = new JsonObject { ["prompt"] = prompt.DeepClone(), ["client_id"] = clientId };
        if (targets is not null) body["partial_execution_targets"] = new JsonArray(targets.Select(t => (JsonNode?)JsonValue.Create(t)).ToArray());
        return PostAsync("/prompt", body);
    }
    public Task<JsonObject> InterruptAsync(string promptId) => PostAsync("/interrupt", new JsonObject { ["prompt_id"] = promptId });
    private Uri Endpoint(string route) => Ready && Address is not null ? new Uri(Address, route) : throw new InvalidOperationException("Host is not ready.");
    private async Task<JsonObject> PostAsync(string route, JsonObject body)
    {
        using var response = await client.PostAsJsonAsync(Endpoint(route), body);
        var text = await response.Content.ReadAsStringAsync();
        if (!response.IsSuccessStatusCode) throw new InvalidOperationException($"Host {(int)response.StatusCode}: {text}");
        return JsonNode.Parse(text) as JsonObject ?? new JsonObject();
    }
    private void StopCore()
    {
        Ready = false; var old = process; process = null;
        if (old is null) return;
        try { if (!old.HasExited) { old.Kill(entireProcessTree: true); old.WaitForExit(5000); } } finally { old.Dispose(); }
    }
    public void Dispose() { if (disposed) return; disposed = true; StopCore(); client.Dispose(); }
}
