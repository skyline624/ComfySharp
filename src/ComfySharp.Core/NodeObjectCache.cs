using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Core;

// Static graph object keys match CacheKeySetID: (node_id, class_type), independent of input values.
// Concurrent executions retain objects until all active prompts finish; pruning then uses the latest prompt.
// This is not a result cache, LRU policy or dynamic-subgraph cache.
internal sealed class NodeObjectCache : IDisposable
{
    private readonly object gate = new();
    private readonly Dictionary<(string Id, string Type), IRuntimeNode> instances = [];
    private HashSet<(string Id, string Type)> latest = [];
    private int active;
    private bool closed;

    public void CheckAvailable() { lock (gate) ObjectDisposedException.ThrowIf(closed, this); }

    public PromptScope BeginPrompt(JsonObject prompt)
    {
        IRuntimeNode[] retired;
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(closed, this);
            latest = prompt.Where(p => p.Value is JsonObject n && n["class_type"] is JsonValue v && v.TryGetValue<string>(out _))
                .Select(p => (p.Key, p.Value!["class_type"]!.GetValue<string>())).ToHashSet();
            retired = active == 0 ? RemoveUnused() : [];
            active++;
        }
        try { Release(retired); return new(this); }
        catch { EndPrompt(); throw; }
    }

    private IRuntimeNode Get(string id, IRuntimeNode definition)
    {
        if (definition is not IRuntimeNodeFactory factory) return definition;
        lock (gate)
        {
            var key = (id, definition.Schema.ClassType);
            if (instances.TryGetValue(key, out var previous)) return previous;
            IRuntimeNode instance = factory.CreateInstance() ?? throw new InvalidOperationException("Node factory returned null.");
            if (ReferenceEquals(instance, definition) || instances.Values.Any(n => ReferenceEquals(n, instance)))
                throw new InvalidOperationException("Node factory must return a fresh, independently owned instance.");
            if (instance.Schema.ClassType != definition.Schema.ClassType)
            {
                (instance as IDisposable)?.Dispose();
                throw new InvalidOperationException("Node factory returned a different class_type.");
            }
            instances.Add(key, instance);
            return instance;
        }
    }

    private IRuntimeNode[] RemoveUnused()
    {
        var keys = instances.Keys.Where(k => closed || !latest.Contains(k)).ToArray();
        var retired = keys.Select(k => instances[k]).ToArray();
        foreach (var key in keys) instances.Remove(key);
        return retired;
    }

    private void EndPrompt()
    {
        IRuntimeNode[] retired;
        lock (gate)
        {
            active--;
            retired = active == 0 ? RemoveUnused() : [];
        }
        Release(retired);
    }

    public void Dispose()
    {
        IRuntimeNode[] retired;
        lock (gate)
        {
            if (closed) return;
            closed = true;
            retired = active == 0 ? RemoveUnused() : [];
        }
        Release(retired);
    }

    private static void Release(IEnumerable<IRuntimeNode> nodes)
    {
        List<Exception>? errors = null;
        foreach (var node in nodes)
        {
            try { (node as IDisposable)?.Dispose(); }
            catch (Exception ex) { (errors ??= []).Add(ex); }
        }
        if (errors is not null) throw new AggregateException("Node instance disposal failed.", errors);
    }

    public sealed class PromptScope(NodeObjectCache cache) : IDisposable
    {
        private bool disposed;
        public IRuntimeNode Get(string id, IRuntimeNode definition)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return cache.Get(id, definition);
        }
        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            cache.EndPrompt();
        }
    }
}
