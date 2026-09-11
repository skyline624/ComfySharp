using System.Collections.ObjectModel;
using System.Text.Json.Nodes;

namespace ComfySharp.Contracts;

public enum RuntimeValueKind { Json, Native, List, Map }

public sealed class RuntimeValueProjectionException : InvalidOperationException
{
    public RuntimeValueProjectionException() : base("Native execution values cannot be projected to JSON. Select a node that produces JSON output.") { }
}

/// <summary>A disposable lease. JSON snapshots are copied; native resources are shared until the final lease is disposed.</summary>
public sealed class RuntimeValue : IDisposable
{
    private readonly object gate = new();
    private readonly RuntimeValueKind kind;
    private readonly JsonNode? json;
    private readonly NativeOwner? native;
    private readonly IReadOnlyList<RuntimeValue>? items;
    private readonly IReadOnlyDictionary<string, RuntimeValue>? properties;
    private bool disposed;

    private RuntimeValue(JsonNode? value) { kind = RuntimeValueKind.Json; json = value?.DeepClone(); }
    private RuntimeValue(NativeOwner value) { kind = RuntimeValueKind.Native; native = value; }
    private RuntimeValue(IReadOnlyList<RuntimeValue> value) { kind = RuntimeValueKind.List; items = value; }
    private RuntimeValue(IReadOnlyDictionary<string, RuntimeValue> value) { kind = RuntimeValueKind.Map; properties = value; }

    public RuntimeValueKind Kind { get { lock (gate) { ThrowIfDisposed(); return kind; } } }
    /// <summary>Borrowed child leases, valid while this container is alive. Retain a child to keep it separately.</summary>
    public IReadOnlyList<RuntimeValue> Items { get { lock (gate) { ThrowIfDisposed(); return items ?? throw new InvalidOperationException("Value is not an execution list."); } } }
    public IReadOnlyDictionary<string, RuntimeValue> Properties { get { lock (gate) { ThrowIfDisposed(); return properties ?? throw new InvalidOperationException("Value is not a map."); } } }
    /// <summary>The resource is borrowed. Do not dispose it or transfer it to another owner; retain this lease instead.</summary>
    public T GetNative<T>() where T : class
    {
        lock (gate) { ThrowIfDisposed(); return native?.Resource as T ?? throw new InvalidOperationException($"Value is not a native {typeof(T).Name}."); }
    }

    public JsonNode? ToJson()
    {
        lock (gate)
        {
            ThrowIfDisposed();
            return kind switch
            {
                RuntimeValueKind.Json => json?.DeepClone(),
                RuntimeValueKind.List => new JsonArray(items!.Select(v => v.ToJson()).ToArray()),
                RuntimeValueKind.Map => new JsonObject(properties!.Select(p => new KeyValuePair<string, JsonNode?>(p.Key, p.Value.ToJson()))),
                _ => throw new RuntimeValueProjectionException()
            };
        }
    }

    public RuntimeValue Retain()
    {
        lock (gate)
        {
            ThrowIfDisposed();
            if (native is not null) { native.Retain(); return new(native); }
            if (items is not null) return FromList(items);
            if (properties is not null) return FromMap(properties);
            return FromJson(json);
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
        }
        if (native is not null) native.Dispose();
        else if (items is not null) ReleaseAll(items);
        else if (properties is not null) ReleaseAll(properties.Values);
    }

    internal static RuntimeValue FromJson(JsonNode? value) => new(value);
    internal static RuntimeValue Own(IDisposable value) => new(new NativeOwner(value));
    internal static RuntimeValue FromList(IEnumerable<RuntimeValue> values)
    {
        var retained = new List<RuntimeValue>();
        try
        {
            foreach (var value in values) retained.Add(value.Retain());
            return new(retained.AsReadOnly());
        }
        catch { ReleaseAll(retained); throw; }
    }
    internal static RuntimeValue FromMap(IReadOnlyDictionary<string, RuntimeValue> values)
    {
        var retained = new Dictionary<string, RuntimeValue>(StringComparer.Ordinal);
        try
        {
            foreach (var (key, value) in values) retained.Add(key, value.Retain());
            return new(new ReadOnlyDictionary<string, RuntimeValue>(retained));
        }
        catch { ReleaseAll(retained.Values); throw; }
    }
    internal static void ReleaseAll(IEnumerable<RuntimeValue> values)
    {
        List<Exception>? errors = null;
        foreach (var value in values)
        {
            try { value.Dispose(); }
            catch (Exception e) { (errors ??= []).Add(e); }
        }
        if (errors is not null) throw new AggregateException("Resource disposal failed.", errors);
    }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);

    private sealed class NativeOwner(IDisposable resource) : IDisposable
    {
        private int references = 1;
        public IDisposable Resource { get; } = resource;
        public void Retain() => Interlocked.Increment(ref references);
        public void Dispose() { if (Interlocked.Decrement(ref references) == 0) Resource.Dispose(); }
    }
}

/// <summary>
/// Owns invocation inputs, allocations and temporary values until invocation completion, including asynchronous failure.
/// Nodes borrow inputs and return borrowed or context-created values. The engine retains outputs before disposing this scope.
/// </summary>
public sealed class RuntimeNodeContext : IDisposable
{
    private readonly object gate = new();
    private readonly List<RuntimeValue> values = [];
    private readonly Dictionary<object, RuntimeValue> resources = new(ReferenceEqualityComparer.Instance);
    private bool disposed;

    public RuntimeValue Json(JsonNode? value) { lock (gate) { ThrowIfDisposed(); return Track(RuntimeValue.FromJson(value)); } }
    /// <summary>Transfers sole resource ownership to this scope. Register allocations immediately, before any await or operation that can fail.</summary>
    public RuntimeValue Own<T>(T resource) where T : class, IDisposable
    {
        ArgumentNullException.ThrowIfNull(resource);
        lock (gate)
        {
            ThrowIfDisposed();
            if (resources.TryGetValue(resource, out var existing)) return Retain(existing);
            var value = Track(RuntimeValue.Own(resource));
            resources.Add(resource, value);
            return value;
        }
    }
    public RuntimeValue List(IEnumerable<RuntimeValue> items) { lock (gate) { ThrowIfDisposed(); return Track(RuntimeValue.FromList(items)); } }
    public RuntimeValue Map(IReadOnlyDictionary<string, RuntimeValue> properties) { lock (gate) { ThrowIfDisposed(); return Track(RuntimeValue.FromMap(properties)); } }
    public RuntimeValue Retain(RuntimeValue value) { lock (gate) { ThrowIfDisposed(); return Track(value.Retain()); } }
    private RuntimeValue Track(RuntimeValue value) { values.Add(value); return value; }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
    public void Dispose()
    {
        RuntimeValue[] owned;
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            owned = values.ToArray();
            values.Clear(); resources.Clear();
        }
        RuntimeValue.ReleaseAll(owned);
    }
}

public sealed class OwnedExecutionResult : IDisposable
{
    private readonly RuntimeNodeContext ownership = new();
    private readonly IReadOnlyDictionary<string, IReadOnlyList<IReadOnlyList<RuntimeValue>>> outputs;
    private bool disposed;
    public string Status { get; }
    public IReadOnlyList<EngineDiagnostic> Diagnostics { get; }
    public IReadOnlyDictionary<string, IReadOnlyList<IReadOnlyList<RuntimeValue>>> Outputs
    {
        get { ObjectDisposedException.ThrowIf(disposed, this); return outputs; }
    }
    /// <summary>Retains independent result leases; the supplied values remain owned by the caller.</summary>
    public OwnedExecutionResult(string status, IReadOnlyDictionary<string, IReadOnlyList<IReadOnlyList<RuntimeValue>>> values,
        IReadOnlyList<EngineDiagnostic> diagnostics)
    {
        Status = status;
        Diagnostics = Array.AsReadOnly(diagnostics.ToArray());
        try
        {
            outputs = new ReadOnlyDictionary<string, IReadOnlyList<IReadOnlyList<RuntimeValue>>>(values.ToDictionary(p => p.Key,
                p => (IReadOnlyList<IReadOnlyList<RuntimeValue>>)Array.AsReadOnly(p.Value.Select(slot =>
                    (IReadOnlyList<RuntimeValue>)Array.AsReadOnly(slot.Select(ownership.Retain).ToArray())).ToArray()), StringComparer.Ordinal));
        }
        catch { ownership.Dispose(); throw; }
    }
    public void Dispose() { if (disposed) return; disposed = true; ownership.Dispose(); }
}
