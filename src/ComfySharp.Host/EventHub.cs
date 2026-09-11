using System.Collections.Concurrent;
using System.Text.Json.Nodes;
using System.Threading.Channels;

namespace ComfySharp.Host;

public sealed class EventHub
{
    private sealed record Subscriber(string? ClientId, Channel<string> Channel);
    private readonly ConcurrentDictionary<Guid, Subscriber> subscribers = new();
    private readonly object gate = new();

    public (Guid Id, ChannelReader<string> Reader) Subscribe(string? clientId = null)
    {
        var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(256)
        { SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
        var id = Guid.NewGuid();
        lock (gate)
        {
            if (clientId is not null)
                foreach (var previous in subscribers.Where(s => s.Value.ClientId == clientId).Select(s => s.Key).ToArray()) Remove(previous);
            subscribers[id] = new(clientId, channel);
        }
        return (id, channel.Reader);
    }

    public void Remove(Guid id)
    {
        lock (gate)
            if (subscribers.TryRemove(id, out var subscriber)) subscriber.Channel.Writer.TryComplete();
    }

    public void Publish(string type, JsonNode data, string? clientId = null)
    {
        var json = new JsonObject { ["type"] = type, ["data"] = data.DeepClone() }.ToJsonString();
        lock (gate)
            foreach (var (id, subscriber) in subscribers)
                if ((clientId is null || subscriber.ClientId == clientId) && !subscriber.Channel.Writer.TryWrite(json))
                    Remove(id); // Disconnect a slow client rather than silently losing terminal events.
    }
}
