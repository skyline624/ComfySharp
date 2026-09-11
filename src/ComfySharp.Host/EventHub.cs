using System.Collections.Concurrent;
using System.Text.Json.Nodes;
using System.Threading.Channels;

namespace ComfySharp.Host;

public sealed class EventHub
{
    private readonly ConcurrentDictionary<Guid, Channel<string>> subscribers = new();

    public (Guid Id, ChannelReader<string> Reader) Subscribe()
    {
        var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(256)
        { SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
        var id = Guid.NewGuid();
        subscribers[id] = channel;
        return (id, channel.Reader);
    }

    public void Remove(Guid id)
    {
        if (subscribers.TryRemove(id, out var channel)) channel.Writer.TryComplete();
    }

    public void Publish(string type, JsonNode data)
    {
        var json = new JsonObject { ["type"] = type, ["data"] = data.DeepClone() }.ToJsonString();
        foreach (var (id, channel) in subscribers)
            if (!channel.Writer.TryWrite(json)) Remove(id); // Disconnect a slow client rather than silently losing terminal events.
    }
}
