namespace ComfySharp.Desktop;

/// <summary>Discards in-flight responses after a Host restart or window closure.</summary>
public sealed class HostSession
{
    public int Id { get; private set; }
    private bool closed;
    public int Restart() { ObjectDisposedException.ThrowIf(closed, this); return ++Id; }
    public void Close() { closed = true; Id++; }
    public void Require(int id)
    {
        if (closed || id != Id) throw new OperationCanceledException("The Host session ended; pending results are discarded.");
    }
    public async Task<T> ObserveAsync<T>(Task<T> response, int id)
    {
        var result = await response;
        Require(id);
        return result;
    }
}
