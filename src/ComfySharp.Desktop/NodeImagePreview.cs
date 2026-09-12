using System.ComponentModel;
using System.Windows.Input;
using Avalonia.Media.Imaging;

namespace ComfySharp.Desktop;

/// <summary>Document-owned preview: one decoded frame, cancellable navigation, no persisted pixels.</summary>
public sealed class NodeImagePreview : INotifyPropertyChanged, IDisposable
{
    private readonly PreviewImageFile[] files;
    private readonly Func<PreviewImageFile, CancellationToken, Task<byte[]>> read;
    private CancellationTokenSource? pending;
    private long sequence;
    private int index;
    public bool IsDisposed { get; private set; }
    public Bitmap? Image { get; private set; }
    public string Status { get; private set; } = "";
    public string Position => files.Length == 0 ? "No images" : $"{index + 1} / {files.Length}";
    public string Filename => files.Length == 0 ? "" : files[index].Filename;
    public int Count => files.Length;
    public bool CanPrevious => !IsDisposed && index > 0;
    public bool CanNext => !IsDisposed && index + 1 < files.Length;
    public ICommand Previous { get; }
    public ICommand Next { get; }
    public event PropertyChangedEventHandler? PropertyChanged;

    public NodeImagePreview(IEnumerable<PreviewImageFile> files, Func<PreviewImageFile, CancellationToken, Task<byte[]>> read)
    {
        this.files = files.ToArray(); this.read = read;
        Previous = new Navigation(() => CanPrevious, () => LoadAsync(index - 1));
        Next = new Navigation(() => CanNext, () => LoadAsync(index + 1));
    }

    public async Task LoadAsync(int selected = 0)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        if (files.Length == 0) { Status = "No images returned."; Notify(); return; }
        ArgumentOutOfRangeException.ThrowIfNegative(selected); ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(selected, files.Length);
        pending?.Cancel();
        using var cancellation = new CancellationTokenSource(); pending = cancellation;
        long request = ++sequence; index = selected; Replace(null); Status = "Loading image…"; Notify();
        try
        {
            var png = await read(files[selected], cancellation.Token);
            if (IsDisposed || request != sequence || cancellation.IsCancellationRequested) return;
            var bitmap = ImagePreviewTransport.Decode(png);
            // Decoding is synchronous on the UI thread; no editor mutation can interleave with publication.
            Replace(bitmap); Status = ""; Notify();
        }
        catch (OperationCanceledException) when (IsDisposed || request != sequence || cancellation.IsCancellationRequested) { }
        catch (Exception error)
        {
            if (!IsDisposed && request == sequence) { Status = "Preview unavailable: " + error.Message; Notify(); }
        }
        finally { if (ReferenceEquals(pending, cancellation)) pending = null; }
    }

    private void Replace(Bitmap? bitmap)
    {
        var old = Image; Image = bitmap; PropertyChanged?.Invoke(this, new(nameof(Image))); old?.Dispose();
    }
    private void Notify()
    {
        PropertyChanged?.Invoke(this, new(null)); ((Navigation)Previous).Changed(); ((Navigation)Next).Changed();
    }
    public void Dispose()
    {
        if (IsDisposed) return; IsDisposed = true; sequence++; pending?.Cancel(); Replace(null); Status = ""; Notify();
    }
    private sealed class Navigation(Func<bool> enabled, Func<Task> run) : ICommand
    {
        public event EventHandler? CanExecuteChanged;
        public bool CanExecute(object? parameter) => enabled();
        public async void Execute(object? parameter) { if (enabled()) await run(); }
        public void Changed() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
    }
}
