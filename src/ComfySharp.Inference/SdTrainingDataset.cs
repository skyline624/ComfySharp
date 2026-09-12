using static TorchSharp.torch;

namespace ComfySharp.Inference;

public enum SdTrainingDatasetMode { Standard, MultiResolution, Buckets }

/// <summary>Owned CPU Float32 snapshots of plain SD VAE latents, prepared as in frozen TrainLoraNode.
/// Standard data passes through latent-format scaling. The source's separate multi-resolution/bucket
/// datasets bypass that guider conversion; this class preserves that behavior explicitly.</summary>
public sealed class SdTrainingDataset : IDisposable
{
    private sealed class Shared(Tensor[] groups, SdTrainingDatasetMode mode)
    {
        internal readonly object Gate = new();
        internal readonly Tensor[] Groups = groups;
        internal readonly SdTrainingDatasetMode Mode = mode;
        internal readonly long Count = groups.Sum(g => g.shape[0]);
        internal int Owners = 1;
    }
    private readonly Shared shared;
    private bool disposed;
    private SdTrainingDataset(Shared shared) => this.shared = shared;

    public SdTrainingDataset(IReadOnlyList<Tensor> inputs, bool bucketMode = false,
        double latentScale = SdSamplingMath.Sd15LatentScale, long maxBytes = 512L * 1024 * 1024,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(inputs);
        if (inputs.Count == 0) throw new ArgumentException("A training dataset must contain latents.", nameof(inputs));
        if (maxBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxBytes));
        if (!double.IsFinite(latentScale) || latentScale <= 0) throw new ArgumentOutOfRangeException(nameof(latentScale));
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var noGrad = no_grad();
        long bytes = 0;
        foreach (var input in inputs)
        {
            SdSamplingMath.ValidateLatent(input, nameof(inputs));
            bytes = checked(bytes + checked(input.numel() * sizeof(float)));
            if (bytes > maxBytes) throw new ArgumentOutOfRangeException(nameof(maxBytes), "Dataset snapshots exceed the configured byte allowance.");
        }
        var copies = new List<Tensor>();
        foreach (var input in inputs) { cancellationToken.ThrowIfCancellationRequested(); copies.Add(input.detach().to(CPU, copy: true)); }
        Tensor[] groups; SdTrainingDatasetMode mode;
        if (bucketMode) { groups = copies.ToArray(); mode = SdTrainingDatasetMode.Buckets; }
        else
        {
            var rows = new List<Tensor>();
            if (copies.Count == 1) rows.Add(copies[0]);
            else foreach (var copy in copies) for (long i = 0; i < copy.shape[0]; i++)
            {
                cancellationToken.ThrowIfCancellationRequested(); rows.Add(copy.narrow(0, i, 1));
            }
            if (rows.All(r => r.shape.SequenceEqual(rows[0].shape)))
            {
                var all = rows.Count == 1 ? rows[0] : cat(rows.ToArray(), 0);
                // CFGGuider.inner_sample avoids shifting an empty latent. For SD scaling this also preserves zero bits.
                if (all.ne(0).any().item<bool>()) all = all * latentScale;
                groups = [all]; mode = SdTrainingDatasetMode.Standard;
            }
            else { groups = rows.ToArray(); mode = SdTrainingDatasetMode.MultiResolution; }
        }
        cancellationToken.ThrowIfCancellationRequested(); shared = new(groups, mode);
        foreach (var group in groups) group.DetachFromDisposeScope();
    }

    public long Count { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Count; } } }
    public SdTrainingDatasetMode Mode { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Mode; } } }
    internal int GroupCount { get { lock (shared.Gate) { ThrowIfDisposed(); return shared.Groups.Length; } } }
    internal Tensor Group(int index) { lock (shared.Gate) { ThrowIfDisposed(); return shared.Groups[index]; } }
    public SdTrainingDataset Retain()
    {
        lock (shared.Gate) { ThrowIfDisposed(); shared.Owners = checked(shared.Owners + 1); return new(shared); }
    }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
    public void Dispose()
    {
        lock (shared.Gate)
        {
            if (disposed) return; disposed = true;
            if (--shared.Owners == 0) foreach (var group in shared.Groups) group.Dispose();
        }
    }
}
