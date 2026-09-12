using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>One owned model-input group; indices select the matching global conditioning rows.</summary>
public sealed class SdTrainingBatchGroup
{
    internal SdTrainingBatchGroup(long[] indices, Tensor latent, Tensor noise, Tensor sigmas)
        => (Indices, Latent, Noise, Sigmas) = (Array.AsReadOnly(indices), latent, noise, sigmas);
    public IReadOnlyList<long> Indices { get; }
    /// <summary>Borrowed until the owning batch is disposed.</summary>
    public Tensor Latent { get; }
    public Tensor Noise { get; }
    public Tensor Sigmas { get; }
    internal void Detach() { Latent.DetachFromDisposeScope(); Noise.DetachFromDisposeScope(); Sigmas.DetachFromDisposeScope(); }
    internal void Release() { Latent.Dispose(); Noise.Dispose(); Sigmas.Dispose(); }
}

public sealed class SdTrainingBatch : IDisposable
{
    private bool disposed;
    internal SdTrainingBatch(long index, ulong seed, SdTrainingBatchGroup[] groups)
        => (Index, NoiseSeed, Groups) = (index, seed, Array.AsReadOnly(groups));
    public long Index { get; }
    public ulong NoiseSeed { get; }
    public IReadOnlyList<SdTrainingBatchGroup> Groups { get; }
    public double LossWeightPerGroup => 1.0 / Groups.Count;
    public void Dispose() { if (disposed) return; disposed = true; foreach (var group in Groups) group.Release(); }
}

/// <summary>Serial source batch/RNG sequence using a private CPU generator. Never changes the global RNG.
/// Native model evaluation must not consume this generator. Mid-batch failures fault the sampler;
/// construct a new training job rather than continuing from an ambiguous random state.</summary>
public sealed class SdTrainingBatchSampler : IDisposable
{
    private readonly SdTrainingDataset dataset;
    private readonly Device device;
    private readonly ulong seed;
    private readonly int batchSize;
    private readonly long[] offsets;
    private readonly Tensor? bucketWeights;
    private Generator generator;
    private long nextIndex;
    private bool disposed, faulted;

    public SdTrainingBatchSampler(SdTrainingDataset dataset, ulong seed, int batchSize, Device device,
        long maxInitialNoiseBytes = 512L * 1024 * 1024, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(dataset);
        if (batchSize < 1 || batchSize > 10000) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (maxInitialNoiseBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxInitialNoiseBytes));
        this.device = InferenceDevice.Validate(device); this.seed = seed; this.batchSize = batchSize;
        this.dataset = dataset.Retain();
        try
        {
            using var scope = NewDisposeScope(); using var noGrad = no_grad();
            long[] initialShape = (long[])this.dataset.Group(0).shape.Clone(); initialShape[0] = this.dataset.Count;
            long bytes = checked(initialShape.Aggregate(1L, (a, b) => checked(a * b)) * sizeof(float));
            if (bytes > maxInitialNoiseBytes) throw new ArgumentOutOfRangeException(nameof(maxInitialNoiseBytes), "Initial guide noise exceeds the configured allowance.");
            offsets = new long[this.dataset.GroupCount]; var weights = new float[offsets.Length];
            for (int i = 0; i < offsets.Length; i++)
            {
                if (i > 0) offsets[i] = checked(offsets[i - 1] + this.dataset.Group(i - 1).shape[0]);
                weights[i] = this.dataset.Group(i).shape[0];
            }
            generator = new Generator(seed, CPU);
            // _run_training_loop generates full dummy noise before the first selection; its RNG consumption matters.
            using var initial = randn(initialShape, generator: generator, dtype: ScalarType.Float32, device: CPU);
            cancellationToken.ThrowIfCancellationRequested();
            if (this.dataset.Mode == SdTrainingDatasetMode.Buckets) bucketWeights = tensor(weights).DetachFromDisposeScope();
        }
        catch { generator?.Dispose(); bucketWeights?.Dispose(); this.dataset.Dispose(); throw; }
    }

    public Tensor CaptureRandomState() { ThrowIfUnavailable(); return generator.get_state(); }

    public SdTrainingBatch Next(CancellationToken cancellationToken = default)
    {
        ThrowIfUnavailable(); cancellationToken.ThrowIfCancellationRequested();
        try
        {
            using var scope = NewDisposeScope(); using var noGrad = no_grad();
            ulong noiseSeed = checked(seed + checked((ulong)nextIndex * 1000));
            var groups = new List<SdTrainingBatchGroup>();
            if (dataset.Mode == SdTrainingDatasetMode.Buckets)
            {
                using var selectedBucket = bucketWeights!.multinomial(1, replacement: false, generator: generator);
                int bucket = checked((int)selectedBucket.item<long>());
                using var indices = Select(dataset.Group(bucket).shape[0]);
                var global = indices.data<long>().ToArray().Select(i => checked(i + offsets[bucket])).ToArray();
                using var latent = dataset.Group(bucket).index_select(0, indices);
                groups.Add(CreateGroup(global, latent, noiseSeed, cancellationToken));
            }
            else
            {
                using var indices = Select(dataset.Count); var selected = indices.data<long>().ToArray();
                if (dataset.Mode == SdTrainingDatasetMode.Standard)
                {
                    using var latent = dataset.Group(0).index_select(0, indices);
                    groups.Add(CreateGroup(selected, latent, noiseSeed, cancellationToken));
                }
                else foreach (long index in selected)
                    groups.Add(CreateGroup([index], dataset.Group(checked((int)index)), noiseSeed, cancellationToken));
            }
            cancellationToken.ThrowIfCancellationRequested();
            var batch = new SdTrainingBatch(nextIndex, noiseSeed, groups.ToArray());
            foreach (var group in groups) group.Detach();
            nextIndex++; return batch;
        }
        catch { faulted = true; throw; }
    }

    private Tensor Select(long count)
    {
        using var scope = NewDisposeScope();
        var indices = randperm(count, dtype: ScalarType.Int64, device: CPU, generator: generator).narrow(0, 0, Math.Min(count, batchSize));
        return indices.MoveToOuterDisposeScope();
    }

    private SdTrainingBatchGroup CreateGroup(long[] indices, Tensor source, ulong noiseSeed, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested(); using var scope = NewDisposeScope();
        // prepare_noise calls manual_seed each time, even for every individual multi-resolution sample.
        var replacement = new Generator(noiseSeed, CPU); generator.Dispose(); generator = replacement;
        var noise = randn(source.shape, generator: generator, dtype: ScalarType.Float32, device: CPU);
        var sigmas = new float[source.shape[0]];
        for (int i = 0; i < sigmas.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested(); using var percent = rand(1, dtype: ScalarType.Float32, device: CPU, generator: generator);
            sigmas[i] = (float)SdDiscreteSampling.Default.PercentToSigma(percent.item<float>(), cancellationToken);
        }
        var latentResult = source.to(device, copy: true); var noiseResult = noise.to(device); var sigmaResult = tensor(sigmas, device: device);
        cancellationToken.ThrowIfCancellationRequested();
        return new(indices, latentResult.MoveToOuterDisposeScope(), noiseResult.MoveToOuterDisposeScope(), sigmaResult.MoveToOuterDisposeScope());
    }

    private void ThrowIfUnavailable()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (faulted) throw new InvalidOperationException("The batch sampler failed after random-state consumption and must be disposed.");
    }
    public void Dispose()
    {
        if (disposed) return; disposed = true;
        generator.Dispose(); bucketWeights?.Dispose(); dataset.Dispose();
    }
}
