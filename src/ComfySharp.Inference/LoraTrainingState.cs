using System.Collections.ObjectModel;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Owns the detached LORA_MODEL output of ordinary LoRA/BiasDiff training.
/// Capture must be serialized with training updates. Borrowed tensors remain valid until disposal;
/// they must not be mutated or disposed by consumers. No file or model weights are involved.</summary>
public sealed class LoraTrainingState : IDisposable
{
    private readonly IReadOnlyDictionary<string, Tensor> tensors;
    private bool disposed;
    private LoraTrainingState(Dictionary<string, Tensor> tensors) =>
        this.tensors = new ReadOnlyDictionary<string, Tensor>(tensors);

    public IReadOnlyDictionary<string, Tensor> Tensors
    {
        get { ObjectDisposedException.ThrowIf(disposed, this); return tensors; }
    }

    /// <summary>Maps canonical .weight/.bias targets to diffusion_model LoRA keys and casts
    /// the completed state to the frozen node's bf16/fp32 output dtype. Storage stays on its device.</summary>
    public static LoraTrainingState Capture<T>(IReadOnlyDictionary<string, T> targets,
        ScalarType dtype = ScalarType.BFloat16, string componentPrefix = "diffusion_model.",
        long maxSnapshotBytes = 512L * 1024 * 1024, CancellationToken cancellationToken = default)
        where T : TrainableWeightPatch
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(targets); ArgumentNullException.ThrowIfNull(componentPrefix);
        if (dtype is not (ScalarType.BFloat16 or ScalarType.Float32))
            throw new ArgumentException("The frozen training node exposes bf16 and fp32 LoRA outputs.", nameof(dtype));
        if (maxSnapshotBytes < 0) throw new ArgumentOutOfRangeException(nameof(maxSnapshotBytes));
        if (targets.Count > 10000) throw new NotSupportedException("Too many adapter targets.");
        var owners = new List<TrainableWeightPatch>();
        var entries = new Dictionary<string, (Tensor? Value, double Alpha, Device Device)>(StringComparer.Ordinal);
        try
        {
            long bytes = 0;
            void Add(string key, Tensor? value, double alpha, Device device)
            {
                bytes = checked(bytes + checked((value?.numel() ?? 1) * (dtype == ScalarType.Float32 ? 4 : 2)));
                if (bytes > maxSnapshotBytes) throw new NotSupportedException("Adapter output exceeds the configured snapshot allowance.");
                if (!entries.TryAdd(key, (value, alpha, device))) throw new ArgumentException("Adapter targets produce duplicate output keys.", nameof(targets));
            }
            foreach (var (name, patch) in targets)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ArgumentException.ThrowIfNullOrWhiteSpace(name); ArgumentNullException.ThrowIfNull(patch);
                bool bias = name.EndsWith(".bias", StringComparison.Ordinal);
                if (!bias && !name.EndsWith(".weight", StringComparison.Ordinal))
                    throw new ArgumentException("Adapter targets must end in .weight or .bias.", nameof(targets));
                string prefix = componentPrefix + name[..^(bias ? 5 : 7)];
                var owner = patch.Retain(); owners.Add(owner);
                switch (owner)
                {
                    case TrainableDifferencePatch difference:
                        Add(prefix + (bias ? ".diff_b" : ".diff"), difference.Difference, 0, difference.Difference.device);
                        break;
                    case TrainableLoraPatch lora when !bias:
                        Add(prefix + ".lora_up.weight", lora.Up, 0, lora.Up.device);
                        Add(prefix + ".lora_down.weight", lora.Down, 0, lora.Down.device);
                        var alpha = lora.AlphaParameter;
                        Add(prefix + ".alpha", alpha, alpha is null ? lora.Alpha : 0, lora.Up.device);
                        break;
                    default: throw new NotSupportedException("This adapter target is not an ordinary LoRA weight or additive difference.");
                }
            }
            NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var noGrad = no_grad();
            var snapshots = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            foreach (var (key, entry) in entries)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var iteration = NewDisposeScope();
                // A fixed alpha is a Float32 parameter in the training contract before output casting.
                var original = entry.Value ?? tensor((float)entry.Alpha, device: entry.Device);
                snapshots.Add(key, original.detach().to_type(dtype).clone().contiguous().MoveToOuterDisposeScope());
            }
            cancellationToken.ThrowIfCancellationRequested();
            var state = new LoraTrainingState(snapshots);
            foreach (var snapshot in snapshots.Values) snapshot.DetachFromDisposeScope();
            return state;
        }
        finally { foreach (var owner in owners) owner.Dispose(); }
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        foreach (var tensor in tensors.Values) tensor.Dispose();
    }
}
