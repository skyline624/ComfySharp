using ComfySharp.Contracts;
using ComfySharp.Inference;
using static TorchSharp.torch;

namespace ComfySharp.Nodes.Tensor;

/// <summary>Publishes completed training adapters using the engine's native value ownership.
/// This bridge does not register a training node or implement its remaining training options.</summary>
public static class TrainingNodeValues
{
    public static RuntimeValue CaptureLosses(RuntimeNodeContext destination,IReadOnlyList<float> losses)
    {
        ArgumentNullException.ThrowIfNull(destination);ArgumentNullException.ThrowIfNull(losses);
        if(losses.Any(v=>!float.IsFinite(v)))throw new ArgumentException("Loss history must contain finite values.",nameof(losses));
        return destination.Map(new Dictionary<string,RuntimeValue>{{"loss",destination.List(losses.Select(v=>destination.Json(System.Text.Json.Nodes.JsonValue.Create(v))))}});
    }
    public static RuntimeValue CaptureAdapters<T>(RuntimeNodeContext destination,
        IReadOnlyDictionary<string, T> targets, ScalarType dtype = ScalarType.BFloat16,
        long maxSnapshotBytes = 512L * 1024 * 1024, CancellationToken cancellationToken = default)
        where T : TrainableWeightPatch
    {
        ArgumentNullException.ThrowIfNull(destination);
        using var state = LoraTrainingState.Capture(targets, dtype, maxSnapshotBytes: maxSnapshotBytes, cancellationToken: cancellationToken);
        using var local = new RuntimeNodeContext();
        var values = new Dictionary<string, RuntimeValue>(StringComparer.Ordinal);
        foreach (var (key, tensor) in state.Tensors)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var alias = tensor.detach();
            try { values.Add(key, local.Own(alias)); }
            catch { alias.Dispose(); throw; }
            alias.DetachFromDisposeScope();
        }
        cancellationToken.ThrowIfCancellationRequested();
        return destination.Retain(local.Map(values));
    }
}
