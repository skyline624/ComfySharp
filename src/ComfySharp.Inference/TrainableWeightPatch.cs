using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Retained owner of Float32 adapter leaves. Parameters are borrowed and callers serialize
/// forward, backward, mutation and disposal. Only supported built-in patch kinds can derive this type.</summary>
public abstract class TrainableWeightPatch : IDisposable
{
    private protected TrainableWeightPatch() { }
    public abstract IReadOnlyList<Tensor> Parameters { get; }
    public abstract TrainableWeightPatch Retain();
    internal abstract Tensor Apply(Tensor weight, CancellationToken cancellationToken);
    public abstract void Dispose();

    internal static IReadOnlyDictionary<string, TrainableWeightPatch> Widen<T>(IReadOnlyDictionary<string, T> patches)
        where T : TrainableWeightPatch
    {
        ArgumentNullException.ThrowIfNull(patches);
        return patches as IReadOnlyDictionary<string, TrainableWeightPatch>
            ?? patches.ToDictionary(p => p.Key, p => (TrainableWeightPatch)p.Value, StringComparer.Ordinal);
    }
}
