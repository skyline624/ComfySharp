using TorchSharp;

namespace ComfySharp.Inference;

/// <summary>Independently disposable owner of all frozen plain SD U-Net parameters.</summary>
public sealed class UnetWeightSet : IDisposable
{
    private readonly CpuModelWeightBank bank;
    private readonly IReadOnlyDictionary<string, LoraWeightPatch> bypass;
    public SdUnetConfig Config { get; }
    public torch.Device Device => bank.Device;
    public UnetWeightSet To(torch.Device device, CancellationToken cancellationToken = default) => Copy(bank.To(device, cancellationToken), bypass, p => p.To(device), cancellationToken);
    private UnetWeightSet(SdUnetConfig config, CpuModelWeightBank bank, IReadOnlyDictionary<string, LoraWeightPatch>? bypass = null)
    { Config = config; this.bank = bank; this.bypass = bypass ?? new Dictionary<string, LoraWeightPatch>(); }
    private UnetWeightSet Copy(CpuModelWeightBank next, IReadOnlyDictionary<string, LoraWeightPatch> factors, Func<LoraWeightPatch, LoraWeightPatch> retain, CancellationToken cancellationToken = default)
    {
        var owned = new Dictionary<string, LoraWeightPatch>(StringComparer.Ordinal);
        try
        {
            foreach (var (name, patch) in factors) { cancellationToken.ThrowIfCancellationRequested(); owned.Add(name, retain(patch)); }
            cancellationToken.ThrowIfCancellationRequested();
            return new(Config, next, owned);
        }
        catch { next.Dispose(); foreach (var patch in owned.Values) patch.Dispose(); throw; }
    }
    /// <summary>Ownership transfers only after complete validation and normalization to 64-byte aligned storage.
    /// Misaligned tensors are copied bit-for-bit into native allocations; their original wrappers are disposed
    /// only on success. Failure leaves every supplied wrapper owned by the caller. Success detaches retained
    /// tensors from ambient scopes; callers must no longer mutate or dispose any transferred wrapper.
    /// Atomic normalization may temporarily require input bytes plus up to a complete second weight bank.
    /// Already aligned tensors, including checkpoint-loader allocations, are retained without copying.</summary>
    public static UnetWeightSet FromOwnedTensors(SdUnetConfig config, IReadOnlyDictionary<string, torch.Tensor> tensors)
        => new(config, CpuModelWeightBank.Create(UnetWeightSchema.Describe(config), tensors));
    public UnetWeightSet Retain() => Copy(bank.Retain(), bypass, p => p.Retain());
    public UnetWeightSet WithLora(IReadOnlyDictionary<string,LoraWeightPatch> patches,
        long maxPatchedWeightBytes=512L*1024*1024,CancellationToken cancellationToken=default)
        => Copy(bank.WithLora(patches,maxPatchedWeightBytes,cancellationToken), bypass, p => p.Retain(), cancellationToken);
    internal UnetWeightSet WithBypassLora(IReadOnlyDictionary<string, LoraWeightPatch> patches,
        long maxPatchedWeightBytes, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(patches);
        var schema = UnetWeightSchema.Describe(Config);
        var regular = new Dictionary<string, LoraWeightPatch>(StringComparer.Ordinal);
        var factors = new Dictionary<string, LoraWeightPatch>(StringComparer.Ordinal);
        foreach (var (name, patch) in patches)
        {
            cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(patch);
            if (!schema.TryGetValue(name, out var shape)) throw new ArgumentException("Unknown bypass target: " + name, nameof(patches));
            if (patch.IsDifference) regular.Add(name, patch);
            else
            {
                if (shape.Count < 2 || !name.EndsWith(".weight", StringComparison.Ordinal)) throw new ArgumentException("Bypass factors require a linear or convolution weight target.", nameof(patches));
                factors.Add(name, patch);
            }
        }
        var next = bank.WithLora(regular, maxPatchedWeightBytes, cancellationToken);
        // The source replaces the named bypass_lora injection group only when new adapters exist.
        return factors.Count == 0 ? Copy(next, bypass, p => p.Retain(), cancellationToken) : Copy(next, factors, p => p.To(next.Device), cancellationToken);
    }
    internal torch.Tensor ApplyBypass(string prefix, torch.Tensor input, torch.Tensor baseOutput,
        IReadOnlyList<long>? kernelSize = null, long stride = 1, long padding = 0)
        => bypass.TryGetValue(prefix + ".weight", out var patch)
            ? patch.ApplyBypass(input, baseOutput, kernelSize, stride, padding) : baseOutput;
    internal torch.Tensor GetTensor(string name) => bank.GetTensor(name);
    internal UnetWeightSet WithTrainingLora(IReadOnlyDictionary<string, TrainableWeightPatch> patches,
        long maxPatchedWeightBytes, CancellationToken cancellationToken)
        => Copy(bank.WithTrainingLora(patches, maxPatchedWeightBytes, cancellationToken), bypass, p => p.Retain(), cancellationToken);
    public void Dispose() { bank.Dispose(); foreach (var patch in bypass.Values) patch.Dispose(); }
}
