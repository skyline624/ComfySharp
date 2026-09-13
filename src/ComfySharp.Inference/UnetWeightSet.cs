using TorchSharp;

namespace ComfySharp.Inference;

/// <summary>Independently disposable owner of all frozen plain SD U-Net parameters.</summary>
public sealed class UnetWeightSet : IDisposable
{
    private readonly CpuModelWeightBank bank;
    private readonly IReadOnlyDictionary<string, LoraWeightPatch> bypass;
    private readonly IReadOnlyDictionary<string, TrainableWeightPatch> trainingBypass;
    public SdUnetConfig Config { get; }
    public torch.Device Device => bank.Device;
    public UnetWeightSet To(torch.Device device, CancellationToken cancellationToken = default) => Copy(bank.To(device, cancellationToken), bypass, p => p.To(device), cancellationToken);
    private UnetWeightSet(SdUnetConfig config, CpuModelWeightBank bank, IReadOnlyDictionary<string, LoraWeightPatch>? bypass = null,
        IReadOnlyDictionary<string, TrainableWeightPatch>? trainingBypass = null)
    { Config = config; this.bank = bank; this.bypass = bypass ?? new Dictionary<string, LoraWeightPatch>(); this.trainingBypass = trainingBypass ?? new Dictionary<string, TrainableWeightPatch>(); }
    private UnetWeightSet Copy(CpuModelWeightBank next, IReadOnlyDictionary<string, LoraWeightPatch> factors, Func<LoraWeightPatch, LoraWeightPatch> retain, CancellationToken cancellationToken = default)
    {
        var owned = new Dictionary<string, LoraWeightPatch>(StringComparer.Ordinal);
        var training = new Dictionary<string, TrainableWeightPatch>(StringComparer.Ordinal);
        try
        {
            foreach (var (name, patch) in factors) { cancellationToken.ThrowIfCancellationRequested(); owned.Add(name, retain(patch)); }
            cancellationToken.ThrowIfCancellationRequested();
            foreach (var (name, patch) in trainingBypass)
            {
                cancellationToken.ThrowIfCancellationRequested();
                foreach(var value in patch.Parameters)InferenceDevice.RequireSame(next.Device,value,name);
                training.Add(name, patch.Retain());
            }
            return new(Config, next, owned, training);
        }
        catch { next.Dispose(); foreach (var patch in owned.Values) patch.Dispose(); foreach (var patch in training.Values) patch.Dispose(); throw; }
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
    {
        using var scope = torch.NewDisposeScope();
        var result = bypass.TryGetValue(prefix + ".weight", out var patch)
            ? patch.ApplyBypass(input, baseOutput, kernelSize, stride, padding) : baseOutput;
        if (trainingBypass.TryGetValue(prefix + ".weight", out var training))
            result = training.ApplyBypass(input, result, kernelSize, stride, padding);
        // Borrowed baseOutput belongs to the caller's scope and must not be moved out of it.
        return ReferenceEquals(result, baseOutput) ? result : result.MoveToOuterDisposeScope();
    }
    internal torch.Tensor GetTensor(string name) => bank.GetTensor(name);
    internal UnetWeightSet WithTrainingLora(IReadOnlyDictionary<string, TrainableWeightPatch> patches,
        long maxPatchedWeightBytes, CancellationToken cancellationToken)
        => Copy(bank.WithTrainingLora(patches, maxPatchedWeightBytes, cancellationToken), bypass, p => p.Retain(), cancellationToken);
    internal UnetWeightSet WithTrainingBypassLora(IReadOnlyDictionary<string, TrainableWeightPatch> patches,
        long maxPatchedWeightBytes, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(patches);
        if (patches.Count == 0) throw new ArgumentException("Training requires at least one adapter.", nameof(patches));
        if (maxPatchedWeightBytes < 0) throw new ArgumentOutOfRangeException(nameof(maxPatchedWeightBytes));
        var factors = new Dictionary<string, TrainableWeightPatch>(StringComparer.Ordinal);
        var regular = new Dictionary<string, TrainableWeightPatch>(StringComparer.Ordinal);
        CpuModelWeightBank? next = null;
        var frozen = new Dictionary<string, LoraWeightPatch>(StringComparer.Ordinal);
        try
        {
            foreach (var (name, patch) in patches)
            {
                cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(patch);
                var weight = bank.GetTensor(name);
                if (patch is TrainableLoraPatch lora)
                {
                    using var retained = lora.Retain();
                    if (weight.dim() < 2 || !name.EndsWith(".weight", StringComparison.Ordinal) ||
                        weight.shape[0] != retained.Up.shape[0] || weight.numel() / weight.shape[0] != retained.Down.shape[1])
                        throw new ArgumentException("Trainable bypass target shape differs: " + name, nameof(patches));
                    InferenceDevice.RequireSame(Device, retained.Up, name);
                    factors.Add(name, retained.Retain());
                }
                else if (patch is TrainableLohaPatch)
                    throw new NotSupportedException("The frozen trainable LohaDiff does not implement bypass execution.");
                else if (patch is TrainableLokrPatch lokr)
                {
                    using var retained=lokr.Retain();
                    if(weight.dim()<2||!name.EndsWith(".weight",StringComparison.Ordinal))throw new ArgumentException("Trainable LoKr bypass needs a module weight: "+name);
                    foreach(var value in retained.Parameters)InferenceDevice.RequireSame(Device,value,name);
                    // Kernel/module details are checked by the actual operator at execution.
                    factors.Add(name,retained.Retain());
                }
                else regular.Add(name, patch);
            }
            next = regular.Count == 0 ? bank.Retain() : bank.WithTrainingLora(regular, maxPatchedWeightBytes, cancellationToken);
            foreach (var (name, patch) in bypass) { cancellationToken.ThrowIfCancellationRequested(); frozen.Add(name, patch.Retain()); }
            cancellationToken.ThrowIfCancellationRequested();
            // This operation-only bank lives until Forward returns; autograd owns saved tensors afterwards.
            return new UnetWeightSet(Config, next, frozen, factors);
        }
        catch { next?.Dispose(); foreach (var patch in frozen.Values) patch.Dispose(); foreach (var patch in factors.Values) patch.Dispose(); throw; }
    }
    public void Dispose() { bank.Dispose(); foreach (var patch in bypass.Values) patch.Dispose(); foreach (var patch in trainingBypass.Values) patch.Dispose(); }
}
