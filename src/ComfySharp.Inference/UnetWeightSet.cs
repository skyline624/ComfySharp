using TorchSharp;

namespace ComfySharp.Inference;

/// <summary>Independently disposable owner of all frozen plain SD U-Net parameters.</summary>
public sealed class UnetWeightSet : IDisposable
{
    private readonly CpuModelWeightBank bank;
    public SdUnetConfig Config { get; }
    public torch.Device Device => bank.Device;
    public UnetWeightSet To(torch.Device device, CancellationToken cancellationToken = default) => new(Config, bank.To(device, cancellationToken));
    private UnetWeightSet(SdUnetConfig config, CpuModelWeightBank bank) { Config = config; this.bank = bank; }
    /// <summary>Ownership transfers only after complete validation and normalization to 64-byte aligned storage.
    /// Misaligned tensors are copied bit-for-bit into native allocations; their original wrappers are disposed
    /// only on success. Failure leaves every supplied wrapper owned by the caller. Success detaches retained
    /// tensors from ambient scopes; callers must no longer mutate or dispose any transferred wrapper.
    /// Atomic normalization may temporarily require input bytes plus up to a complete second weight bank.
    /// Already aligned tensors, including checkpoint-loader allocations, are retained without copying.</summary>
    public static UnetWeightSet FromOwnedTensors(SdUnetConfig config, IReadOnlyDictionary<string, torch.Tensor> tensors)
        => new(config, CpuModelWeightBank.Create(UnetWeightSchema.Describe(config), tensors));
    public UnetWeightSet Retain() => new(Config, bank.Retain());
    internal torch.Tensor GetTensor(string name) => bank.GetTensor(name);
    public void Dispose() => bank.Dispose();
}
