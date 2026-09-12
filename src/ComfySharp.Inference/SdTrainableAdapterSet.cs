using System.Collections.ObjectModel;
using System.Security.Cryptography;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Owns all ordinary LoRA and BiasDiff targets in the frozen plain SD module traversal order.
/// New Float32 adapters only; loading existing adapters and other algorithms are separate capabilities.</summary>
public sealed class SdTrainableAdapterSet : IDisposable
{
    private readonly Dictionary<string, TrainableWeightPatch> patches;
    private readonly IReadOnlyDictionary<string, TrainableWeightPatch> view;
    private bool disposed;
    private static readonly Lazy<string[]> Order = new(() =>
    {
        using var stream = typeof(SdTrainableAdapterSet).Assembly.GetManifestResourceStream("ComfySharp.Inference.unet-training-order.json")
            ?? throw new InvalidDataException("Missing frozen SD module traversal order.");
        return JsonSerializer.Deserialize<string[]>(stream) ?? throw new InvalidDataException("Invalid SD module traversal order.");
    });

    public SdTrainableAdapterSet(SdUnetConfig config, int rank, ulong seed, Device device,
        long maxParameterBytes = 512L * 1024 * 1024, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(config);
        if (rank < 1 || rank > 1024) throw new ArgumentOutOfRangeException(nameof(rank));
        if (maxParameterBytes < 0) throw new ArgumentOutOfRangeException(nameof(maxParameterBytes));
        var schema = UnetWeightSchema.Describe(config); var order = Order.Value;
        if (order.Length != schema.Count || !order.ToHashSet(StringComparer.Ordinal).SetEquals(schema.Keys))
            throw new InvalidDataException("The SD schema differs from the frozen training traversal.");
        long bytes = 0;
        foreach (string name in order)
        {
            var shape = schema[name];
            long count = shape.Count == 1 ? shape[0] : checked(checked((shape[0] + shape.Skip(1).Aggregate(1L, (a, b) => checked(a * b))) * rank) + 1);
            bytes = checked(bytes + checked(count * sizeof(float)));
        }
        if (bytes > maxParameterBytes) throw new NotSupportedException("Adapter leaves exceed the configured allowance; temporary tensors and optimizer state are additional.");
        NativeRuntimeBootstrap.Initialize(); device = InferenceDevice.Validate(device);
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        using var cpuGenerator = new Generator(seed, CPU);
        using var acceleratorGenerator = device.type == DeviceType.CPU ? null : NativeGenerator.Create(seed, device);
        var weightGenerator = acceleratorGenerator ?? cpuGenerator;
        patches = new(StringComparer.Ordinal); view = new ReadOnlyDictionary<string, TrainableWeightPatch>(patches);
        try
        {
            foreach (string name in order)
            {
                cancellationToken.ThrowIfCancellationRequested(); using var iteration = NewDisposeScope();
                var shape = schema[name];
                if (shape.Count == 1) patches.Add(name, new TrainableDifferencePatch(zeros(shape.ToArray(), device: device)));
                else
                {
                    long columns = shape.Skip(1).Aggregate(1L, (a, b) => checked(a * b));
                    var up = empty([shape[0], rank], device: device); var down = zeros([rank, columns], device: device);
                    Kaiming(up, rank, weightGenerator);
                    // LoraDiff constructs two default CPU nn.Linear layers before copying up/down.
                    // Their discarded initial values still consume CPU RNG, including for CUDA factors.
                    using var discardedUp = empty([shape[0], rank], device: CPU); Kaiming(discardedUp, rank, cpuGenerator);
                    using var discardedDown = empty([rank, columns], device: CPU); Kaiming(discardedDown, columns, cpuGenerator);
                    patches.Add(name, new TrainableLoraPatch(up, down, 1, trainAlpha: true));
                }
            }
            using var cpuState = cpuGenerator.get_state(); InitialCpuRandomStateSha256 = Convert.ToHexStringLower(SHA256.HashData(cpuState.bytes));
            if (acceleratorGenerator is not null)
            {
                using var deviceState = acceleratorGenerator.get_state(); InitialDeviceRandomStateSha256 = Convert.ToHexStringLower(SHA256.HashData(deviceState.bytes));
            }
            ParameterBytes = bytes; cancellationToken.ThrowIfCancellationRequested();
        }
        catch { foreach (var patch in patches.Values) patch.Dispose(); throw; }
    }
    private static void Kaiming(Tensor tensor, long fanIn, Generator generator)
    {
        double a = Math.Sqrt(5), gain = Math.Sqrt(2 / (1 + a * a));
        double bound = Math.Sqrt(3) * (gain / Math.Sqrt(fanIn));
        tensor.uniform_(-bound, bound, generator);
    }
    public long ParameterBytes { get; }
    public string InitialCpuRandomStateSha256 { get; }
    public string? InitialDeviceRandomStateSha256 { get; }
    /// <summary>Borrowed until this set is disposed. Retain individual patches for longer operations.</summary>
    public IReadOnlyDictionary<string, TrainableWeightPatch> Patches { get { ObjectDisposedException.ThrowIf(disposed, this); return view; } }
    public void Dispose() { if (disposed) return; disposed = true; foreach (var patch in patches.Values) patch.Dispose(); }
}
