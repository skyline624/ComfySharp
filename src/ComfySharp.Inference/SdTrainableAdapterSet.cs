using System.Collections.ObjectModel;
using System.Security.Cryptography;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Owns LoRA/LoHa and BiasDiff targets in the frozen plain SD module traversal order.
/// Fresh Float32 LoRA/LoHa and resumed two-factor LoRA; other resume algorithms remain separate capabilities.</summary>
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
        long maxParameterBytes = 512L * 1024 * 1024, CancellationToken cancellationToken = default, ILoraTensorSource? existing = null,
        string algorithm = "LoRA")
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(config);
        if (rank < 1 || rank > 1024) throw new ArgumentOutOfRangeException(nameof(rank));
        if (algorithm is not ("LoRA" or "LoHa")) throw new NotSupportedException("This SD training adapter algorithm remains unported: " + algorithm);
        if (maxParameterBytes < 0) throw new ArgumentOutOfRangeException(nameof(maxParameterBytes));
        var schema = UnetWeightSchema.Describe(config); var order = Order.Value;
        if (order.Length != schema.Count || !order.ToHashSet(StringComparer.Ordinal).SetEquals(schema.Keys))
            throw new InvalidDataException("The SD schema differs from the frozen training traversal.");
        var resume = existing is null ? null : SdLoraResumePlan.Inspect(existing,schema,cancellationToken);
        ResumedTargets = Array.AsReadOnly(order.Where(n => resume?.Targets.ContainsKey(n) == true).ToArray());
        IgnoredExistingKeys = resume?.IgnoredKeys ?? Array.Empty<string>();
        long bytes = 0;
        foreach (string name in order)
        {
            var shape = schema[name];
            long actualRank = resume?.Targets.GetValueOrDefault(name)?.Rank ?? rank;
            int pairs = algorithm == "LoHa" && resume?.Targets.ContainsKey(name) != true ? 2 : 1;
            long count = shape.Count == 1 ? shape[0] : checked(checked((shape[0] + shape.Skip(1).Aggregate(1L, (a, b) => checked(a * b))) * actualRank * pairs) + 1);
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
                    if (algorithm == "LoHa" && resume?.Targets.ContainsKey(name) != true)
                    {
                        // In source normal_(tensor, 0.1), 0.1 is the mean; std remains 1.
                        var a = empty([shape[0], rank], device: device).normal_(.1, 1, weightGenerator);
                        var b = zeros([rank, columns], device: device);
                        var c = empty([shape[0], rank], device: device).normal_(.1, 1, weightGenerator);
                        var d = empty([rank, columns], device: device).normal_(.01, 1, weightGenerator);
                        patches.Add(name, new TrainableLohaPatch(a, b, c, d));
                        continue; // LohaDiff has no discarded random Linear constructors.
                    }
                    double alpha = 1;
                    // The source reads module.weight.alpha, not the usual exported module.alpha.
                    if (resume?.Alphas.TryGetValue(name,out string? alphaKey) == true)
                    {
                        using var value = existing!.ReadTensor(alphaKey,cancellationToken);
                        alpha = value.to_type(ScalarType.Float64).item<double>();
                        if (!double.IsFinite(alpha) || !float.IsFinite((float)alpha)) throw new InvalidDataException("Resume alpha must be finite Float32: " + alphaKey);
                    }
                    long actualRank = rank; Tensor up, down;
                    if (resume?.Targets.TryGetValue(name,out var factors) == true)
                    {
                        actualRank = factors.Rank;
                        // LoraDiff copies source weights into default CPU Float32 Linear layers before moving them.
                        up = existing!.ReadTensor(factors.Up,cancellationToken).to_type(ScalarType.Float32).to(device);
                        down = existing.ReadTensor(factors.Down,cancellationToken).to_type(ScalarType.Float32).to(device);
                    }
                    else
                    {
                        up = empty([shape[0], rank], device: device); down = zeros([rank, columns], device: device);
                        Kaiming(up, rank, weightGenerator); alpha = 1;
                    }
                    // LoraDiff constructs two default CPU nn.Linear layers before copying up/down.
                    // Their discarded initial values still consume CPU RNG, including for CUDA factors.
                    using var discardedUp = empty([shape[0], actualRank], device: CPU); Kaiming(discardedUp, actualRank, cpuGenerator);
                    using var discardedDown = empty([actualRank, columns], device: CPU); Kaiming(discardedDown, columns, cpuGenerator);
                    patches.Add(name, new TrainableLoraPatch(up, down, alpha, trainAlpha: true));
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
    public IReadOnlyList<string> ResumedTargets { get; }
    /// <summary>Includes source-reset differences and ordinary exported alpha keys not used by the training factory.</summary>
    public IReadOnlyList<string> IgnoredExistingKeys { get; }
    public string InitialCpuRandomStateSha256 { get; }
    public string? InitialDeviceRandomStateSha256 { get; }
    /// <summary>Borrowed until this set is disposed. Retain individual patches for longer operations.</summary>
    public IReadOnlyDictionary<string, TrainableWeightPatch> Patches { get { ObjectDisposedException.ThrowIf(disposed, this); return view; } }
    public void Dispose() { if (disposed) return; disposed = true; foreach (var patch in patches.Values) patch.Dispose(); }
}
