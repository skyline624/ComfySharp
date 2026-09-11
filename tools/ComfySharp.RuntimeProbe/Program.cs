using System.Diagnostics;
using System.Text.Json;
using ComfySharp.Inference;
using TorchSharp;
using static TorchSharp.torch;

string requested = "cpu";
try
{
    int repeats = 25;
    for (int i = 0; i < args.Length; i++)
        switch (args[i])
        {
            case "--device" when i + 1 < args.Length: requested = args[++i]; break;
            case "--repeat" when i + 1 < args.Length: repeats = int.Parse(args[++i]); break;
            default: throw new ArgumentException("Usage: --device cpu|cuda|mps --repeat 1..1000");
        }
    if (repeats is < 1 or > 1000) throw new ArgumentOutOfRangeException(nameof(repeats));
    if (requested is not ("cpu" or "cuda" or "mps")) throw new ArgumentException("Unknown backend.");
    if (requested == "mps") throw new PlatformNotSupportedException("MPS has not been implemented or qualified in this probe.");
    if (requested == "cuda" && !cuda.is_available()) throw new PlatformNotSupportedException("CUDA unavailable; this probe ships CPU libtorch by default.");
    var device = requested == "cpu" ? CPU : CUDA;
    var memory = new List<long>();
    float updated = 0, gradient = 0, convolution = 0, attention = 0;
    for (int iteration = 0; iteration < repeats; iteration++)
    {
        using (var scope = NewDisposeScope())
        {
            var parameter = nn.Parameter(tensor(new float[] { 2 }, device: device));
            if (requested == "cuda" && parameter.device.type != DeviceType.CUDA) throw new Exception("Requested CUDA tensor did not land on GPU.");
            using var optimizer = optim.SGD(new[] { parameter }, 0.1);
            optimizer.zero_grad();
            var loss = parameter.square().sum();
            loss.backward();
            gradient = parameter.grad!.item<float>();
            optimizer.step();
            updated = parameter.item<float>();
            if (Math.Abs(gradient - 4) > 1e-6 || Math.Abs(updated - 1.6f) > 1e-6) throw new Exception("Autograd/optimizer assertion failed.");
            var original = tensor(new float[] { 1, 2, 3, 4 }, device: device);
            var view = original.reshape(2, 2);
            original.Dispose();
            if (view.sum().item<float>() != 10) throw new Exception("View storage lifetime failed.");
            var input = ones(new long[] { 1, 1, 3, 3 }, device: device);
            var weights = ones(new long[] { 1, 1, 2, 2 }, device: device);
            convolution = nn.functional.conv2d(input, weights).sum().item<float>();
            var q = ones(new long[] { 1, 1, 2, 2 }, device: device);
            attention = nn.functional.scaled_dot_product_attention(q, q, q).sum().item<float>();
            if (convolution != 16 || Math.Abs(attention - 4) > 1e-5) throw new Exception("Convolution/attention assertion failed.");
            var noise1 = NativeMath.CpuNoise(new long[] { 16 }, 123);
            var noise2 = NativeMath.CpuNoise(new long[] { 16 }, 123);
            if (!equal(noise1, noise2).all().item<bool>()) throw new Exception("Native CPU generator reproducibility failed.");
            var transferredNoise = noise1.to(device);
            if (requested == "cuda" && transferredNoise.device.type != DeviceType.CUDA) throw new Exception("CPU noise transfer did not land on GPU.");
            if (!equal(transferredNoise.cpu(), noise1).all().item<bool>()) throw new Exception("CPU noise device roundtrip failed.");
            var matrix = view.matmul(view);
            if (matrix.sum().item<float>() != 54) throw new Exception("Matrix multiplication failed.");
            if (requested == "cuda") cuda.synchronize();
        }
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        memory.Add(process.PrivateMemorySize64);
    }
    Console.WriteLine(JsonSerializer.Serialize(new { success = true, device = requested,
        torchSharp = typeof(Tensor).Assembly.GetName().Version?.ToString(), libtorchPackage = "2.10.0", repeats,
        gradient, updated, convolution, attention, matrixMultiplicationSum = 54, tensorDevice = device.ToString(),
        viewSurvivedOwnerDisposal = true, cpuRngRepeatable = true, cpuNoiseDeviceRoundtrip = true,
        privateBytes = memory, memoryQualification = "Observations after deterministic dispose scopes; allocator caching and process noise prevent a no-leak conclusion.",
        modelInferenceQualified = false }));
    return 0;
}
catch (Exception e)
{
    Console.WriteLine(JsonSerializer.Serialize(new { success = false, device = requested, error = e.GetType().Name, message = e.Message }));
    return 1;
}
