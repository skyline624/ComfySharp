using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Test-only, explicitly requested reduced U-Net/CFG evidence. No product hook is added.</summary>
internal static class SdNativeSuiteDiagnostic
{
    private const string Variable = "COMFYSHARP_SD_NATIVE_SUITE_TRACE_DIR";

    internal static bool Unet(SdUnet model, UnetWeightSet bank, Tensor latent, Tensor timesteps, Tensor context,
        JsonElement reference, JsonElement parameterReferences, JsonElement config)
    {
        string? directory = Environment.GetEnvironmentVariable(Variable);
        if (string.IsNullOrWhiteSpace(directory)) return false;
        ValidateOptions();
        var shapes = reference.GetProperty("intermediates").EnumerateObject()
            .ToDictionary(p => p.Name, p => Shape(p.Value), StringComparer.Ordinal);
        Assert.Equal(new[] { "down0", "down1", "down2", "down3", "middle", "timeEmbedding", "up0", "up1", "up2" },
            shapes.Keys.Order(StringComparer.Ordinal));
        var captures = shapes.ToDictionary(p => p.Key, p => new HostCapture(p.Value), StringComparer.Ordinal);
        var inputs = new Dictionary<string, Tensor>(StringComparer.Ordinal)
            { ["latent"] = latent, ["timesteps"] = timesteps, ["context"] = context };
        try
        {
            Run(directory, reference.GetProperty("id").GetString()!, "unet", bank, parameterReferences, config,
                new { }, inputs, reference.GetProperty("output"), captures,
                repeat =>
                {
                    model.DiagnosticObserver = repeat == 1 ? (name, tensor) => captures[name].Capture(tensor) : null;
                    return model.Forward(latent, timesteps, context);
                }, new[] { "off", "on", "off" });
        }
        finally { model.DiagnosticObserver = null; }
        return true;
    }

    internal static bool Guidance(SdDenoiser denoiser, UnetWeightSet bank, Tensor latent, Tensor sigma,
        Tensor positive, Tensor negative, SdGuidanceOptions options, string policy, JsonElement reference,
        JsonElement parameterReferences, JsonElement config)
    {
        string? directory = Environment.GetEnvironmentVariable(Variable);
        if (string.IsNullOrWhiteSpace(directory)) return false;
        ValidateOptions();
        var inputs = new Dictionary<string, Tensor>(StringComparer.Ordinal)
            { ["latent"] = latent, ["sigma"] = sigma, ["positive"] = positive, ["negative"] = negative };
        bool eligible = SdDenoiser.TryCommonContextLength(positive.shape[1], negative.shape[1], out long common);
        Run(directory, reference.GetProperty("id").GetString()! + "/" + policy, "guidance", bank, parameterReferences,
            config, new { policy, scale = options.Scale, predictionKind = "epsilon", concatEligible = eligible, commonLength = common },
            inputs, reference.GetProperty("outputs").GetProperty(policy), new(StringComparer.Ordinal),
            _ => denoiser.DenoiseGuided(latent, sigma, positive, negative, options), new[] { "off", "off", "off" });
        return true;
    }

    private static void ValidateOptions()
    {
        Assert.True(BitConverter.IsLittleEndian);
        Assert.Equal(1, get_num_threads());
        Assert.Equal(1, get_num_interop_threads());
        foreach (string variable in new[] { "COMFYSHARP_SD_UNET_TRACE_DIR", "COMFYSHARP_SD_UNET_FINE_TRACE_DIR",
            "COMFYSHARP_SD_LATENT_OFFSET", "COMFYSHARP_SD_GUIDANCE_TRACE_DIR" })
            Assert.Null(Environment.GetEnvironmentVariable(variable));
    }

    private static void Run(string directory, string id, string kind, UnetWeightSet bank, JsonElement expectedParameters,
        JsonElement config, object options, Dictionary<string, Tensor> borrowed, JsonElement expectedOutput,
        Dictionary<string, HostCapture> captures, Func<int, Tensor> forward, string[] sequence)
    {
        var parameterInputs = Parameters(bank, expectedParameters);
        var inputs = borrowed.ToDictionary(p => p.Key, p => Snapshot(p.Value), StringComparer.Ordinal);
        var hashesBefore = borrowed.ToDictionary(p => p.Key, p => Hash(p.Value), StringComparer.Ordinal);
        var outputs = new List<Tensor>(3);
        var hashes = new string[3];
        // Metadata and managed buffers can affect allocation/timing. No added callback
        // allocates native tensors; compare off/on/off and the original/copy control.
        captures.Add("output", new HostCapture(Shape(expectedOutput)));
        try
        {
            for (int repeat = 0; repeat < 3; repeat++)
            {
                var output = forward(repeat);
                outputs.Add(output);
                hashes[repeat] = Hash(output);
                if (repeat == 1) captures["output"].Capture(output);
            }
            var tensors = captures.ToDictionary(p => p.Key, p => p.Value.Record(), StringComparer.Ordinal);
            var hashesAfter = borrowed.ToDictionary(p => p.Key, p => Hash(p.Value), StringComparer.Ordinal);
            var parametersAfter = Parameters(bank, expectedParameters);
            bool unchanged = hashesBefore.All(p => hashesAfter[p.Key] == p.Value);
            bool identical = hashes.Distinct(StringComparer.Ordinal).Count() == 1;
            var trace = new
            {
                id, kind, synthetic = true, config, options,
                parameters = parameterInputs.Parameters, parameterLayouts = parameterInputs.Layouts,
                parametersAfter = parametersAfter.Parameters, parameterLayoutsAfter = parametersAfter.Layouts,
                inputs, inputHashesAfter = hashesAfter, inputsUnchanged = unchanged, tensors,
                observer = new { sequence, outputHashes = hashes, bitIdentical = identical,
                    scope = kind == "unet" ? "coarse callbacks enabled only in second forward" : "no model observer; output capture after forward" },
                native = NativeIdentity()
            };
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, id.Replace("/", "--", StringComparison.Ordinal) + ".json");
            File.WriteAllText(path + ".tmp", JsonSerializer.Serialize(trace));
            File.Move(path + ".tmp", path, overwrite: false);
            Assert.True(unchanged, "Diagnostic forward changed a borrowed input.");
            Assert.True(identical, "Diagnostic repetitions or observation changed the output.");
            // The same three existing committed-reference assertions remain required.
            foreach (var output in outputs) SdSamplingReferenceTests.Compare(output, expectedOutput);
        }
        finally { foreach (var output in outputs) output.Dispose(); }
    }

    private sealed record ParameterEvidence(List<object> Parameters, Dictionary<string, object> Layouts);

    private static ParameterEvidence Parameters(UnetWeightSet bank, JsonElement references)
    {
        var result = new List<object>(686);
        var layouts = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var item in references.EnumerateArray())
        {
            string name = item.GetProperty("name").GetString()!;
            var value = bank.GetTensor(name);
            string hash = Hash(value);
            Assert.Equal(item.GetProperty("sha256").GetString(), hash);
            Assert.Equal(Shape(item), value.shape);
            result.Add(new { name, shape = value.shape, sha256 = hash });
            layouts.Add(name, new { shape = value.shape, stride = value.stride(), aligned64 = CpuModelWeightBank.IsAligned(value) });
        }
        Assert.Equal(686, result.Count);
        return new(result, layouts);
    }

    private static long[] Shape(JsonElement record) => record.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray();

    private static object Snapshot(Tensor tensor)
    {
        var capture = new HostCapture(tensor.shape);
        capture.Capture(tensor);
        return capture.Record();
    }

    private static string Hash(Tensor tensor)
    {
        Assert.Equal(ScalarType.Float32, tensor.dtype);
        Assert.Equal(DeviceType.CPU, tensor.device_type);
        Assert.True(tensor.is_contiguous(), "Diagnostic refuses unexpected strided storage instead of changing its layout.");
        return Convert.ToHexStringLower(SHA256.HashData(tensor.bytes));
    }

    private sealed class HostCapture(long[] expectedShape)
    {
        private readonly byte[] bytes = new byte[checked((int)expectedShape.Aggregate(4L, (n, d) => checked(n * d)))];
        private long[]? stride;
        private bool aligned64;
        private bool seen;
        internal void Capture(Tensor value)
        {
            Assert.False(seen);
            Assert.Equal(ScalarType.Float32, value.dtype);
            Assert.Equal(DeviceType.CPU, value.device_type);
            Assert.True(value.is_contiguous(), "Coarse diagnostic requires contiguous storage; it never calls contiguous in a hook.");
            Assert.Equal(expectedShape, value.shape);
            value.bytes.CopyTo(bytes);
            stride = value.stride();
            aligned64 = CpuModelWeightBank.IsAligned(value);
            seen = true;
        }
        internal object Record()
        {
            Assert.True(seen, "Required diagnostic boundary missing.");
            return new { shape = expectedShape, stride, aligned64, dtype = "float32",
                sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)), values = MemoryMarshal.Cast<byte, float>(bytes).ToArray() };
        }
    }

    private static object NativeIdentity()
    {
        using var process = Process.GetCurrentProcess();
        string managedPath = Path.GetFullPath(typeof(Tensor).Assembly.Location);
        var pathComparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        // Enumerate afresh after every completed case. Never collapse different loaded paths
        // merely because their basename or SHA is equal. Absolute paths stay private.
        var libraries = process.Modules.Cast<ProcessModule>().Select(m => m.FileName).Distinct(StringComparer.Ordinal)
            // Exclude only the actual loaded managed assembly path. A different image
            // with the same basename remains visible to the native-origin gate.
            .Where(p => !string.Equals(Path.GetFullPath(p), managedPath, pathComparison))
            .Where(p => new[] { "torch", "c10", "gomp", "iomp", "libomp" }.Any(s => Path.GetFileName(p).Contains(s, StringComparison.OrdinalIgnoreCase)))
            .Order(StringComparer.Ordinal).Select(p =>
            {
                using var stream = File.OpenRead(p);
                return new { name = Path.GetFileName(p), bytes = stream.Length, sha256 = Convert.ToHexStringLower(SHA256.HashData(stream)) };
            }).ToArray();
        using var managedStream = File.OpenRead(managedPath);
        var managedAssemblies = new[] { new { name = Path.GetFileName(managedPath), bytes = managedStream.Length,
            sha256 = Convert.ToHexStringLower(SHA256.HashData(managedStream)) } };
        return new { torchSharp = typeof(Tensor).Assembly.GetName().Version?.ToString(), declaredLibtorchPackage = "2.10.0",
            requestedCapability = Environment.GetEnvironmentVariable("ATEN_CPU_CAPABILITY") ?? "auto",
            threads = get_num_threads(), interopThreads = get_num_interop_threads(), os = RuntimeInformation.OSDescription,
            architecture = RuntimeInformation.ProcessArchitecture.ToString(), dotnet = RuntimeInformation.FrameworkDescription,
            operationBoundary = "SdUnet.Forward / SdDenoiser.DenoiseGuided; existing CPU Float32 implementation", managedAssemblies, libraries };
    }
}
