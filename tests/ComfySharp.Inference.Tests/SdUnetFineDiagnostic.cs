using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Opt-in synthetic primitive capture, called by the existing square reference case.</summary>
internal static class SdUnetFineDiagnostic
{
    internal static void Run(string directory, SdUnet model, UnetWeightSet bank, Tensor latent,
        Tensor timesteps, Tensor context, JsonElement reference, JsonElement parameterReferences)
    {
        Assert.True(BitConverter.IsLittleEndian, "This diagnostic writes little-endian Float32 payloads.");
        try
        {
            Assert.Equal(1, get_num_threads());
            Assert.Equal(1, get_num_interop_threads());
            Assert.Equal("sd15-reduced/square", reference.GetProperty("id").GetString());
            Assert.Null(Environment.GetEnvironmentVariable("COMFYSHARP_SD_LATENT_OFFSET"));
            var schema = UnetWeightSchema.Describe(bank.Config);
            var sourceParameters = parameterReferences.EnumerateArray().ToArray();
            Assert.Equal(686, schema.Count);
            Assert.Equal(686, sourceParameters.Length);
            Assert.Equal(schema.Keys.Order(StringComparer.Ordinal),
                sourceParameters.Select(item => item.GetProperty("name").GetString()!).Order(StringComparer.Ordinal));
            var parameters = new List<object>(686);
            var parameterLayouts = new Dictionary<string, object>(StringComparer.Ordinal);
            foreach (var expected in sourceParameters)
            {
                string name = expected.GetProperty("name").GetString()!;
                var tensor = bank.GetTensor(name);
                long[] shape = expected.GetProperty("shape").EnumerateArray().Select(item => item.GetInt64()).ToArray();
                Assert.Equal(shape, schema[name]);
                Assert.Equal(shape, tensor.shape);
                string hash = Hash(tensor);
                Assert.Equal(expected.GetProperty("sha256").GetString(), hash);
                parameters.Add(new { name, shape, sha256 = hash });
                parameterLayouts.Add(name, new { shape, stride = tensor.stride(), aligned64 = CpuModelWeightBank.IsAligned(tensor) });
            }
            Assert.Equal(new long[] { 1, 4, 8, 8 }, latent.shape);
            Assert.Equal(new long[] { 1 }, timesteps.shape);
            Assert.Equal(new long[] { 1, 3, 16 }, context.shape);
            var inputs = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["latent"] = Snapshot(latent), ["timesteps"] = Snapshot(timesteps), ["context"] = Snapshot(context)
            };
            // Allocate host capture buffers before any forward. Callbacks only copy existing
            // contiguous CPU spans and metadata; serialization and hashing happen afterwards.
            var captures = AllocateCaptures();
            var outputHashes = new string[3];
            for (int repetition = 0; repetition < 3; repetition++)
            {
                if (repetition == 1)
                {
                    model.FineDiagnosticObserver = (name, tensor) => captures[name].Capture(tensor);
                    model.DiagnosticObserver = (name, tensor) =>
                    {
                        if (name is "timeEmbedding" or "down2") captures[name].Capture(tensor);
                    };
                }
                else
                {
                    model.FineDiagnosticObserver = null;
                    model.DiagnosticObserver = null;
                }
                using var actual = model.Forward(latent, timesteps, context);
                outputHashes[repetition] = Hash(actual);
                if (repetition == 1) captures["output"].Capture(actual);
                if (repetition != 2) continue;

                var tensors = captures.ToDictionary(item => item.Key, item => item.Value.Record(), StringComparer.Ordinal);
                bool identical = outputHashes.Distinct(StringComparer.Ordinal).Count() == 1;
                var trace = new
                {
                    id = "sd15-reduced/square", synthetic = true, modelCompatibility = "not_assessed",
                    parameters, parameterLayouts, inputs, tensors,
                    observer = new { sequence = new[] { "off", "on", "off" }, outputHashes, bitIdentical = identical },
                    native = new
                    {
                        torchSharp = typeof(Tensor).Assembly.GetName().Version?.ToString(),
                        declaredLibtorchPackage = "2.10.0", requestedCapability = Environment.GetEnvironmentVariable("ATEN_CPU_CAPABILITY") ?? "auto",
                        threads = get_num_threads(), interopThreads = get_num_interop_threads(),
                        os = RuntimeInformation.OSDescription, architecture = RuntimeInformation.ProcessArchitecture.ToString(),
                        dotnet = RuntimeInformation.FrameworkDescription, libraries = NativeLibraries()
                    }
                };
                Directory.CreateDirectory(directory);
                string path = Path.Combine(directory, "sd15-reduced--square.json");
                string temporary = path + ".tmp";
                File.WriteAllText(temporary, JsonSerializer.Serialize(trace));
                File.Move(temporary, path, overwrite: false);
                Console.WriteLine("Synthetic U-Net primitive diagnostic: sd15-reduced--square.json");
                Assert.True(identical, "Enabling the fine observer changed the actual forward output.");
                // Preserve the committed-reference assertion after diagnostic evidence is safe.
                SdSamplingReferenceTests.Compare(actual, reference.GetProperty("output"));
            }
        }
        finally
        {
            model.FineDiagnosticObserver = null;
            model.DiagnosticObserver = null;
        }
    }

    private static Dictionary<string, HostCapture> AllocateCaptures()
    {
        var result = new Dictionary<string, HostCapture>(StringComparer.Ordinal)
        {
            ["timeEmbedding"] = new([1, 128]), ["down2"] = new([1, 128, 2, 2]),
            ["input_blocks.9.0.op.input"] = new([1, 128, 2, 2]),
            ["input_blocks.9.0.op.output"] = new([1, 128, 1, 1]), ["output"] = new([1, 4, 8, 8])
        };
        foreach (int block in new[] { 10, 11 })
        {
            string prefix = $"input_blocks.{block}.0";
            result.Add(prefix + ".input", new([1, 128, 1, 1]));
            result.Add(prefix + ".embedding", new([1, 128]));
            result.Add(prefix + ".output", new([1, 128, 1, 1]));
            foreach (string module in new[] { "in_layers.0", "in_layers.1", "in_layers.2", "emb_layers.0", "emb_layers.1", "out_layers.0", "out_layers.1", "out_layers.3" })
            {
                long[] shape = module.StartsWith("emb_", StringComparison.Ordinal) ? [1, 128] : [1, 128, 1, 1];
                result.Add(prefix + "." + module + ".input", new(shape));
                result.Add(prefix + "." + module + ".output", new(shape));
            }
        }
        Assert.Equal(43, result.Count);
        return result;
    }

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
        Assert.True(tensor.is_contiguous());
        return Convert.ToHexStringLower(SHA256.HashData(tensor.bytes));
    }

    private sealed class HostCapture(long[] expectedShape)
    {
        private readonly byte[] bytes = new byte[checked((int)expectedShape.Aggregate(4L, (size, dimension) => checked(size * dimension)))];
        private long[]? stride;
        private bool aligned64;
        private bool seen;

        internal void Capture(Tensor tensor)
        {
            Assert.False(seen, "A fine diagnostic stage was invoked more than once.");
            Assert.Equal(ScalarType.Float32, tensor.dtype);
            Assert.Equal(DeviceType.CPU, tensor.device_type);
            Assert.True(tensor.is_contiguous(), "Fine capture refuses unexpected strided tensors instead of making a native copy.");
            Assert.Equal(expectedShape, tensor.shape);
            var source = tensor.bytes;
            Assert.Equal(bytes.Length, source.Length);
            source.CopyTo(bytes);
            stride = tensor.stride();
            aligned64 = CpuModelWeightBank.IsAligned(tensor);
            seen = true;
        }

        internal object Record()
        {
            Assert.True(seen, "A required fine diagnostic stage was not captured.");
            return new { shape = expectedShape, stride, aligned64, dtype = "float32",
                sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)), values = MemoryMarshal.Cast<byte, float>(bytes).ToArray() };
        }
    }

    private static object[] NativeLibraries()
    {
        using var process = Process.GetCurrentProcess();
        return process.Modules.Cast<ProcessModule>()
            .Where(module => module.ModuleName.Contains("torch", StringComparison.OrdinalIgnoreCase)
                || module.ModuleName.Contains("c10", StringComparison.OrdinalIgnoreCase)
                || module.ModuleName.Contains("gomp", StringComparison.OrdinalIgnoreCase)
                || module.ModuleName.Contains("iomp", StringComparison.OrdinalIgnoreCase))
            .Select(module => module.FileName).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal)
            .Select(path =>
            {
                using var stream = File.OpenRead(path);
                return (object)new { name = Path.GetFileName(path), bytes = stream.Length,
                    sha256 = Convert.ToHexStringLower(SHA256.HashData(stream)) };
            }).ToArray();
    }
}
