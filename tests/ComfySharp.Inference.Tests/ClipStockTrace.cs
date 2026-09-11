using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using TorchSharp;

namespace ComfySharp.Inference.Tests;

/// <summary>Opt-in observations of the explicit synthetic stock bank; never an acceptance oracle.</summary>
internal static class ClipStockTrace
{
    internal static bool CaptureIfRequested(string id, ClipWeightSet weights, ClipTextEncoder encoder,
        IReadOnlyList<IReadOnlyList<int>> tokens, IReadOnlyList<int> tokenCounts)
    {
        string? requestedRoot = Environment.GetEnvironmentVariable("COMFYSHARP_CLIP_TRACE_DIR");
        if (string.IsNullOrWhiteSpace(requestedRoot)) return false;
        string variant = id switch { "stock/l" => "l", "stock/g" => "g", _ => throw new ArgumentException("Unknown synthetic stock trace profile.", nameof(id)) };
        string directory = Path.Combine(Path.GetFullPath(requestedRoot), variant);
        // Unique payload names keep a previously published manifest consistent if a later capture fails.
        string runName = "capture-" + Guid.NewGuid().ToString("N");
        string runDirectory = Path.Combine(directory, runName);
        Directory.CreateDirectory(runDirectory);
        var config = weights.Config;
        var tensors = new Dictionary<string, object>(StringComparer.Ordinal);
        var parameters = new Dictionary<string, string>(StringComparer.Ordinal);
        var parameterShapes = ClipWeightSchema.Describe(config);

        NativeRuntimeBootstrap.Initialize();
        using var scope = torch.NewDisposeScope();
        using var noGrad = torch.no_grad();
        foreach (string name in parameterShapes.Keys)
            parameters.Add(name, HashAndWrite(weights.GetTensor(name), null));

        // These tensors come from the production graph. All-layer selection is an extra observation;
        // the original scalar-layer reference assertions and three repeated forwards still run below it.
        using (var result = encoder.Forward(tokens, new()
        {
            AllIntermediateLayers = true, NormalizeIntermediate = false, TokenCounts = tokenCounts
        }))
        {
            Save("final", result.FinalHidden);
            Save("all", result.IntermediateHidden!);
            Save("pooled", result.Pooled);
            Save("projected", result.ProjectedPooled!);
        }

        // Reconstruct only the first operator boundary from the SAME borrowed bank. These are
        // observational diagnostics, not values used by the production forward or by acceptance.
        int batch = tokens.Count;
        var ids = torch.tensor(tokens.SelectMany(row => row).Select(value => (long)value).ToArray(),
            dtype: torch.ScalarType.Int64, device: torch.CPU);
        var embedding = weights.GetTensor("text_model.embeddings.token_embedding.weight").index_select(0, ids)
            .reshape(batch, ClipTextConfig.MaxPositions, config.HiddenSize)
            + weights.GetTensor("text_model.embeddings.position_embedding.weight");
        Save("embeddings.output", embedding);
        var normalized = torch.nn.functional.layer_norm(embedding, new long[] { config.HiddenSize },
            weights.GetTensor("text_model.encoder.layers.0.layer_norm1.weight"),
            weights.GetTensor("text_model.encoder.layers.0.layer_norm1.bias"), eps: 1e-5);
        Save("layer.0.norm1", normalized);
        foreach (string projection in new[] { "q", "k", "v" })
        {
            using var operatorScope = torch.NewDisposeScope();
            string prefix = $"text_model.encoder.layers.0.self_attn.{projection}_proj";
            var projected = torch.nn.functional.linear(normalized.reshape(batch * ClipTextConfig.MaxPositions, config.HiddenSize),
                weights.GetTensor(prefix + ".weight"), weights.GetTensor(prefix + ".bias"))
                .reshape(batch, ClipTextConfig.MaxPositions, config.HiddenSize);
            Save($"layer.0.{projection}", projected);
        }

        var manifest = new Dictionary<string, object>
        {
            ["schemaVersion"] = 1, ["backendCommit"] = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a",
            ["profile"] = variant, ["synthetic"] = true, ["modelWeightsUsed"] = false,
            ["observationalOnly"] = true, ["config"] = config,
            ["tokens"] = tokens, ["tokenCounts"] = tokenCounts,
            ["parameters"] = parameters, ["parameterShapes"] = parameterShapes, ["tensors"] = tensors,
            ["native"] = new
            {
                torchSharp = typeof(torch.Tensor).Assembly.GetName().Version?.ToString(),
                libtorchPackage = "2.10.0", threads = torch.get_num_threads(),
                operatingSystem = OperatingSystem.IsWindows() ? "windows" : OperatingSystem.IsLinux() ? "linux" : OperatingSystem.IsMacOS() ? "macos" : "other",
                osDescription = RuntimeInformation.OSDescription,
                processArchitecture = RuntimeInformation.ProcessArchitecture.ToString(),
                osArchitecture = RuntimeInformation.OSArchitecture.ToString(), dotNet = RuntimeInformation.FrameworkDescription
            },
            ["CPUfeatures.NET"] = new
            {
                avx2 = Avx2.IsSupported, avx512F = Avx512F.IsSupported, fma = Fma.IsSupported,
                advSimd = AdvSimd.IsSupported, qualification = "DotNet intrinsic availability only; not ATen dispatch capability."
            }
        };
        string temporaryManifest = Path.Combine(directory, runName + ".manifest.tmp");
        File.WriteAllText(temporaryManifest, JsonSerializer.Serialize(manifest,
            new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
        File.Move(temporaryManifest, Path.Combine(directory, "manifest.json"), overwrite: true);
        return true;

        void Save(string name, torch.Tensor value)
        {
            string fileName = name + ".f32";
            string digest;
            using (var file = File.Create(Path.Combine(runDirectory, fileName))) digest = HashAndWrite(value, file);
            tensors.Add(name, new { file = runName + "/" + fileName, shape = value.shape, dtype = "float32",
                byteOrder = "little", bytes = checked(value.numel() * 4), sha256 = digest });
        }
    }

    private static string HashAndWrite(torch.Tensor value, Stream? output)
    {
        using var scope = torch.NewDisposeScope();
        var contiguous = value.contiguous();
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var bytes = contiguous.bytes;
        const int chunkSize = 1024 * 1024;
        for (int offset = 0; offset < bytes.Length;)
        {
            int length = Math.Min(chunkSize, bytes.Length - offset);
            var chunk = bytes.Slice(offset, length);
            if (BitConverter.IsLittleEndian)
            {
                hash.AppendData(chunk);
                output?.Write(chunk);
            }
            else
            {
                byte[] littleEndian = chunk.ToArray();
                for (int i = 0; i < littleEndian.Length; i += 4) Array.Reverse(littleEndian, i, 4);
                hash.AppendData(littleEndian);
                output?.Write(littleEndian);
            }
            offset += length;
        }
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
}
