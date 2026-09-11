using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using ComfySharp.Tokenization;
using TorchSharp;

namespace ComfySharp.RuntimeProbe;

/// <summary>Explicit local checkpoint inspection and CPU/F32 conditioning diagnostic.</summary>
public static class ClipDiagnostic
{
    private const string Usage = "clip --weights <safetensors> --layout <canonical|clip-l|clip-g|sd1|sdxl-l|sdxl-g|openclip> --profile <sd1-l|sdxl-l|sdxl-g> [--text <prompt>] [--inspect] [--sha256] [--unprojected-pooled] [--repeat 1..100] [--synthetic-config <H,M,N,heads,quick_gelu|gelu|gelu_pytorch_tanh>]";

    public static int Run(string[] args, TextWriter output, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        string stage = "arguments";
        Options? options = null;
        bool nativeInitialized = false;
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (args is ["--help"] or ["-h"])
            {
                Write(output, new { diagnostic = "clip", status = "ok", operation = "help", usage = Usage,
                    modelCompatibility = "not_assessed", nativeInitialized = false,
                    notes = "Inspection needs no text or native initialization. Synthetic configuration changes dimensions only and never creates weights. No checkpoint path, prompt or token IDs are included in diagnostic output." });
                return 0;
            }
            options = Parse(args);
            stage = "open";
            using var file = new SafeTensorFile(options.Weights);
            cancellationToken.ThrowIfCancellationRequested();
            stage = "inspect";
            bool projected = options.Profile != ClipProfile.Sd1L && !options.UnprojectedPooled;
            var plan = ClipCheckpointLoader.Inspect(file, options.Config, options.Layout, requireProjection: projected);
            cancellationToken.ThrowIfCancellationRequested();
            stage = "hash";
            string? fileSha256 = options.Sha256 ? file.ComputeSha256(cancellationToken) : null;
            var report = new Dictionary<string, object?>
            {
                ["diagnostic"] = "clip", ["status"] = "ok", ["operation"] = options.Inspect ? "inspect" : "encode",
                ["modelCompatibility"] = "not_assessed", ["synthetic"] = options.Synthetic,
                ["profile"] = options.ProfileName, ["layout"] = options.LayoutName,
                ["pooledKind"] = projected ? "projected" : "unprojected",
                ["config"] = new { hiddenSize = options.Config.HiddenSize, intermediateSize = options.Config.IntermediateSize,
                    layerCount = options.Config.LayerCount, headCount = options.Config.HeadCount,
                    activation = ActivationName(options.Config.Activation), vocabularySize = ClipTextConfig.VocabularySize,
                    maxPositions = ClipTextConfig.MaxPositions },
                ["device"] = "cpu", ["computeDtype"] = "Float32", ["attentionBackend"] = "attention_basic",
                ["fileBytes"] = file.FileSizeBytes,
                ["weightPlan"] = new { tensorCount = plan.Mappings.Count, hasProjection = plan.HasProjection,
                    requireProjection = plan.RequireProjection, residentBytes = plan.ResidentBytes,
                    temporaryBytes = plan.TemporaryBytes, estimatedPeakWeightBytes = plan.EstimatedPeakWeightBytes,
                    memoryEstimateScope = "weights_only_excludes_activations_allocator_and_runtime",
                    mappings = plan.Mappings.Select(mapping => new { canonicalName = mapping.CanonicalName,
                        sourceShape = mapping.SourceShape, canonicalShape = mapping.CanonicalShape,
                        sourceDtype = mapping.SourceDType, transform = mapping.Transform.ToString(), sliceStart = mapping.SliceStart }).ToArray(),
                    ignoredEncoderTensorCount = plan.IgnoredEncoderKeys.Count }
            };
            // Do not include arbitrary source keys/metadata; they can contain private strings or paths.
            if (fileSha256 is not null) report["fileSha256"] = fileSha256;
            if (options.Inspect)
            {
                report["nativeInitialized"] = false;
                cancellationToken.ThrowIfCancellationRequested();
                Write(output, report);
                return 0;
            }

            stage = "tokenize";
            var tokenizer = new ComfyClipTokenizer(ClipTokenizer.CreateDefault(), options.Profile);
            var tokens = tokenizer.Tokenize(options.Text!, cancellationToken: cancellationToken);
            stage = "load";
            NativeRuntimeBootstrap.Initialize();
            nativeInitialized = true;
            // The diagnostic is a standalone process: pin the numerical profile before materialization.
            torch.set_num_threads(1);
            var loadWatch = Stopwatch.StartNew();
            using var bank = ClipCheckpointLoader.Load(file, plan, cancellationToken);
            loadWatch.Stop();
            using var graph = new ClipTextEncoder(bank);
            using var encoder = new ComfyClipEncoder(graph, options.Profile);
            var executionOptions = new ClipConditioningOptions { ProjectPooled = projected };
            var measurements = new List<object>(options.Repeats);
            var afterLoad = Memory();
            string? firstHiddenHash = null, firstPooledHash = null;
            bool hashesRepeat = true;
            stage = "encode";
            for (int i = 0; i < options.Repeats; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var watch = Stopwatch.StartNew();
                using var result = encoder.Encode(tokens, executionOptions, cancellationToken);
                watch.Stop();
                var hidden = Describe(result.Hidden, cancellationToken);
                var pooled = Describe(result.Pooled, cancellationToken);
                firstHiddenHash ??= hidden.Sha256;
                firstPooledHash ??= pooled.Sha256;
                hashesRepeat &= firstHiddenHash == hidden.Sha256 && firstPooledHash == pooled.Sha256;
                result.Dispose();
                measurements.Add(new { iteration = i + 1, encodeMilliseconds = watch.Elapsed.TotalMilliseconds,
                    hidden, pooled, memoryAfterOutputDispose = Memory() });
            }
            report["nativeInitialized"] = true;
            report["torchSharp"] = typeof(torch.Tensor).Assembly.GetName().Version?.ToString();
            report["libtorchPackage"] = "2.10.0";
            report["threads"] = 1;
            report["loadMilliseconds"] = loadWatch.Elapsed.TotalMilliseconds;
            report["memoryAfterLoad"] = afterLoad;
            report["repeats"] = options.Repeats;
            report["outputHashesRepeat"] = hashesRepeat;
            report["pooledUse"] = options.Profile == ClipProfile.SdXlL ? "discarded_in_sdxl_composition" : "conditioning";
            report["executions"] = measurements;
            report["memoryQualification"] = "Process observations after explicit tensor disposal; allocator caching and process noise prevent a no-leak conclusion.";
            cancellationToken.ThrowIfCancellationRequested();
            Write(output, report);
            return 0;
        }
        catch (OperationCanceledException)
        {
            Write(output, new { diagnostic = "clip", status = "cancelled", stage,
                modelCompatibility = "not_assessed", synthetic = options?.Synthetic ?? false, nativeInitialized });
            return 130;
        }
        catch (Exception exception)
        {
            int code = exception is DiagnosticUsageException or ArgumentException or IOException or InvalidDataException or UnauthorizedAccessException or NotSupportedException ? 2 : 1;
            // Raw exception messages/stacks may embed private paths, text, metadata or native environment paths.
            string message = exception is DiagnosticUsageException usage ? usage.Message : stage switch
            {
                "open" => "Unable to open a valid safetensors file at the supplied location.",
                "inspect" => "Checkpoint metadata does not satisfy the selected CLIP layout, configuration or projection requirement.",
                "hash" => "Unable to hash the selected checkpoint.",
                "tokenize" => "Prompt tokenization failed for the selected profile.",
                "load" => "Checkpoint tensor materialization or CPU native runtime initialization failed.",
                "encode" => "CLIP encoding or output inspection failed.",
                _ => "Invalid diagnostic arguments. Use clip --help for supported options."
            };
            Write(output, new { diagnostic = "clip", status = "error", stage, error = exception.GetType().Name,
                message, modelCompatibility = "not_assessed", synthetic = options?.Synthetic ?? false, nativeInitialized });
            return code;
        }
    }

    private static Options Parse(string[] args)
    {
        string? weights = null, text = null, layoutName = null, profileName = null, syntheticConfig = null;
        bool inspect = false, sha256 = false, unprojectedPooled = false;
        int repeats = 1;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            if (!seen.Add(args[i])) throw new DiagnosticUsageException("Repeated diagnostic argument.");
            switch (args[i])
            {
                case "--weights": weights = Value(args, ref i); break;
                case "--text": text = Value(args, ref i); break;
                case "--layout": layoutName = Value(args, ref i); break;
                case "--profile": profileName = Value(args, ref i); break;
                case "--synthetic-config": syntheticConfig = Value(args, ref i); break;
                case "--inspect": inspect = true; break;
                case "--sha256": sha256 = true; break;
                case "--unprojected-pooled": unprojectedPooled = true; break;
                case "--repeat":
                    if (!int.TryParse(Value(args, ref i), NumberStyles.None, CultureInfo.InvariantCulture, out repeats) || repeats is < 1 or > 100)
                        throw new DiagnosticUsageException("--repeat must be an integer from 1 through 100.");
                    break;
                default: throw new DiagnosticUsageException("Unknown diagnostic argument. Use clip --help for supported options.");
            }
        }
        if (string.IsNullOrWhiteSpace(weights)) throw new DiagnosticUsageException("--weights must select a local safetensors file.");
        if (!inspect && text is null) throw new DiagnosticUsageException("--text is required for encoding; inspection may omit it.");
        var layout = layoutName switch
        {
            "canonical" => ClipCheckpointLayout.Canonical, "clip-l" => ClipCheckpointLayout.ClipL,
            "clip-g" => ClipCheckpointLayout.ClipG, "sd1" => ClipCheckpointLayout.Sd1,
            "sdxl-l" => ClipCheckpointLayout.SdxlL, "sdxl-g" => ClipCheckpointLayout.SdxlG,
            "openclip" => ClipCheckpointLayout.OpenClip,
            _ => throw new DiagnosticUsageException("--layout must explicitly select a supported CLIP checkpoint layout.")
        };
        var profile = profileName switch
        {
            "sd1-l" => ClipProfile.Sd1L, "sdxl-l" => ClipProfile.SdXlL, "sdxl-g" => ClipProfile.SdXlG,
            _ => throw new DiagnosticUsageException("--profile must explicitly select sd1-l, sdxl-l or sdxl-g.")
        };
        bool g = profile == ClipProfile.SdXlG;
        if (g && unprojectedPooled)
            throw new DiagnosticUsageException("The SDXL-G diagnostic profile requires projected pooled output.");
        if ((g && layout is ClipCheckpointLayout.ClipL or ClipCheckpointLayout.Sd1 or ClipCheckpointLayout.SdxlL)
            || (!g && layout is ClipCheckpointLayout.ClipG or ClipCheckpointLayout.SdxlG))
            throw new DiagnosticUsageException("The selected L/G checkpoint namespace conflicts with the selected profile.");
        var config = g ? ClipTextConfig.Giant : ClipTextConfig.Large;
        if (syntheticConfig is not null)
        {
            var fields = syntheticConfig.Split(',');
            if (fields.Length != 5 || !int.TryParse(fields[0], NumberStyles.None, CultureInfo.InvariantCulture, out int h)
                || !int.TryParse(fields[1], NumberStyles.None, CultureInfo.InvariantCulture, out int m)
                || !int.TryParse(fields[2], NumberStyles.None, CultureInfo.InvariantCulture, out int n)
                || !int.TryParse(fields[3], NumberStyles.None, CultureInfo.InvariantCulture, out int heads))
                throw new DiagnosticUsageException("--synthetic-config requires H,M,N,heads,activation.");
            var activation = fields[4] switch
            {
                "quick_gelu" => ClipActivation.QuickGelu, "gelu" => ClipActivation.Gelu,
                "gelu_pytorch_tanh" => ClipActivation.GeluTanh,
                _ => throw new DiagnosticUsageException("Synthetic activation must be quick_gelu, gelu or gelu_pytorch_tanh.")
            };
            config = new(h, m, n, heads, activation);
            try { config.Validate(); }
            catch (ArgumentException) { throw new DiagnosticUsageException("Synthetic dimensions must be positive and hidden width divisible by head count."); }
            if (profile != ClipProfile.Sd1L && n < 2)
                throw new DiagnosticUsageException("SDXL profiles require at least two layers for penultimate hidden selection.");
        }
        return new(weights, text, layoutName!, profileName!, layout, profile, config, inspect, sha256, repeats, syntheticConfig is not null, unprojectedPooled);
    }

    private static string Value(string[] args, ref int index)
    {
        if (++index == args.Length) throw new DiagnosticUsageException("Missing diagnostic argument value.");
        return args[index];
    }

    private static string ActivationName(ClipActivation activation) => activation switch
    {
        ClipActivation.QuickGelu => "quick_gelu", ClipActivation.Gelu => "gelu", _ => "gelu_pytorch_tanh"
    };

    private static TensorReport Describe(torch.Tensor value, CancellationToken cancellationToken)
    {
        using var scope = torch.NewDisposeScope();
        var contiguous = value.contiguous();
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var bytes = contiguous.bytes;
        const int chunkLength = 1024 * 1024;
        for (int offset = 0; offset < bytes.Length; offset += Math.Min(chunkLength, bytes.Length - offset))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var chunk = bytes.Slice(offset, Math.Min(chunkLength, bytes.Length - offset));
            if (BitConverter.IsLittleEndian) hash.AppendData(chunk);
            else
            {
                var littleEndian = chunk.ToArray();
                for (int i = 0; i < littleEndian.Length; i += 4) Array.Reverse(littleEndian, i, 4);
                hash.AppendData(littleEndian);
            }
        }
        var sample = contiguous.flatten().narrow(0, 0, Math.Min(8, contiguous.numel())).data<float>().ToArray()
            .Select(value => float.IsFinite(value) ? (object)value : float.IsNaN(value) ? "NaN" : value > 0 ? "Infinity" : "-Infinity").ToArray();
        return new(value.shape, value.dtype.ToString(), "little_endian_float32", Convert.ToHexStringLower(hash.GetHashAndReset()), sample);
    }

    private static object Memory()
    {
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        return new { privateBytes = process.PrivateMemorySize64, workingSetBytes = process.WorkingSet64,
            peakWorkingSetBytes = process.PeakWorkingSet64 };
    }

    private static void Write(TextWriter output, object report) => output.WriteLine(JsonSerializer.Serialize(report,
        new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));

    private sealed record Options(string Weights, string? Text, string LayoutName, string ProfileName,
        ClipCheckpointLayout Layout, ClipProfile Profile, ClipTextConfig Config, bool Inspect, bool Sha256, int Repeats, bool Synthetic, bool UnprojectedPooled);
    private sealed record TensorReport(long[] Shape, string Dtype, string HashEncoding, string Sha256, object[] Sample);
    private sealed class DiagnosticUsageException(string message) : ArgumentException(message);
}
