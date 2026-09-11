using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using ComfySharp.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

/// <summary>Explicit reduced synthetic composition. Neither a checkpoint loader nor a parity verdict.</summary>
internal static class Sd15PipelineDiagnostic
{
    private const long MiB = 1024 * 1024;
    private const long RuntimeAllowance = 512 * MiB;
    private const long GraphAllowance = 512 * MiB;
    private const long Headroom = 256 * MiB;
    private const string Usage = "sd15-pipeline --case empty-one-step|weighted-three-step|two-chunks-separate|maximum-start [--memory-budget-mib N] [--synthetic-reduced --execute --output <new-absolute-directory>]";
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    internal static readonly ClipTextConfig ClipConfig = new(16, 32, 2, 4, ClipActivation.QuickGelu);
    internal sealed record Options(Sd15PipelineCase Case, bool Execute, long? BudgetBytes, string? OutputDirectory);
    internal sealed class BudgetExceededException : Exception { }
    internal sealed record TensorRecord(string? File, long[] Shape, long[] Stride, long StorageOffset,
        bool Contiguous, long Bytes, string Sha256);

    internal static int Run(string[] args, TextWriter output, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        string stage = "arguments";
        bool nativeInitialized = false;
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (args is ["--help"] or ["-h"])
            {
                Write(output, new { diagnostic = "sd15-pipeline", status = "ok", operation = "help", usage = Usage,
                    nativeInitialized, modelCompatibility = "not_assessed" });
                return 0;
            }
            var options = Parse(args);
            stage = "plan";
            var clip = SdSyntheticWeightBuilder.Describe(ClipWeightSchema.Describe(ClipConfig));
            var unet = SdSyntheticWeightBuilder.DescribeUnet(Sd15PipelineExecution.UnetConfig);
            var vae = SdSyntheticWeightBuilder.Describe(ClassicalVaeWeightSchema.Describe(Sd15PipelineExecution.VaeConfig));
            long weightBytes = checked(clip.ResidentBytes + unet.ResidentBytes + vae.ResidentBytes);
            int weightCount = checked(clip.TensorCount + unet.TensorCount + vae.TensorCount);
            if (weightBytes != Sd15PipelineCases.ParameterBytes || weightCount != Sd15PipelineCases.ParameterCount ||
                SdSyntheticRecipe.Version != Sd15PipelineCases.ParameterRecipe)
                throw new InvalidDataException("Synthetic graph plans differ from the frozen protocol.");
            var conditioning = options.Case.TokenizeAndVerify(cancellationToken);
            long inputBytes = checked(320 + (options.Case.Steps + 1) * 4L +
                (conditioning.Positive.Chunks.Count + conditioning.Negative.Chunks.Count) * 77L * 16);
            long estimate = checked(weightBytes + inputBytes + RuntimeAllowance + GraphAllowance + Headroom);
            var report = new Dictionary<string, object?>
            {
                ["diagnostic"] = "sd15-pipeline", ["status"] = "ok", ["operation"] = options.Execute ? "execute" : "plan",
                ["configurationKind"] = "reduced_diagnostic", ["syntheticWeights"] = true,
                ["pretrainedWeightsUsed"] = false, ["modelCompatibility"] = "not_assessed",
                ["numericalQualification"] = "not_performed", ["protocolSha256"] = Sd15PipelineCases.ProtocolSha256,
                ["profile"] = Sd15PipelineCases.Profile, ["backendCommit"] = Sd15PipelineCases.BackendCommit,
                ["caseId"] = options.Case.Id, ["device"] = "cpu", ["dtype"] = "float32",
                ["repetitions"] = 3, ["boundaryObserverSequence"] = new[] { "off", "on", "off" },
                ["eulerStepCapture"] = "not_exposed_by_this_cli", ["parameterRecipe"] = SdSyntheticRecipe.Version,
                ["parameterCount"] = weightCount, ["parameterBytes"] = weightBytes,
                ["configurations"] = new { clip = ClipConfig, unet = Sd15PipelineExecution.UnetConfig,
                    vae = Sd15PipelineExecution.VaeConfig, clipProjectionPresent = true, projectionUsedForConditioning = false },
                ["sampling"] = new { prediction = "epsilon", guidance = "separate", options.Case.GuidanceScale,
                    options.Case.MaximumDenoise, options.Case.Steps, churn = 0, latentScale = SdSamplingMath.Sd15LatentScale },
                ["memoryPlan"] = new { weightBytes, inputBytes, runtimeAllowanceBytes = RuntimeAllowance,
                    graphAllowanceBytes = GraphAllowance, headroomBytes = Headroom, estimatedProcessBytes = estimate,
                    requestedBudgetBytes = options.BudgetBytes, estimateIsPeakGuarantee = false },
                ["nativeInitialized"] = false, ["weightsGenerated"] = false
            };
            if (options.BudgetBytes is long budget && estimate > budget)
            {
                report["status"] = "budget_exceeded";
                Write(output, report);
                return 3;
            }
            cancellationToken.ThrowIfCancellationRequested();
            if (!options.Execute) { Write(output, report); return 0; }
            stage = "initialize";
            NativeRuntimeBootstrap.Initialize();
            nativeInitialized = true;
            int previousThreads = get_num_threads();
            try
            {
                set_num_threads(1);
                // This command is dispatched in its own process before any other native operation.
                // Inter-op thread count cannot safely be changed back after native work has started.
                if (get_num_interop_threads() != 1) set_num_interop_threads(1);
                stage = "execute";
                Execute(options, conditioning, clip, unet, vae, report, cancellationToken);
            }
            finally { set_num_threads(previousThreads); }
            Write(output, report);
            return 0;
        }
        catch (OperationCanceledException)
        {
            Write(output, new { diagnostic = "sd15-pipeline", status = "cancelled", stage, nativeInitialized,
                modelCompatibility = "not_assessed" });
            return 130;
        }
        catch (Exception error)
        {
            int code = error is BudgetExceededException ? 3 : error is ArgumentException or IOException or
                UnauthorizedAccessException or OverflowException or NotSupportedException ? 2 : 1;
            Write(output, new { diagnostic = "sd15-pipeline", status = code == 3 ? "budget_exceeded" : "error",
                stage, nativeInitialized, modelCompatibility = "not_assessed", error = error.GetType().Name,
                message = stage == "arguments" ? "Invalid arguments. Use sd15-pipeline --help."
                    : code == 3 ? "The explicit memory budget is insufficient for this diagnostic."
                    : "The reduced synthetic pipeline did not complete. No model compatibility was assessed." });
            return code;
        }
    }

    internal static Options Parse(string[] args)
    {
        string? caseId = null, directory = null;
        bool execute = false, synthetic = false;
        long? budget = null;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            if (!seen.Add(args[i])) throw new ArgumentException("Duplicate option.");
            switch (args[i])
            {
                case "--case": caseId = Value(); break;
                case "--execute": execute = true; break;
                case "--synthetic-reduced": synthetic = true; break;
                case "--output": directory = Value(); break;
                case "--memory-budget-mib":
                    if (!long.TryParse(Value(), NumberStyles.None, CultureInfo.InvariantCulture, out long n) || n <= 0)
                        throw new ArgumentException("Expected a positive integer budget.");
                    budget = checked(n * MiB); break;
                default: throw new ArgumentException("Unknown option.");
            }
            string Value() => ++i < args.Length ? args[i] : throw new ArgumentException("Missing value.");
        }
        if (caseId is null || (execute && (!synthetic || budget is null || directory is null)) ||
            (!execute && (directory is not null || synthetic)))
            throw new ArgumentException("Execution requires an explicit synthetic opt-in, budget and output.");
        if (directory is not null)
        {
            if (!Path.IsPathFullyQualified(directory)) throw new ArgumentException("Output must be absolute.");
            directory = Path.GetFullPath(directory);
            if (File.Exists(directory) || Directory.Exists(directory)) throw new ArgumentException("Output must be new.");
        }
        return new(Sd15PipelineCases.Get(caseId), execute, budget, directory);
    }

    private static void Execute(Options options, Sd15PipelineConditioning conditioning,
        SyntheticWeightPlan clipPlan, SyntheticWeightPlan unetPlan, SyntheticWeightPlan vaePlan,
        Dictionary<string, object?> report, CancellationToken token)
    {
        long budget = options.BudgetBytes!.Value;
        long before = Memory();
        if (checked(before + Sd15PipelineCases.ParameterBytes + GraphAllowance + Headroom) > budget)
            throw new BudgetExceededException();
        string directory = options.OutputDirectory!;
        // CreateNew claims the evidence set even if another writer races directory creation.
        Directory.CreateDirectory(directory);
        using var claim = new FileStream(Path.Combine(directory, "execution.lock"), FileMode.CreateNew,
            FileAccess.Write, FileShare.None);
        report["nativeInitialized"] = true;
        report["workingSetBeforeGeneration"] = before;
        var borrowed = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        T Transfer<T>(string component, IReadOnlyDictionary<string, Tensor> values,
            Func<IReadOnlyDictionary<string, Tensor>, T> create) where T : IDisposable
        {
            // The synthetic builder requires aligned native allocations. These three factories
            // retain aligned wrappers without copying. Read-only snapshots stay inside bank lifetime.
            foreach (var (name, tensor) in values) borrowed.Add(component + "/" + name, tensor);
            return create(values);
        }
        using (var clipWeights = SdSyntheticWeightBuilder.Create(clipPlan, new(clipPlan.ResidentBytes),
            values => Transfer("clip", values, v => ClipWeightSet.FromOwnedTensors(ClipConfig, v)), token))
        using (var unetWeights = SdSyntheticWeightBuilder.Create(unetPlan, new(unetPlan.ResidentBytes),
            values => Transfer("unet", values, v => UnetWeightSet.FromOwnedTensors(Sd15PipelineExecution.UnetConfig, v)), token))
        using (var vaeWeights = SdSyntheticWeightBuilder.Create(vaePlan, new(vaePlan.ResidentBytes),
            values => Transfer("vae", values, v => ClassicalVaeWeightSet.FromOwnedTensors(Sd15PipelineExecution.VaeConfig, v)), token))
        {
            report["weightsGenerated"] = true;
            CheckBudget(budget);
            var parameterRecords = new Dictionary<string, TensorRecord>(StringComparer.Ordinal);
            foreach (var (name, tensor) in borrowed) parameterRecords.Add(name, Observe(tensor, null, null, token));
            CheckGenerated("clip", clipWeights.Parameters);
            CheckGenerated("unet", unetWeights.Parameters);
            CheckGenerated("vae", vaeWeights.Parameters);
            using (var clipGraph = new ClipTextEncoder(clipWeights.Weights))
            using (var clip = new ComfyClipEncoder(clipGraph, ClipProfile.Sd1L))
            using (var unet = new SdUnet(unetWeights.Weights))
            using (var vaeGraph = new ClassicalVae(vaeWeights.Weights))
            using (var vae = new ComfyImageVae(vaeGraph))
            using (var noise = Input(options.Case.NoiseValues(), [1, 4, 4, 5]))
            using (var sigmas = Input(options.Case.SigmaValues(), [options.Case.Steps + 1]))
            {
                var inputs = new Dictionary<string, TensorRecord>
                {
                    ["noise"] = Observe(noise, directory, "noise.f32", token),
                    ["sigmas"] = Observe(sigmas, directory, "sigmas.f32", token)
                };
                if (inputs["noise"].Sha256 != options.Case.NoiseSha256 || inputs["sigmas"].Sha256 != options.Case.SigmaSha256)
                    throw new InvalidDataException("Materialized input bytes differ from the protocol.");
                var captures = new Dictionary<string, TensorRecord>(StringComparer.Ordinal);
                var executions = new List<object>(3);
                string[]? expectedFinals = null;
                for (int repetition = 0; repetition < 3; repetition++)
                {
                    token.ThrowIfCancellationRequested();
                    CheckBudget(budget);
                    var timer = Stopwatch.StartNew();
                    using var result = Sd15PipelineExecution.Run(clip, unet, vae, conditioning, noise, sigmas, token,
                        repetition == 1 ? (boundary, value) => captures.Add(boundary.ToString(),
                            Observe(value, directory, boundary + ".f32", token)) : null);
                    timer.Stop();
                    var finals = new[] { Observe(result.DiffusionLatent, null, null, token),
                        Observe(result.RawVaeLatent, null, null, token), Observe(result.Image, null, null, token) };
                    string[] hashes = finals.Select(t => t.Sha256).ToArray();
                    if (expectedFinals is not null && !expectedFinals.SequenceEqual(hashes))
                        throw new InvalidDataException("Repeated final tensors differ.");
                    expectedFinals ??= hashes;
                    Verify(inputs["noise"], Observe(noise, null, null, token));
                    Verify(inputs["sigmas"], Observe(sigmas, null, null, token));
                    executions.Add(new { iteration = repetition + 1, boundaryObserver = repetition == 1,
                        milliseconds = timer.Elapsed.TotalMilliseconds, finals });
                }
                if (captures.Count != 8) throw new InvalidDataException("Incomplete pipeline boundary capture.");
                report["inputs"] = inputs;
                report["boundaryCaptures"] = captures;
                report["executions"] = executions;
                report["outputHashesRepeat"] = true;
                report["inputHashesUnchanged"] = true;
            }
            foreach (var (name, tensor) in borrowed) Verify(parameterRecords[name], Observe(tensor, null, null, token));
            report["parametersBefore"] = parameterRecords;
            report["parameterHashesUnchanged"] = true;
            report["parameterCapturePoint"] = "aligned_transferred_wrappers_before_graph_creation_and_after_graph_disposal";

            void CheckGenerated(string component, IReadOnlyList<SyntheticTensorRecord> records)
            {
                foreach (var record in records)
                    if (parameterRecords[component + "/" + record.Name].Sha256 != record.Sha256)
                        throw new InvalidDataException("Transferred parameter bytes differ from generated bytes.");
            }
        }
        report["workingSetAfterDispose"] = CheckBudget(budget);
        report["memoryQualification"] = "Observations and allowances do not prove a peak limit or absence of leaks.";
        report["runtime"] = CaptureRuntime();
        token.ThrowIfCancellationRequested();
        // The completion manifest appears only after repeats, input/weight integrity and disposal succeed.
        string temporary = Path.Combine(directory, "manifest.json.tmp");
        using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            JsonSerializer.Serialize(stream, report, JsonOptions);
        token.ThrowIfCancellationRequested();
        File.Move(temporary, Path.Combine(directory, "manifest.json"), overwrite: false);
        report["manifest"] = "manifest.json";
    }

    private static Tensor Input(float[] values, long[] shape)
    {
        var tensor = empty(shape, dtype: ScalarType.Float32, device: CPU);
        try { values.AsSpan().CopyTo(MemoryMarshal.Cast<byte, float>(tensor.bytes)); return tensor; }
        catch { tensor.Dispose(); throw; }
    }

    internal static TensorRecord Observe(Tensor tensor, string? directory, string? file, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        if (!BitConverter.IsLittleEndian || tensor.device_type != DeviceType.CPU || tensor.dtype != ScalarType.Float32 || tensor.is_sparse)
            throw new NotSupportedException("Evidence requires dense little-endian CPU/F32 tensors.");
        // Preserve computation layout; only the serialized snapshot is made contiguous.
        long[] shape = tensor.shape, stride = tensor.stride();
        long offset = tensor.storage_offset();
        bool contiguous = tensor.is_contiguous();
        using var snapshot = tensor.contiguous();
        var bytes = snapshot.bytes;
        foreach (float value in MemoryMarshal.Cast<byte, float>(bytes))
            if (!float.IsFinite(value)) throw new InvalidDataException("Nonfinite diagnostic tensor.");
        string hash = Convert.ToHexStringLower(SHA256.HashData(bytes));
        if (file is not null)
        {
            using var stream = new FileStream(Path.Combine(directory!, file), FileMode.CreateNew, FileAccess.Write, FileShare.None);
            stream.Write(bytes);
        }
        token.ThrowIfCancellationRequested();
        return new(file, shape, stride, offset, contiguous, bytes.Length, hash);
    }

    private static void Verify(TensorRecord before, TensorRecord after)
    {
        if (before.Sha256 != after.Sha256 || before.Bytes != after.Bytes || before.StorageOffset != after.StorageOffset ||
            before.Contiguous != after.Contiguous || !before.Shape.SequenceEqual(after.Shape) || !before.Stride.SequenceEqual(after.Stride))
            throw new InvalidDataException("A retained input or parameter changed during execution.");
    }

    private static object CaptureRuntime()
    {
        var modules = new List<object>();
        string? errorType = null;
        try
        {
            using var process = Process.GetCurrentProcess();
            foreach (ProcessModule module in process.Modules)
            {
                string name = Path.GetFileName(module.FileName);
                if (name.Equals("TorchSharp.dll", StringComparison.OrdinalIgnoreCase) ||
                    !new[] { "torch", "libtorch", "liblibtorchsharp", "c10", "libc10", "libgomp", "libomp", "libiomp", "iomp", "libshm" }
                        .Any(prefix => name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))) continue;
                using var stream = File.OpenRead(module.FileName);
                modules.Add(new { name, bytes = stream.Length, sha256 = Convert.ToHexStringLower(SHA256.HashData(stream)) });
            }
        }
        catch (Exception error) { errorType = error.GetType().Name; }
        string? requested = Environment.GetEnvironmentVariable("ATEN_CPU_CAPABILITY");
        return new { capturePoint = "after_all_repeats_and_bank_disposal_in_this_process", dotnet = RuntimeInformation.FrameworkDescription,
            os = RuntimeInformation.OSDescription, architecture = RuntimeInformation.ProcessArchitecture.ToString(),
            torchSharp = typeof(Tensor).Assembly.GetName().Version?.ToString(), declaredLibtorchPackage = "2.10.0",
            threads = get_num_threads(), interopThreads = get_num_interop_threads(),
            requestedAtenCpuCapability = requested is null ? "unset" : requested is "default" or "avx2" or "avx512" or "vsx" or "zvector" or "sve256"
                ? requested : "unrecognized_value_redacted", actualAtenCpuCapability = "not_observed",
            nativeLibraries = new { status = errorType is not null ? "partial" : modules.Count == 0 ? "unavailable" : "available", modules, errorType } };
    }

    private static long Memory() { using var process = Process.GetCurrentProcess(); return process.WorkingSet64; }
    private static long CheckBudget(long budget)
    {
        long current = Memory();
        if (current > budget) throw new BudgetExceededException();
        return current;
    }
    private static void Write(TextWriter output, object value) => output.WriteLine(JsonSerializer.Serialize(value, JsonOptions));
}
