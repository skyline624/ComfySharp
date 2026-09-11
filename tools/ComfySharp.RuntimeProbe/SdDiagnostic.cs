using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using TorchSharp;

namespace ComfySharp.RuntimeProbe;

/// <summary>Explicit synthetic stock-width U-Net diagnostic, never a compatibility claim.</summary>
internal static class SdDiagnostic
{
    private const long MiB = 1024 * 1024;
    private const long RuntimeAllowance = 512 * MiB;
    private const long GraphAllowance = 1024 * MiB;
    private const long Headroom = 256 * MiB;
    private const string Usage = "sd --model sd15|sd2 [--inspect] [--memory-budget-mib N] [--chunk-elements 1..262144] [--repeat 1..3] [--synthetic --execute --output <new-absolute-directory>]";
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    internal sealed record Options(string Model, SdUnetConfig Config, bool Execute, long? BudgetBytes,
        int ChunkElements, int Repeats, string? OutputDirectory);
    internal sealed class BudgetExceededException : Exception { }

    // The optional resolver is internal and lets ordinary tests exercise the actual
    // execution with a reduced bank. The CLI exposes only the two stock configurations.
    internal static int Run(string[] args, TextWriter output, CancellationToken cancellationToken = default,
        Func<string, SdUnetConfig>? resolveConfig = null)
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
                Write(output, new { diagnostic = "sd", status = "ok", operation = "help", usage = Usage,
                    nativeInitialized, modelCompatibility = "not_assessed" });
                return 0;
            }
            var options = Parse(args, resolveConfig);
            stage = "plan";
            var plan = SdSyntheticWeightBuilder.DescribeUnet(options.Config);
            long inputBytes = checked((4L * 16 * 16 + 77L * options.Config.ContextSize + 1) * 4);
            long estimate = checked(plan.ResidentBytes + inputBytes + RuntimeAllowance + GraphAllowance + Headroom);
            var report = new Dictionary<string, object?>
            {
                ["diagnostic"] = "sd", ["status"] = "ok", ["operation"] = options.Execute ? "execute" : "plan",
                ["synthetic"] = true, ["modelCompatibility"] = "not_assessed", ["numericalQualification"] = "not_performed",
                ["model"] = options.Model, ["component"] = "unet", ["device"] = "cpu", ["dtype"] = "float32",
                ["repeats"] = options.Repeats,
                ["configurationKind"] = options.Config == Stock(options.Model) ? "stock" : "reduced_diagnostic",
                ["configuration"] = options.Config, ["parameterRecipe"] = SdSyntheticRecipe.Version,
                ["caseId"] = CaseId(options.Model), ["latentShape"] = new long[] { 1, 4, 16, 16 },
                ["contextShape"] = new long[] { 1, 77, options.Config.ContextSize }, ["timesteps"] = new[] { 0.125f },
                ["weightPlan"] = new { plan.TensorCount, plan.ResidentBytes, plan.LargestTensorBytes,
                    temporaryWeightPayloadBytes = 0, fillChunkBytes = checked(options.ChunkElements * 4),
                    fillChunkStorage = "view_of_native_destination_no_managed_payload_buffer" },
                ["memoryPlan"] = new { inputBytes, runtimeAllowanceBytes = RuntimeAllowance,
                    graphAllowanceBytes = GraphAllowance, headroomBytes = Headroom, estimatedProcessBytes = estimate,
                    requestedBudgetBytes = options.BudgetBytes, estimateIsPeakGuarantee = false,
                    scope = "resident_weights_inputs_plus_explicit_estimated_allowances_not_measured_free_ram" },
                ["nativeInitialized"] = false, ["weightsGenerated"] = false
            };
            if (options.BudgetBytes is long budget && estimate > budget)
            {
                report["status"] = "budget_exceeded";
                Write(output, report);
                return 3;
            }
            if (!options.Execute)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Write(output, report);
                return 0;
            }
            stage = "initialize";
            cancellationToken.ThrowIfCancellationRequested();
            NativeRuntimeBootstrap.Initialize();
            nativeInitialized = true;
            int previousThreads = torch.get_num_threads();
            try
            {
                torch.set_num_threads(1);
                if (torch.get_num_interop_threads() != 1) torch.set_num_interop_threads(1);
                stage = "execute";
                Execute(options, plan, inputBytes, report, cancellationToken);
            }
            finally { torch.set_num_threads(previousThreads); }
            Write(output, report);
            return 0;
        }
        catch (OperationCanceledException)
        {
            Write(output, new { diagnostic = "sd", status = "cancelled", stage, nativeInitialized,
                modelCompatibility = "not_assessed" });
            return 130;
        }
        catch (Exception error)
        {
            int code = error is BudgetExceededException ? 3 : error is ArgumentException or OverflowException
                or NotSupportedException or IOException or UnauthorizedAccessException ? 2 : 1;
            // Do not expose supplied paths, arbitrary arguments, native search paths or stacks.
            Write(output, new { diagnostic = "sd", status = code == 3 ? "budget_exceeded" : "error", stage,
                nativeInitialized, modelCompatibility = "not_assessed", error = error.GetType().Name,
                message = stage == "arguments" ? "Invalid SD diagnostic arguments. Use sd --help."
                    : code == 3 ? "The requested budget is insufficient for the estimated remaining work or observed process memory."
                    : "The synthetic SD diagnostic could not complete. No model compatibility was assessed." });
            return code;
        }
    }

    internal static Options Parse(string[] args, Func<string, SdUnetConfig>? resolveConfig = null)
    {
        string? model = null, directory = null;
        bool execute = false, synthetic = false, inspect = false;
        long? budget = null;
        int chunk = 262_144, repeats = 3;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            string option = args[i];
            if (!seen.Add(option)) throw new ArgumentException("Duplicate option.");
            switch (option)
            {
                case "--execute": execute = true; break;
                case "--synthetic": synthetic = true; break;
                case "--inspect": inspect = true; break;
                case "--model": model = Value(); break;
                case "--output": directory = Value(); break;
                case "--memory-budget-mib": budget = checked(PositiveLong(Value()) * MiB); break;
                case "--chunk-elements": chunk = checked((int)PositiveLong(Value())); break;
                case "--repeat": repeats = checked((int)PositiveLong(Value())); break;
                default: throw new ArgumentException("Unknown option.");
            }
            string Value() => ++i < args.Length ? args[i] : throw new ArgumentException("Missing value.");
        }
        if (model is not ("sd15" or "sd2") || repeats is < 1 or > 3 || chunk is < 1 or > 262_144)
            throw new ArgumentException("Invalid model, repeats or chunk length.");
        if (execute && (!synthetic || inspect || budget is null || directory is null))
            throw new ArgumentException("Execution requires explicit synthetic mode, budget and output directory.");
        if (!execute && directory is not null) throw new ArgumentException("Output is only valid for explicit execution.");
        if (directory is not null)
        {
            if (!Path.IsPathFullyQualified(directory)) throw new ArgumentException("Output must be absolute.");
            directory = Path.GetFullPath(directory);
            if (Directory.Exists(directory) || File.Exists(directory)) throw new ArgumentException("Output must be new.");
        }
        var config = resolveConfig is null ? Stock(model) : resolveConfig(model);
        config.Validate();
        return new(model, config, execute, budget, chunk, repeats, directory);
    }

    private static long PositiveLong(string value) => long.TryParse(value, NumberStyles.None,
        CultureInfo.InvariantCulture, out long result) && result > 0 ? result : throw new ArgumentException("Expected a positive integer.");
    private static SdUnetConfig Stock(string model) => model == "sd15" ? SdUnetConfig.Sd15 : SdUnetConfig.Sd2;
    private static string CaseId(string model) => "sd-stock-cpu-f32-cases-v1/" + model + "/square";

    private static void Execute(Options options, SyntheticWeightPlan plan, long inputBytes,
        Dictionary<string, object?> report, CancellationToken token)
    {
        long budget = options.BudgetBytes!.Value;
        var before = Memory();
        if (checked(before.WorkingSetBytes + plan.ResidentBytes + inputBytes + GraphAllowance + Headroom) > budget)
            throw new BudgetExceededException();
        Directory.CreateDirectory(options.OutputDirectory!);
        report["nativeInitialized"] = true;
        report["memoryBeforeGeneration"] = before;
        var timer = Stopwatch.StartNew();
        var buildOptions = new SyntheticBuildOptions(plan.ResidentBytes, options.ChunkElements);
        using (var generated = SdSyntheticWeightBuilder.CreateUnet(options.Config, buildOptions, token))
        {
            timer.Stop();
            report["generationMilliseconds"] = timer.Elapsed.TotalMilliseconds;
            report["weightsGenerated"] = true;
            report["memoryAfterGeneration"] = CheckObservedBudget(budget);
            using var model = new SdUnet(generated.Weights);
            string caseId = CaseId(options.Model);
            using var latent = SdSyntheticWeightBuilder.CreateInput(caseId + "/latent", [1, 4, 16, 16],
                options.ChunkElements, token, out _);
            using var context = SdSyntheticWeightBuilder.CreateInput(caseId + "/context", [1, 77, options.Config.ContextSize],
                options.ChunkElements, token, out _);
            using var time = torch.empty([1], dtype: torch.ScalarType.Float32, device: torch.CPU);
            MemoryMarshal.Cast<byte, float>(time.bytes)[0] = 0.125f;
            var inputs = new[] { WriteTensor(latent, "latent.f32", options.OutputDirectory!, token),
                WriteTensor(context, "context.f32", options.OutputDirectory!, token),
                WriteTensor(time, "timesteps.f32", options.OutputDirectory!, token) };
            var executions = new List<object>(options.Repeats);
            var hashes = new List<string>(options.Repeats);
            object? outputRecord = null;
            for (int repetition = 0; repetition < options.Repeats; repetition++)
            {
                token.ThrowIfCancellationRequested();
                CheckObservedBudget(budget);
                timer.Restart();
                using var result = model.Forward(latent, time, context, token);
                timer.Stop();
                string hash = Hash(result, token);
                hashes.Add(hash);
                if (repetition == 0) outputRecord = WriteTensor(result, "output.f32", options.OutputDirectory!, token);
                result.Dispose();
                executions.Add(new { iteration = repetition + 1, forwardMilliseconds = timer.Elapsed.TotalMilliseconds,
                    sha256 = hash, memoryAfterOutputDispose = CheckObservedBudget(budget) });
            }
            report["executions"] = executions;
            report["outputHashesRepeat"] = hashes.Distinct(StringComparer.Ordinal).Count() == 1;
            if (hashes.Distinct(StringComparer.Ordinal).Count() != 1) report["status"] = "non_repeatable";
            report["output"] = outputRecord;
            report["threads"] = torch.get_num_threads();
            report["interopThreads"] = torch.get_num_interop_threads();
            report["torchSharp"] = typeof(torch.Tensor).Assembly.GetName().Version?.ToString();
            report["declaredLibtorchPackage"] = "2.10.0";
            report["requestedAtenCpuCapability"] = Environment.GetEnvironmentVariable("ATEN_CPU_CAPABILITY") ?? "auto";
            token.ThrowIfCancellationRequested();
            var manifest = new { diagnosticOnly = true, modelCompatibility = "not_assessed", synthetic = true,
                caseId, parameterRecipe = SdSyntheticRecipe.Version, configuration = options.Config,
                parameters = generated.Parameters, inputs, output = outputRecord, executions,
                report = new Dictionary<string, object?>(report) };
            string temporary = Path.Combine(options.OutputDirectory!, "manifest.json.tmp");
            using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                JsonSerializer.Serialize(stream, manifest, JsonOptions);
            token.ThrowIfCancellationRequested();
            File.Move(temporary, Path.Combine(options.OutputDirectory!, "manifest.json"), overwrite: false);
            report["manifest"] = "manifest.json";
            if (hashes.Distinct(StringComparer.Ordinal).Count() != 1)
                throw new InvalidDataException("Repeated synthetic forward hashes differ.");
        }
        report["memoryAfterDispose"] = Memory();
        report["memoryQualification"] = "Observations and explicit allowances are not a peak-memory or OOM guarantee.";
    }

    private static object WriteTensor(torch.Tensor tensor, string file, string directory, CancellationToken token)
    {
        if (!tensor.is_contiguous() || tensor.dtype != torch.ScalarType.Float32 || tensor.device_type != DeviceType.CPU)
            throw new InvalidDataException("Synthetic evidence requires contiguous CPU Float32 tensors.");
        using var stream = new FileStream(Path.Combine(directory, file), FileMode.CreateNew, FileAccess.Write, FileShare.None);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var bytes = tensor.bytes;
        for (int start = 0; start < bytes.Length;)
        {
            token.ThrowIfCancellationRequested();
            int count = Math.Min(1024 * 1024, bytes.Length - start);
            stream.Write(bytes.Slice(start, count));
            hash.AppendData(bytes.Slice(start, count));
            start += count;
        }
        var values = MemoryMarshal.Cast<byte, float>(bytes);
        return new { file, shape = tensor.shape, stride = tensor.stride(), dtype = "float32", byteOrder = "little",
            bytes = bytes.Length, sha256 = Convert.ToHexStringLower(hash.GetHashAndReset()),
            sample = values[..Math.Min(8, values.Length)].ToArray() };
    }

    private static string Hash(torch.Tensor tensor, CancellationToken token)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var bytes = tensor.bytes;
        for (int start = 0; start < bytes.Length;)
        {
            token.ThrowIfCancellationRequested();
            int count = Math.Min(1024 * 1024, bytes.Length - start);
            hash.AppendData(bytes.Slice(start, count));
            start += count;
        }
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
    private sealed record ProcessMemory(long WorkingSetBytes, long PeakWorkingSetBytes, long PrivateBytes);
    private static ProcessMemory Memory()
    {
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        return new(process.WorkingSet64, process.PeakWorkingSet64, process.PrivateMemorySize64);
    }
    private static ProcessMemory CheckObservedBudget(long budget)
    {
        var memory = Memory();
        if (memory.WorkingSetBytes > budget) throw new BudgetExceededException();
        return memory;
    }
    private static void Write(TextWriter output, object value) => output.WriteLine(JsonSerializer.Serialize(value, JsonOptions));
}
