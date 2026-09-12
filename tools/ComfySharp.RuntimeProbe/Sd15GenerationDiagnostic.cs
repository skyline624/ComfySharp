using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using ComfySharp.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

/// <summary>Explicit single-checkpoint, CPU/F32 text-to-image diagnostic. No downloads or weight copies.</summary>
internal static class Sd15GenerationDiagnostic
{
    internal const string Usage = "sd15-generate --checkpoint <file> [--report-outside-components] [--execute --output <new.png> [--trace-dir <new-directory>]] [--prompt <text>] [--negative <text>] [--width 32..512] [--height 32..512] [--steps 1..100] [--seed <uint64>] [--cfg 0..30] [--threads 1..64] [--weight-budget-mib N]";

    internal sealed record Options(string Checkpoint, string? Output, string Prompt, string Negative,
        bool Execute, bool ReportOutsideComponents, int Width, int Height, int Steps, ulong Seed, double Cfg, int Threads, long WeightBudgetBytes,
        string? TraceDirectory);

    internal sealed record TraceRecord(string File, long[] Shape, string Dtype, long Bytes, string Sha256);

    internal static TraceRecord Capture(Tensor value, string directory, string name, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!BitConverter.IsLittleEndian || name.Length == 0 || name.Any(c => !char.IsAsciiLetterOrDigit(c) && c != '-'))
            throw new ArgumentException("Trace names must be ASCII letters, digits or hyphens on a little-endian platform.");
        using var scope = NewDisposeScope();
        if (value.dtype != ScalarType.Float32 || value.device_type != DeviceType.CPU || value.is_sparse ||
            !value.isfinite().all().item<bool>()) throw new ArgumentException("Trace requires finite dense CPU/F32 data.");
        var flat = value.contiguous();
        byte[] bytes = flat.bytes.ToArray();
        cancellationToken.ThrowIfCancellationRequested();
        string filename = name + ".f32";
        using (var file = new FileStream(Path.Combine(directory, filename), FileMode.CreateNew, FileAccess.Write, FileShare.None))
            file.Write(bytes);
        return new(filename, value.shape, "float32-le", bytes.LongLength, Convert.ToHexStringLower(SHA256.HashData(bytes)));
    }

    internal static Options Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        bool execute = false, reportOutside = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--report-outside-components")
            {
                if (reportOutside) throw new ArgumentException("Duplicate --report-outside-components.");
                reportOutside = true;
                continue;
            }
            if (args[i] == "--execute")
            {
                if (execute) throw new ArgumentException("Duplicate --execute.");
                execute = true;
                continue;
            }
            string key = args[i];
            if (key is not ("--checkpoint" or "--output" or "--prompt" or "--negative" or "--width" or "--height" or
                "--steps" or "--seed" or "--cfg" or "--threads" or "--weight-budget-mib" or "--trace-dir") || ++i >= args.Length || !values.TryAdd(key, args[i]))
                throw new ArgumentException("Unknown, incomplete or duplicate argument.");
        }
        string Value(string key, string fallback) => values.GetValueOrDefault(key, fallback);
        int Number(string key, int fallback, int min, int max)
        {
            if (!int.TryParse(Value(key, fallback.ToString(CultureInfo.InvariantCulture)), NumberStyles.None,
                CultureInfo.InvariantCulture, out int value) || value < min || value > max)
                throw new ArgumentException($"Invalid {key}.");
            return value;
        }
        string checkpoint = Value("--checkpoint", "");
        if (string.IsNullOrWhiteSpace(checkpoint)) throw new ArgumentException("Checkpoint is required.");
        checkpoint = Path.GetFullPath(checkpoint);
        string? destination = values.TryGetValue("--output", out string? path) ? Path.GetFullPath(path) : null;
        if (execute && destination is null) throw new ArgumentException("Execution requires --output.");
        if (!execute && destination is not null) throw new ArgumentException("Output requires --execute.");
        string? traceDirectory = values.TryGetValue("--trace-dir", out string? tracePath) ? Path.GetFullPath(tracePath) : null;
        if (traceDirectory is not null && (!execute || File.Exists(traceDirectory) || Directory.Exists(traceDirectory) ||
            !Directory.Exists(Path.GetDirectoryName(traceDirectory))))
            throw new ArgumentException("Tracing requires execution and a new directory in an existing parent.");
        if (destination is not null && (!destination.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
            File.Exists(destination) || Directory.Exists(destination) || !Directory.Exists(Path.GetDirectoryName(destination))))
            throw new ArgumentException("Output must be a new PNG file in an existing directory.");
        int width = Number("--width", 512, 32, 512), height = Number("--height", 512, 32, 512);
        if (width % 8 != 0 || height % 8 != 0) throw new ArgumentException("Dimensions must be multiples of eight.");
        if (!ulong.TryParse(Value("--seed", "0"), NumberStyles.None, CultureInfo.InvariantCulture, out ulong seed))
            throw new ArgumentException("Invalid seed.");
        if (!double.TryParse(Value("--cfg", "7"), NumberStyles.Float, CultureInfo.InvariantCulture, out double cfg) ||
            !double.IsFinite(cfg) || cfg is < 0 or > 30) throw new ArgumentException("Invalid CFG scale.");
        string positive = Value("--prompt", "a photograph of a red apple on a wooden table"), negative = Value("--negative", "");
        if (positive.Length > 4096 || negative.Length > 4096) throw new ArgumentException("Diagnostic text limit is 4096 characters.");
        return new(checkpoint, destination, positive, negative, execute, reportOutside, width, height, Number("--steps", 20, 1, 100), seed, cfg,
            Number("--threads", Math.Min(Environment.ProcessorCount, 8), 1, 64), (long)Number("--weight-budget-mib", 16384, 1, 131072) * 1024 * 1024,
            traceDirectory);
    }

    internal static int Run(string[] args, TextWriter output, TextWriter progress, CancellationToken cancellationToken = default)
    {
        string stage = "arguments";
        var watch = Stopwatch.StartNew();
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (args is ["--help"])
            {
                Write(new { status = "ok", usage = Usage, backend = "cpu", dtype = "Float32", familyQualified = false });
                return 0;
            }
            var options = Parse(args);
            Stage("inspect");
            using var file = new SafeTensorFile(options.Checkpoint);
            var inspection = new Sd15CheckpointInspectionOptions { UnclaimedTensors = options.ReportOutsideComponents
                ? Sd15UnclaimedTensorHandling.ReportAndIgnoreOutsideComponents : Sd15UnclaimedTensorHandling.Reject };
            var plan = Sd15CheckpointLoader.Inspect(file, inspection, cancellationToken);
            if (plan.EstimatedPeakWeightBytes > options.WeightBudgetBytes)
                throw new InvalidOperationException("Estimated peak weights exceed --weight-budget-mib. This limit excludes activations and runtime overhead.");
            if (!options.Execute)
            {
                Write(new { status = "ok", operation = "inspect", familyQualified = false, plan.SourceBytes, plan.ResidentBytes,
                    plan.EstimatedPeakWeightBytes, plan.HasClipProjection, plan.UnclaimedTensorNames });
                return 0;
            }
            Stage("hash");
            string modelHash = file.ComputeSha256(cancellationToken);
            var traces = new List<TraceRecord>();
            if (options.TraceDirectory is not null) Directory.CreateDirectory(options.TraceDirectory);
            Stage("load");
            NativeRuntimeBootstrap.Initialize();
            set_num_threads(options.Threads);
            using var checkpoint = Sd15CheckpointLoader.Load(file, plan, options.WeightBudgetBytes, cancellationToken);
            using var clip = checkpoint.CreateClipEncoder();
            using var unet = checkpoint.CreateUnet();
            using var vae = checkpoint.CreateImageVae();
            using var scope = NewDisposeScope();
            using var inference = no_grad();
            Stage("encode");
            var tokenizer = new ComfyClipTokenizer(ClipTokenizer.CreateDefault(), ClipProfile.Sd1L);
            using var positive = clip.Encode(tokenizer.Tokenize(options.Prompt, cancellationToken: cancellationToken), cancellationToken: cancellationToken);
            using var negative = clip.Encode(tokenizer.Tokenize(options.Negative, cancellationToken: cancellationToken), cancellationToken: cancellationToken);
            Trace("positive-hidden", positive.Hidden); Trace("negative-hidden", negative.Hidden);
            using var noise = NativeMath.CpuNoise([1, 4, options.Height / 8, options.Width / 8], options.Seed, cancellationToken);
            var sampling = SdDiscreteSampling.Default;
            using var sigmas = SigmaSchedules.Karras(options.Steps, sampling.SigmaMin, sampling.SigmaMax, cancellationToken: cancellationToken);
            Trace("noise", noise); Trace("sigmas", sigmas);
            using var empty = zeros_like(noise);
            using var firstSigma = sigmas[0];
            using var initial = SdSamplingMath.NoiseScaling(noise, empty, firstSigma, true, cancellationToken);
            Trace("initial", initial);
            using var denoiser = new SdDenoiser(unet, SdPredictionKind.Epsilon, sampling);
            using var sampler = new SdEulerSampler(denoiser);
            Stage("sample");
            using var diffusion = sampler.Sample(initial, sigmas, positive.Hidden, negative.Hidden,
                new SdGuidanceOptions { Scale = options.Cfg, BatchMode = SdGuidanceBatchMode.Separate }, cancellationToken);
            Trace("diffusion", diffusion);
            Stage("decode");
            using var raw = SdSamplingMath.ProcessLatentOut(diffusion, SdSamplingMath.Sd15LatentScale, cancellationToken);
            Trace("raw-vae", raw);
            using var image = vae.Decode(raw, cancellationToken);
            Trace("image", image);
            var shape = image.shape;
            if (!shape.SequenceEqual(new long[] { 1, options.Height, options.Width, 3 }) || !image.isfinite().all().item<bool>())
                throw new InvalidDataException("Decoded image has invalid dimensions or non-finite pixels.");
            Stage("png");
            var settings = new { modelSha256 = modelHash, prompt = options.Prompt, negative = options.Negative,
                width = options.Width, height = options.Height, steps = options.Steps, seed = options.Seed, cfg = options.Cfg,
                sampler = "euler", scheduler = "karras", backend = "cpu", dtype = "Float32", threads = options.Threads,
                ignoredOutsideComponentTensors = plan.UnclaimedTensorNames };
            byte[] png = ImagePngEncoder.EncodeFrame(image, 0, [new PngText("comfysharp.sd15", JsonSerializer.Serialize(settings))], cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            // CreateNew also protects against another writer appearing after argument validation.
            using (var destination = new FileStream(options.Output!, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                destination.Write(png);
            Write(new { status = "ok", operation = "generate", familyQualified = false, settings, shape,
                pngSha256 = Convert.ToHexStringLower(SHA256.HashData(png)), pngBytes = png.Length,
                elapsedSeconds = watch.Elapsed.TotalSeconds, peakWorkingSetBytes = Process.GetCurrentProcess().PeakWorkingSet64, traces });
            return 0;

            void Trace(string name, Tensor value)
            {
                if (options.TraceDirectory is not null) traces.Add(Capture(value, options.TraceDirectory, name, cancellationToken));
            }
        }
        catch (OperationCanceledException)
        {
            Write(new { status = "cancelled", stage });
            return 130;
        }
        catch (Exception error)
        {
            Write(new { status = "error", stage, error = error.GetType().Name, message = error.Message, familyQualified = false });
            return stage == "arguments" ? 2 : 1;
        }
        void Write(object value) { output.WriteLine(JsonSerializer.Serialize(value)); output.Flush(); }
        void Stage(string value) { cancellationToken.ThrowIfCancellationRequested(); stage = value; progress.WriteLine($"{watch.Elapsed.TotalSeconds:F1}s {value}"); progress.Flush(); }
    }
}
