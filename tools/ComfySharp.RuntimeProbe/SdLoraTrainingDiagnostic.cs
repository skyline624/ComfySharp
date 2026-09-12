using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

/// <summary>Explicit miniature raw-prediction training/serialization probe with real SD1.5 weights.
/// This is not the TrainLoraNode dataset, noise schedule or a style-quality test.</summary>
internal static class SdLoraTrainingDiagnostic
{
    internal static int Run(string[] args, TextWriter output, TextWriter progress, CancellationToken cancellationToken)
    {
        var watch = Stopwatch.StartNew();
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            var options = new Dictionary<string, string>(StringComparer.Ordinal);
            for (int i = 0; i < args.Length; i += 2)
            {
                if (i + 1 >= args.Length || args[i] is not ("--checkpoint" or "--adapter-output" or "--report" or "--device" or "--optimizer" or "--loss" or "--accumulation-steps"))
                    throw new ArgumentException("Usage: sd-lora-train --checkpoint FILE --adapter-output NEW.safetensors --report NEW.json --device cpu|cuda:0 [--optimizer Adam|AdamW|SGD|RMSprop --loss MSE|L1|Huber|SmoothL1 --accumulation-steps N]");
                options.Add(args[i], args[i + 1]);
            }
            string Required(string name) => options.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value) ? value : throw new ArgumentException("Missing " + name);
            string checkpointPath = Path.GetFullPath(Required("--checkpoint"));
            string destination = Path.GetFullPath(Required("--adapter-output")); string report = Path.GetFullPath(Required("--report"));
            if (string.Equals(destination, report, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal) ||
                new[] { destination, report }.Any(p => File.Exists(p) || Directory.Exists(p) || !Directory.Exists(Path.GetDirectoryName(p))))
                throw new IOException("Output and report must be distinct new files in existing directories.");
            if (!destination.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("Adapter output requires .safetensors.");
            string requested = Required("--device");
            if (requested is not ("cpu" or "cuda:0")) throw new ArgumentException("Select cpu or cuda:0.");
            string optimizerName = options.GetValueOrDefault("--optimizer", "SGD"), lossName = options.GetValueOrDefault("--loss", "MSE");
            if (optimizerName is not ("Adam" or "AdamW" or "SGD" or "RMSprop") || lossName is not ("MSE" or "L1" or "Huber" or "SmoothL1"))
                throw new ArgumentException("Unknown optimizer or training loss.");
            if (!int.TryParse(options.GetValueOrDefault("--accumulation-steps", "1"), out int accumulationSteps) || accumulationSteps < 1 || accumulationSteps > 1024)
                throw new ArgumentException("Accumulation steps must be between 1 and 1024.");
            using var file = new SafeTensorFile(checkpointPath);
            var plan = Sd15CheckpointLoader.Inspect(file, new() { UnclaimedTensors = Sd15UnclaimedTensorHandling.ReportAndIgnoreOutsideComponents }, cancellationToken);
            string modelHash = file.ComputeSha256(cancellationToken);
            NativeRuntimeBootstrap.Initialize(); set_num_threads(1);
            if (requested == "cuda:0" && !cuda.is_available()) throw new NotSupportedException("Requested CUDA training device is unavailable.");
            var device = requested == "cpu" ? CPU : new Device(DeviceType.CUDA, 0);
            InferenceDevice.ConfigureFloat32(device);
            using var loaded = Sd15CheckpointLoader.Load(file, plan, 16L * 1024 * 1024 * 1024, cancellationToken);
            using var cpu = loaded.CreateUnet(); using var model = cpu.To(device, cancellationToken);
            if (requested != "cpu") { cpu.Dispose(); loaded.Dispose(); }
            using var scope = NewDisposeScope(); using var gradMode = set_grad_enabled(true);
            Tensor Noise(long[] shape, ulong seed, float scale = 1)
            {
                using var noise = NativeMath.CpuNoise(shape, seed, cancellationToken);
                return noise.to(device) * scale;
            }
            var latent = Noise([1, 4, 8, 8], 111); var context = Noise([1, 3, 768], 112); var target = Noise([1, 4, 8, 8], 110);
            var timesteps = tensor(new[] { 17.25f }, device: device);
            var patches = new Dictionary<string, TrainableLoraPatch>(StringComparer.Ordinal);
            try
            {
                patches.Add("input_blocks.0.0.weight", new(Noise([320, 2], 121, .001f), Noise([2, 36], 122, .01f), 2.5));
                patches.Add("out.2.weight", new(Noise([4, 2], 123, .001f), Noise([2, 2880], 124, .01f), 2.5));
                var initialHashes = ParameterHashes(patches);
                using var before = model.Forward(latent, timesteps, context, cancellationToken);
                string baseHash = Hash(before);
                var steps = new List<object>();
                using var optimizer = new LoraTrainingOptimizer(patches.Values, optimizerName, .0005, accumulationSteps);
                for (int step = 0; step < 2; step++)
                {
                    cancellationToken.ThrowIfCancellationRequested(); using var iteration = NewDisposeScope();
                    var microbatchLosses = new List<float>();
                    for (int micro = 0; micro < accumulationSteps; micro++)
                    {
                        using var microScope = NewDisposeScope();
                        progress.WriteLine($"SD LoRA {optimizerName}/{lossName}: step {step + 1}/2, microbatch {micro + 1}/{accumulationSteps} on {requested}");
                        using var prediction = model.ForwardForTraining(latent, timesteps, context, patches, cancellationToken: cancellationToken);
                        using var loss = TrainingLoss.Calculate(lossName, prediction, target);
                        optimizer.Accumulate(loss, cancellationToken); microbatchLosses.Add(loss.item<float>());
                    }
                    var norms = new Dictionary<string, object>(StringComparer.Ordinal);
                    foreach (var (name, patch) in patches)
                    {
                        using var up = patch.Up.grad ?? throw new InvalidOperationException("Missing up gradient.");
                        using var down = patch.Down.grad ?? throw new InvalidOperationException("Missing down gradient.");
                        if (!up.isfinite().all().item<bool>() || !down.isfinite().all().item<bool>()) throw new ArithmeticException("Nonfinite adapter gradients.");
                        double upNorm = up.norm().item<float>(), downNorm = down.norm().item<float>();
                        if (upNorm == 0 || downNorm == 0) throw new InvalidOperationException("Expected nonzero full-graph gradients.");
                        norms.Add(name, new { upNorm, downNorm });
                    }
                    optimizer.Step(cancellationToken);
                    steps.Add(new { step = step + 1, loss = microbatchLosses.Average(), microbatchLosses, gradientNorms = norms });
                }
                using var baseAfter = model.Forward(latent, timesteps, context, cancellationToken);
                if (Hash(baseAfter) != baseHash) throw new InvalidOperationException("Base model changed during adapter training.");
                var parameterHashes = ParameterHashes(patches);
                if (parameterHashes.Any(p => p.Value == initialHashes[p.Key])) throw new InvalidOperationException("An adapter did not update.");
                LoraTrainingFile.SaveNew(destination, patches.ToDictionary(p => "diffusion_model." + p.Key[..^7], p => p.Value, StringComparer.Ordinal), cancellationToken: cancellationToken);
                using var adapterFile = new SafeTensorFile(destination);
                var aliasPlan = LoraFileLoader.Inspect(adapterFile, LoraModelAliases.ForUnet(model.Config), cancellationToken: cancellationToken);
                using var reloaded = LoraFileLoader.Load(adapterFile, aliasPlan, cancellationToken: cancellationToken);
                using var reloadedModel = reloaded.ApplyTo(model, cancellationToken: cancellationToken);
                var snapshots = patches.ToDictionary(p => p.Key, p => p.Value.Snapshot(), StringComparer.Ordinal);
                try
                {
                    using var baked = model.WithLora(snapshots, cancellationToken: cancellationToken);
                    using var expected = baked.Forward(latent, timesteps, context, cancellationToken);
                    using var actual = reloadedModel.Forward(latent, timesteps, context, cancellationToken);
                    if (Hash(actual) != Hash(expected)) throw new InvalidOperationException("Reloaded adapter prediction differs from the in-memory snapshot.");
                    string json = JsonSerializer.Serialize(new
                    {
                        status = "ok", familyQualified = false, trainingNodeQualified = false, trainedStyleQualified = false,
                        checkpoint = Path.GetFileName(checkpointPath), modelSha256 = modelHash, checkpointBytes = file.FileSizeBytes,
                        backend = requested, dtype = "Float32", tf32Allowed = false, learningRate = .0005, rank = 2, alpha = 2.5, steps,
                        optimizer = optimizerName, lossFunction = lossName, accumulationSteps, optimizerSteps = optimizer.CompletedSteps,
                        inputRecipe = new { latentShape = new[] { 1, 4, 8, 8 }, contextShape = new[] { 1, 3, 768 }, latentSeed = 111, contextSeed = 112, targetSeed = 110, timestep = 17.25 },
                        adapter = Path.GetFileName(destination), adapterBytes = adapterFile.FileSizeBytes, adapterSha256 = adapterFile.ComputeSha256(cancellationToken),
                        baseUnchanged = true, initialParameterHashes = initialHashes, updatedParameterHashes = parameterHashes,
                        reloadedPredictionIdentical = true, predictionSha256 = Hash(actual), elapsedSeconds = watch.Elapsed.TotalSeconds,
                        scope = "Pretrained SD1.5 raw U-Net with synthetic miniature inputs/targets, accumulated gradients and two optimizer updates. No dataset training, denoising schedule, semantic/style quality or complete TrainLoraNode qualification."
                    }, new JsonSerializerOptions { WriteIndented = true });
                    using var stream = new FileStream(report, FileMode.CreateNew, FileAccess.Write, FileShare.None); using var writer = new StreamWriter(stream); writer.Write(json);
                    output.WriteLine(json); return 0;
                }
                finally { foreach (var snapshot in snapshots.Values) snapshot.Dispose(); }
            }
            finally { foreach (var patch in patches.Values) patch.Dispose(); }
        }
        catch (Exception error)
        {
            output.WriteLine(JsonSerializer.Serialize(new { status = "error", familyQualified = false, error = error.GetType().Name, error.Message })); return 1;
        }
    }

    private static string Hash(Tensor tensor)
    {
        using var scope = NewDisposeScope(); var cpu = tensor.detach().cpu().contiguous();
        return Convert.ToHexStringLower(SHA256.HashData(cpu.bytes));
    }

    private static Dictionary<string, string> ParameterHashes(IReadOnlyDictionary<string, TrainableLoraPatch> patches)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (name, patch) in patches) { result.Add(name + "/up", Hash(patch.Up)); result.Add(name + "/down", Hash(patch.Down)); }
        return result;
    }
}
