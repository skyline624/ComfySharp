using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

/// <summary>Explicit real-weight diagnostic of two OFT targets; not an all-target factory or inference-loader test.</summary>
internal static class SdOftTrainingDiagnostic
{
    internal static int Run(string[] args, TextWriter output, TextWriter progress, CancellationToken cancellationToken)
    {
        var watch = Stopwatch.StartNew();
        try
        {
            var options = new Dictionary<string, string>(StringComparer.Ordinal);
            for (int i = 0; i < args.Length; i += 2)
            {
                if (i + 1 >= args.Length || args[i] is not ("--checkpoint" or "--report"))
                    throw new ArgumentException("Usage: sd-oft-train --checkpoint EXISTING.safetensors --report NEW.json (CPU Float32, two targets)");
                options.Add(args[i], args[i + 1]);
            }
            string checkpoint = Path.GetFullPath(options["--checkpoint"]), report = Path.GetFullPath(options["--report"]);
            if (File.Exists(report) || Directory.Exists(report) || !Directory.Exists(Path.GetDirectoryName(report)))
                throw new IOException("Report must be a new file in an existing directory.");
            cancellationToken.ThrowIfCancellationRequested();
            using var file = new SafeTensorFile(checkpoint);
            var plan = Sd15CheckpointLoader.Inspect(file, new() { UnclaimedTensors = Sd15UnclaimedTensorHandling.ReportAndIgnoreOutsideComponents }, cancellationToken);
            string hash = file.ComputeSha256(cancellationToken);
            NativeRuntimeBootstrap.Initialize(); set_num_threads(16); InferenceDevice.ConfigureFloat32(CPU);
            using var loaded = Sd15CheckpointLoader.Load(file, plan, 16L * 1024 * 1024 * 1024, cancellationToken);
            using var model = loaded.CreateUnet(); using var scope = NewDisposeScope(); using var enabled = set_grad_enabled(true);
            using var input = NativeMath.CpuNoise([1, 4, 8, 8], 511, cancellationToken);
            using var context = NativeMath.CpuNoise([1, 3, 768], 512, cancellationToken);
            using var target = NativeMath.CpuNoise([1, 4, 8, 8], 513, cancellationToken);
            using var time = tensor(new[] { 17.25f });
            using var baseline = model.Forward(input, time, context, cancellationToken); string baselineHash = Hash(baseline);
            var modes = new List<object>();
            foreach (bool bypass in new[] { false, true })
            {
                using var iteration = NewDisposeScope();
                using var convolution = new TrainableOftPatch(zeros([1, 4, 4]), .1);
                using var linear = new TrainableOftPatch(zeros([80, 4, 4]), .1);
                var patches = new Dictionary<string, TrainableWeightPatch>
                {
                    { "out.2.weight", convolution },
                    { "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight", linear }
                };
                using var optimizer = new LoraTrainingOptimizer(patches.Values, "SGD", .01);
                var steps = new List<object>();
                Tensor Forward() => model.ForwardForTraining(input, time, context, patches, bypass ? 0 : 8L * 1024 * 1024, cancellationToken, bypass);
                for (int step = 0; step < 2; step++)
                {
                    cancellationToken.ThrowIfCancellationRequested(); using var stepScope = NewDisposeScope();
                    progress.WriteLine($"OFT two-target diagnostic: bypass={bypass}, step {step + 1}/2");
                    using var prediction = Forward(); using var loss = TrainingLoss.Calculate("MSE", prediction, target);
                    optimizer.Accumulate(loss, cancellationToken);
                    var gradients = new Dictionary<string, float>(StringComparer.Ordinal);
                    foreach (var (name, patch) in patches)
                    {
                        var oft = (TrainableOftPatch)patch;
                        using var gradient = oft.NamedParameters["oft_blocks"].grad;
                        if (gradient is null || !gradient.isfinite().all().item<bool>()) throw new ArithmeticException("Missing or nonfinite OFT block gradient: " + name);
                        float norm = gradient.abs().sum().item<float>();
                        if (!(norm > 0)) throw new ArithmeticException("Zero OFT block gradient: " + name);
                        gradients.Add(name, norm);
                        using var alphaGradient = oft.NamedParameters["alpha"].grad;
                        if (alphaGradient is not null) throw new InvalidOperationException("Captured OFT alpha unexpectedly received a gradient.");
                    }
                    optimizer.Step(cancellationToken); steps.Add(new { step = step + 1, loss = loss.item<float>(), gradientAbsoluteSums = gradients });
                }
                using var trained = Forward();
                if (Hash(trained) == baselineHash) throw new InvalidOperationException("OFT updates did not change the model prediction.");
                using var state = LoraTrainingState.Capture(patches, ScalarType.Float32, cancellationToken: cancellationToken);
                Tensor State(string name, string key) => state.Tensors["diffusion_model." + name[..^7] + "." + key];
                var rebuilt = new Dictionary<string, TrainableWeightPatch>(StringComparer.Ordinal);
                try
                {
                    foreach (var name in patches.Keys) rebuilt.Add(name, new TrainableOftPatch(State(name, "oft_blocks"), State(name, "alpha").item<float>()));
                    // Explicit reconstruction of trainable leaves only: this does not exercise LoraFileLoader or the resume factory.
                    state.Dispose(); convolution.Dispose(); linear.Dispose();
                    using var reconstructed = model.ForwardForTraining(input, time, context, rebuilt, bypass ? 0 : 8L * 1024 * 1024, cancellationToken, bypass);
                    if (Hash(trained) != Hash(reconstructed)) throw new InvalidOperationException("OFT parameter reconstruction changed prediction.");
                    using var unchanged = model.Forward(input, time, context, cancellationToken);
                    if (Hash(unchanged) != baselineHash) throw new InvalidOperationException("OFT training changed base weights.");
                    modes.Add(new { bypassMode = bypass, targets = patches.Keys.ToArray(), steps, baseUnchanged = true,
                        trainedPredictionSha256 = Hash(trained), reconstructedTrainingPredictionExact = true, exportedAdapterFile = (string?)null });
                }
                finally { foreach (var patch in rebuilt.Values) patch.Dispose(); }
            }
            var result = new { status = "ok", familyQualified = false, trainingNodeQualified = false, inferenceLoaderQualified = false,
                checkpoint = Path.GetFileName(checkpoint), modelSha256 = hash, backend = "cpu", dtype = "Float32", threads = 16,
                algorithm = "OFT", scope = "Two existing SD1.5 targets, two SGD steps in each training mode, in-memory leaf reconstruction; no image generation or full factory/resume qualification.",
                baselinePredictionSha256 = baselineHash, modes, elapsedSeconds = watch.Elapsed.TotalSeconds };
            string json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
            using (var stream = new FileStream(report, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            using (var writer = new StreamWriter(stream)) writer.WriteLine(json);
            output.WriteLine(json); return 0;
        }
        catch (Exception error) { output.WriteLine(JsonSerializer.Serialize(new { status = "failed", familyQualified = false, error = error.GetType().Name, message = error.Message })); return 1; }
    }
    private static string Hash(Tensor value) { using var scope = NewDisposeScope(); return Convert.ToHexStringLower(SHA256.HashData(value.detach().contiguous().bytes)); }
}
