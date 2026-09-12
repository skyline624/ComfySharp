using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

/// <summary>Explicit pretrained all-target gradient diagnostic. Writes metadata only, no model or adapter copy.</summary>
internal static class SdAllAdapterDiagnostic
{
    internal static int Run(string[] args, TextWriter output, TextWriter progress, CancellationToken cancellationToken)
    {
        var watch = Stopwatch.StartNew();
        try
        {
            var options = new Dictionary<string, string>(StringComparer.Ordinal);
            for (int i = 0; i < args.Length; i += 2)
            {
                if (i + 1 >= args.Length || args[i] is not ("--checkpoint" or "--report" or "--device"))
                    throw new ArgumentException("Usage: sd-all-adapter-train --checkpoint FILE --report NEW.json --device cpu|cuda:0");
                options.Add(args[i], args[i + 1]);
            }
            string Required(string name) => options.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value) ? value : throw new ArgumentException("Missing " + name);
            string checkpoint = Path.GetFullPath(Required("--checkpoint")), report = Path.GetFullPath(Required("--report")), requested = Required("--device");
            if (File.Exists(report) || Directory.Exists(report) || !Directory.Exists(Path.GetDirectoryName(report))) throw new IOException("Report must be a new file in an existing directory.");
            if (requested is not ("cpu" or "cuda:0")) throw new ArgumentException("Select cpu or cuda:0.");
            cancellationToken.ThrowIfCancellationRequested();
            using var file = new SafeTensorFile(checkpoint);
            var plan = Sd15CheckpointLoader.Inspect(file, new() { UnclaimedTensors = Sd15UnclaimedTensorHandling.ReportAndIgnoreOutsideComponents }, cancellationToken);
            string modelSha256 = file.ComputeSha256(cancellationToken);
            NativeRuntimeBootstrap.Initialize(); set_num_threads(1);
            if (requested == "cuda:0" && !cuda.is_available()) throw new NotSupportedException("CUDA is unavailable.");
            var device = requested == "cpu" ? CPU : new Device(DeviceType.CUDA, 0); InferenceDevice.ConfigureFloat32(device);
            using var loaded = Sd15CheckpointLoader.Load(file, plan, 16L * 1024 * 1024 * 1024, cancellationToken);
            using var cpu = loaded.CreateUnet(); using var model = cpu.To(device, cancellationToken);
            if (requested != "cpu") { cpu.Dispose(); loaded.Dispose(); }
            using var scope = NewDisposeScope(); using var enabled = set_grad_enabled(true);
            Tensor Noise(long[] shape, ulong seed) { using var noise = NativeMath.CpuNoise(shape, seed, cancellationToken); return noise.to(device); }
            using var input = Noise([1, 4, 8, 8], 511); using var context = Noise([1, 3, 768], 512);
            using var target = Noise([1, 4, 8, 8], 513); using var time = tensor(new[] { 17.25f }, device: device);
            using var adapters = new SdTrainableAdapterSet(model.Config, 2, 317, device, cancellationToken: cancellationToken);
            var leaves = adapters.Patches.SelectMany(p => p.Value.Parameters.Select((v, i) => (Name: p.Key + "/" + i, Value: v, IsAlpha: p.Value is TrainableLoraPatch l && ReferenceEquals(v, l.AlphaParameter)))).ToArray();
            var initialHashes = leaves.ToDictionary(p => p.Name, p => Hash(p.Value));
            using var baseline = model.Forward(input, time, context, cancellationToken); string baseHash = Hash(baseline);
            using var optimizer = new LoraTrainingOptimizer(adapters.Patches.Values, "SGD", .01);
            var steps = new List<object>(); int lastNonzeroAlpha = 0;
            for (int step = 0; step < 2; step++)
            {
                cancellationToken.ThrowIfCancellationRequested(); using var iteration = NewDisposeScope();
                progress.WriteLine($"SD all-adapter training: {step + 1}/2, {adapters.Patches.Count} targets, {leaves.Length} leaves on {requested}");
                using var predicted = model.ForwardForTraining(input, time, context, adapters.Patches, maxPatchedWeightBytes: 4L * 1024 * 1024 * 1024, cancellationToken);
                if (step == 0 && Hash(predicted) != baseHash) throw new InvalidOperationException("Zero-initialized adapters changed the initial prediction.");
                using var loss = TrainingLoss.Calculate("MSE", predicted, target); optimizer.Accumulate(loss, cancellationToken);
                int finite = 0, nonzero = 0, alphaNonzero = 0;
                foreach (var leaf in leaves)
                {
                    using var gradient = leaf.Value.grad ?? throw new InvalidOperationException("Missing adapter gradient: " + leaf.Name);
                    if (!gradient.isfinite().all().item<bool>()) throw new ArithmeticException("Nonfinite adapter gradient: " + leaf.Name);
                    finite++; if (gradient.abs().sum().item<float>() > 0) { nonzero++; if (leaf.IsAlpha) alphaNonzero++; }
                }
                optimizer.Step(cancellationToken); lastNonzeroAlpha = alphaNonzero;
                steps.Add(new { step = step + 1, loss = loss.item<float>(), finiteGradientLeaves = finite, nonzeroGradientLeaves = nonzero, nonzeroAlphaGradients = alphaNonzero });
            }
            using var unchanged = model.Forward(input, time, context, cancellationToken);
            if (Hash(unchanged) != baseHash) throw new InvalidOperationException("Base model changed.");
            if (lastNonzeroAlpha == 0) throw new InvalidOperationException("No alpha gradient reached the second update.");
            int changed = leaves.Count(p => Hash(p.Value) != initialHashes[p.Name]);
            if (changed == 0) throw new InvalidOperationException("No adapter parameter changed.");
            string json = JsonSerializer.Serialize(new
            {
                status = "ok", familyQualified = false, trainingNodeQualified = false, checkpoint = Path.GetFileName(checkpoint), modelSha256,
                checkpointBytes = file.FileSizeBytes, backend = requested, dtype = "Float32", tf32Allowed = false, seed = 317, rank = 2,
                optimizer = "SGD", learningRate = .01, targetCount = adapters.Patches.Count, parameterCount = leaves.Length,
                alphaParameters = leaves.Count(p => p.IsAlpha), adapterParameterBytes = adapters.ParameterBytes, changedParameterCount = changed,
                adapters.InitialCpuRandomStateSha256, adapters.InitialDeviceRandomStateSha256, steps, baseUnchanged = true,
                inputRecipe = new { shape = new[] { 1, 4, 8, 8 }, inputSeed = 511, contextShape = new[] { 1, 3, 768 }, contextSeed = 512, targetSeed = 513, timestep = 17.25 },
                elapsedSeconds = watch.Elapsed.TotalSeconds,
                scope = "Real SD1.5 all-target ordinary LoRA/BiasDiff/alpha gradients with synthetic miniature raw-prediction inputs. Metadata-only diagnostic; no image-dataset, complete node, serialization/reload of mixed adapters, pretrained source-gradient or platform qualification."
            }, new JsonSerializerOptions { WriteIndented = true });
            using var stream = new FileStream(report, FileMode.CreateNew, FileAccess.Write, FileShare.None); using var writer = new StreamWriter(stream); writer.Write(json);
            output.WriteLine(json); return 0;
        }
        catch (Exception error) { output.WriteLine(JsonSerializer.Serialize(new { status = "error", error = error.GetType().Name, error.Message })); return 1; }
    }
    private static string Hash(Tensor value)
    {
        using var scope = NewDisposeScope(); return Convert.ToHexStringLower(SHA256.HashData(value.detach().cpu().contiguous().bytes));
    }
}
