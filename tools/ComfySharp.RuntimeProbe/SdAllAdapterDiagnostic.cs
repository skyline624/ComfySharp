using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

/// <summary>Explicit pretrained all-target gradient diagnostic with optional new mixed-adapter export and reload.</summary>
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
                if (i + 1 >= args.Length || args[i] is not ("--checkpoint" or "--report" or "--device" or "--adapter" or "--bypass" or "--resume" or "--algorithm"))
                    throw new ArgumentException("Usage: sd-all-adapter-train --checkpoint FILE --report NEW.json --device cpu|cuda:0 [--algorithm LoRA|LoHa --adapter NEW.safetensors --bypass true|false --resume EXISTING.safetensors]");
                options.Add(args[i], args[i + 1]);
            }
            string Required(string name) => options.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value) ? value : throw new ArgumentException("Missing " + name);
            string checkpoint = Path.GetFullPath(Required("--checkpoint")), report = Path.GetFullPath(Required("--report")), requested = Required("--device");
            string? adapterPath = options.TryGetValue("--adapter",out var adapterOption)?Path.GetFullPath(adapterOption):null;
            if(adapterPath is not null&&(File.Exists(adapterPath)||Directory.Exists(adapterPath)||!Directory.Exists(Path.GetDirectoryName(adapterPath))))throw new IOException("Adapter must be a new file in an existing directory.");
            if(adapterPath is not null&&string.Equals(adapterPath,report,StringComparison.OrdinalIgnoreCase))throw new ArgumentException("Adapter and report must use distinct paths.");
            if (File.Exists(report) || Directory.Exists(report) || !Directory.Exists(Path.GetDirectoryName(report))) throw new IOException("Report must be a new file in an existing directory.");
            if (requested is not ("cpu" or "cuda:0")) throw new ArgumentException("Select cpu or cuda:0.");
            if (!bool.TryParse(options.GetValueOrDefault("--bypass", "false"), out bool bypassMode)) throw new ArgumentException("Bypass must be true or false.");
            string algorithm = options.GetValueOrDefault("--algorithm", "LoRA");
            if (algorithm is not ("LoRA" or "LoHa")) throw new NotSupportedException("Select LoRA or LoHa.");
            cancellationToken.ThrowIfCancellationRequested();
            using var file = new SafeTensorFile(checkpoint);
            using var existingFile = options.TryGetValue("--resume",out var resumePath) ? new SafeTensorFile(Path.GetFullPath(resumePath)) : null;
            string? resumeSha256 = existingFile?.ComputeSha256(cancellationToken);
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
            using var adapters = new SdTrainableAdapterSet(model.Config, 2, 317, device, cancellationToken: cancellationToken, existing:existingFile, algorithm:algorithm);
            if(bypassMode&&adapters.Patches.Values.Any(p=>p is TrainableLohaPatch))
                throw new NotSupportedException("Source trainable LoHa has no bypass; use ordinary training for these targets.");
            var leaves = adapters.Patches.SelectMany(p => p.Value.Parameters.Select((v, i) => (Name: p.Key + "/" + i, Value: v,
                IsAlpha: p.Value is TrainableLoraPatch l && ReferenceEquals(v, l.AlphaParameter) || p.Value is TrainableLohaPatch h && ReferenceEquals(v,h.NamedParameters["alpha"]),
                ExpectsNullGradient: p.Value is TrainableLohaPatch loha && ReferenceEquals(v,loha.NamedParameters["alpha"])))).ToArray();
            var initialHashes = leaves.ToDictionary(p => p.Name, p => Hash(p.Value));
            using var baseline = model.Forward(input, time, context, cancellationToken); string baseHash = Hash(baseline);
            using var optimizer = new LoraTrainingOptimizer(adapters.Patches.Values, "SGD", .01);
            var steps = new List<object>(); int lastNonzeroAlpha = 0;
            for (int step = 0; step < 2; step++)
            {
                cancellationToken.ThrowIfCancellationRequested(); using var iteration = NewDisposeScope();
                progress.WriteLine($"SD all-adapter training: {step + 1}/2, {adapters.Patches.Count} targets, {leaves.Length} leaves on {requested}");
                using var predicted = model.ForwardForTraining(input, time, context, adapters.Patches, maxPatchedWeightBytes: 4L * 1024 * 1024 * 1024, cancellationToken, bypassMode);
                if (step == 0 && existingFile is null && Hash(predicted) != baseHash) throw new InvalidOperationException("Zero-initialized adapters changed the initial prediction.");
                using var loss = TrainingLoss.Calculate("MSE", predicted, target); optimizer.Accumulate(loss, cancellationToken);
                int finite = 0, nonzero = 0, alphaNonzero = 0, nullSourceGradients = 0;
                foreach (var leaf in leaves)
                {
                    using var gradient = leaf.Value.grad;
                    if (leaf.ExpectsNullGradient)
                    {
                        if (gradient is not null) throw new InvalidOperationException("Frozen LoHa alpha unexpectedly received a gradient.");
                        nullSourceGradients++; continue;
                    }
                    if (gradient is null) throw new InvalidOperationException("Missing adapter gradient: " + leaf.Name);
                    if (!gradient.isfinite().all().item<bool>()) throw new ArithmeticException("Nonfinite adapter gradient: " + leaf.Name);
                    finite++; if (gradient.abs().sum().item<float>() > 0) { nonzero++; if (leaf.IsAlpha) alphaNonzero++; }
                }
                optimizer.Step(cancellationToken); lastNonzeroAlpha = alphaNonzero;
                steps.Add(new { step = step + 1, loss = loss.item<float>(), finiteGradientLeaves = finite, nonzeroGradientLeaves = nonzero, nonzeroAlphaGradients = alphaNonzero, nullSourceGradients });
            }
            using var unchanged = model.Forward(input, time, context, cancellationToken);
            if (Hash(unchanged) != baseHash) throw new InvalidOperationException("Base model changed.");
            if (leaves.Any(p=>p.IsAlpha&&!p.ExpectsNullGradient) && lastNonzeroAlpha == 0) throw new InvalidOperationException("No trainable LoRA alpha gradient reached the second update.");
            int changed = leaves.Count(p => Hash(p.Value) != initialHashes[p.Name]);
            if (changed == 0) throw new InvalidOperationException("No adapter parameter changed.");
            object? inMemoryReload = null;
            if (bypassMode || existingFile is not null || algorithm == "LoHa")
            {
                using var trained = model.ForwardForTraining(input,time,context,adapters.Patches,4L*1024*1024*1024,cancellationToken,bypassMode);
                using var snapshot = LoraTrainingState.Capture(adapters.Patches,ScalarType.Float32,cancellationToken:cancellationToken);
                using var source = new NativeLoraTensorSource(snapshot.Tensors,cancellationToken:cancellationToken);
                var snapshotPlan = LoraFileLoader.Inspect(source,LoraModelAliases.ForUnet(model.Config),cancellationToken:cancellationToken);
                using var frozen = LoraFileLoader.Load(source,snapshotPlan,cancellationToken:cancellationToken);
                using var inference = bypassMode ? frozen.ApplyBypassTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:cancellationToken)
                    : frozen.ApplyTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:cancellationToken);
                using var predicted = inference.Forward(input,time,context,cancellationToken);
                if (Hash(trained) != Hash(predicted)) throw new InvalidOperationException("In-memory reload differs from trained prediction.");
                inMemoryReload = new { targets=snapshotPlan.Bindings.Count,tensors=snapshot.Tensors.Count,predictionExact=true,predictionSha256=Hash(predicted) };
            }
            object? export=null;
            if(adapterPath is not null)
            {
                var parameterHashes=new Dictionary<string,string>(StringComparer.Ordinal);
                foreach(var (name,patch) in adapters.Patches)
                {
                    bool bias=name.EndsWith(".bias",StringComparison.Ordinal);
                    string prefix="diffusion_model."+name[..^(bias?5:7)];
                    if(patch is TrainableDifferencePatch diff)parameterHashes.Add(prefix+(bias?".diff_b":".diff"),Hash(diff.Difference));
                    else if(patch is TrainableLoraPatch lora)
                    {
                        parameterHashes.Add(prefix+".lora_up.weight",Hash(lora.Up));parameterHashes.Add(prefix+".lora_down.weight",Hash(lora.Down));
                        parameterHashes.Add(prefix+".alpha",Hash(lora.AlphaParameter!));
                    }
                    else if(patch is TrainableLohaPatch loha)
                        foreach(var (key,value) in loha.NamedParameters)parameterHashes.Add(prefix+"."+key,Hash(value));
                }
                string trainedPrediction;
                using(var noGrad=no_grad())
                using(var prediction=model.ForwardForTraining(input,time,context,adapters.Patches,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken,bypassMode))trainedPrediction=Hash(prediction);
                LoraTrainingFile.SaveTargetsNew(adapterPath,adapters.Patches,maxFactorBytes:adapters.ParameterBytes,cancellationToken:cancellationToken);
                using var adapterFile=new SafeTensorFile(adapterPath);
                var adapterPlan=LoraFileLoader.Inspect(adapterFile,LoraModelAliases.ForUnet(model.Config),cancellationToken:cancellationToken);
                if(adapterPlan.Bindings.Count!=adapters.Patches.Count)throw new InvalidDataException("Export lost adapter targets.");
                using var snapshots=LoraFileLoader.Load(adapterFile,adapterPlan,cancellationToken:cancellationToken);
                using var baked=bypassMode ? snapshots.ApplyBypassTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:cancellationToken)
                    : snapshots.ApplyTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:cancellationToken);
                using var reloaded=baked.Forward(input,time,context,cancellationToken);
                if(Hash(reloaded)!=trainedPrediction)throw new InvalidOperationException("Reloaded mixed adapter prediction differs from trained parameters.");
                export=new{adapter=Path.GetFileName(adapterPath),adapterSha256=adapterFile.ComputeSha256(cancellationToken),bytes=adapterFile.FileSizeBytes,
                    tensors=adapterFile.Tensors.Count,targets=adapterPlan.Bindings.Count,parameterHashes,predictionSha256=trainedPrediction,reloadPredictionExact=true};
                progress.WriteLine("Mixed adapter exported and reloaded: all targets retained, prediction exactly matches trained parameters.");
            }
            string json = JsonSerializer.Serialize(new
            {
                status = "ok", familyQualified = false, trainingNodeQualified = false, checkpoint = Path.GetFileName(checkpoint), modelSha256,
                checkpointBytes = file.FileSizeBytes, backend = requested, dtype = "Float32", tf32Allowed = false, seed = 317, rank = 2,
                algorithm, optimizer = "SGD", learningRate = .01, targetCount = adapters.Patches.Count, parameterCount = leaves.Length,
                alphaParameters = leaves.Count(p => p.IsAlpha), adapterParameterBytes = adapters.ParameterBytes, changedParameterCount = changed,
                adapters.InitialCpuRandomStateSha256, adapters.InitialDeviceRandomStateSha256, steps, baseUnchanged = true, export, bypassMode, inMemoryReload,
                resume = existingFile is null ? null : new { fileSha256=resumeSha256, factorTargets=adapters.ResumedTargets.Count,
                    ignoredKeys=adapters.IgnoredExistingKeys.Count, rules="Frozen weight factory only: norm/bias reset and module.weight.alpha lookup. Filename step counter is tested separately, not used by this diagnostic." },
                inputRecipe = new { shape = new[] { 1, 4, 8, 8 }, inputSeed = 511, contextShape = new[] { 1, 3, 768 }, contextSeed = 512, targetSeed = 513, timestep = 17.25 },
                elapsedSeconds = watch.Elapsed.TotalSeconds,
                scope = "Real SD1.5 all-target selected adapter/BiasDiff gradients with synthetic miniature raw-prediction inputs. LoHa alpha has no source gradient. Bypass mode and optional export are reported separately. No image-dataset, complete node, pretrained source-gradient or platform qualification."
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
