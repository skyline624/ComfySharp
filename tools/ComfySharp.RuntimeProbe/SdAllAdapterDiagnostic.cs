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
                if (i + 1 >= args.Length || args[i] is not ("--checkpoint" or "--report" or "--device" or "--adapter" or "--bypass" or "--resume" or "--algorithm" or "--rank" or "--resume-roundtrip"))
                    throw new ArgumentException("Usage: sd-all-adapter-train --checkpoint FILE --report NEW.json --device cpu|cuda:0 [--algorithm LoRA|LoHa|LoKr --rank 1..1024 --adapter NEW.safetensors --bypass true|false --resume EXISTING.safetensors --resume-roundtrip true|false]");
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
            if (algorithm is not ("LoRA" or "LoHa" or "LoKr")) throw new NotSupportedException("Select LoRA, LoHa or LoKr.");
            if (!int.TryParse(options.GetValueOrDefault("--rank","2"), System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int rank) || rank<1 || rank>1024)
                throw new ArgumentException("Rank must be an integer from 1 to 1024.");
            if (!bool.TryParse(options.GetValueOrDefault("--resume-roundtrip","false"),out bool resumeRoundtrip))throw new ArgumentException("Resume roundtrip must be true or false.");
            if(resumeRoundtrip&&(algorithm!="LoKr"||bypassMode||options.ContainsKey("--resume")))throw new ArgumentException("Memory resume roundtrip currently qualifies fresh ordinary LoKr only.");
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
            using var adapters = new SdTrainableAdapterSet(model.Config, rank, 317, device, cancellationToken: cancellationToken, existing:existingFile, algorithm:algorithm);
            if(bypassMode&&adapters.Patches.Values.Any(p=>p is TrainableLohaPatch))
                throw new NotSupportedException("Source trainable LoHa has no bypass; use ordinary training for these targets.");
            var leaves = adapters.Patches.SelectMany(p => p.Value.Parameters.Select((v, i) => (Name: p.Key + "/" + i, Value: v,
                IsAlpha: p.Value is TrainableLoraPatch l && ReferenceEquals(v, l.AlphaParameter) || p.Value is TrainableLohaPatch h && ReferenceEquals(v,h.NamedParameters["alpha"]) || p.Value is TrainableLokrPatch k && ReferenceEquals(v,k.NamedParameters["alpha"]),
                ExpectsNullGradient: p.Value is TrainableLohaPatch loha && ReferenceEquals(v,loha.NamedParameters["alpha"]) || p.Value is TrainableLokrPatch lokr && lokr.NamedParameters.ContainsKey("lokr_w1") && lokr.NamedParameters.ContainsKey("lokr_w2") && ReferenceEquals(v,lokr.NamedParameters["alpha"])))).ToArray();
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
                        if (gradient is not null) throw new InvalidOperationException("Source-inactive adapter alpha unexpectedly received a gradient.");
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
            object? trainingResumeRoundtrip = null;
            if (bypassMode || existingFile is not null || algorithm is "LoHa" or "LoKr")
            {
                using var trained = model.ForwardForTraining(input,time,context,adapters.Patches,4L*1024*1024*1024,cancellationToken,bypassMode);
                // LoKr's transposed ND linear selects mm or bmm from leaf metadata,
                // even under no_grad. Compare reloads in the same frozen mode;
                // record the training-mode difference without hiding it or changing tolerances.
                using var reloadReference=algorithm=="LoKr"&&bypassMode
                    ?FrozenAdapterEvaluation.Run(adapters.Patches.Values,()=>model.ForwardTrainingDiagnostic(input,time,context,adapters.Patches,4L*1024*1024*1024,cancellationToken,bypassMode,false))
                    :trained.alias();
                using var snapshot = LoraTrainingState.Capture(adapters.Patches,ScalarType.Float32,cancellationToken:cancellationToken);
                using var source = new NativeLoraTensorSource(snapshot.Tensors,cancellationToken:cancellationToken);
                var snapshotPlan = LoraFileLoader.Inspect(source,LoraModelAliases.ForUnet(model.Config),cancellationToken:cancellationToken);
                using var frozen = LoraFileLoader.Load(source,snapshotPlan,cancellationToken:cancellationToken);
                using var inference = bypassMode ? frozen.ApplyBypassTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:cancellationToken)
                    : frozen.ApplyTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:cancellationToken);
                using var predicted = inference.Forward(input,time,context,cancellationToken);
                if (Hash(reloadReference) != Hash(predicted))
                {
                    progress.WriteLine(JsonSerializer.Serialize(new{phase="in-memory-reload-mismatch",algorithm,bypassMode,
                        trainedSha256=Hash(trained),referenceSha256=Hash(reloadReference),reloadedSha256=Hash(predicted),maxAbsoluteError=(reloadReference-predicted).abs().max().item<float>()}));
                    var observed=new Dictionary<string,float[]>(StringComparer.Ordinal);var comparisons=new List<object>();
                    var layers=new Dictionary<string,(string Input,string Base,float[] Output)>(StringComparer.Ordinal);
                    var layerDifferences=new List<object>();
                    static bool Readable(Tensor value)=>value.device_type==DeviceType.CPU&&value.is_contiguous();
                    static string BorrowedHash(Tensor value)=>Convert.ToHexStringLower(SHA256.HashData(value.bytes));
                    model.BypassDiagnosticObserver=(name,x,y,z)=>
                    {
                        if(Readable(x)&&Readable(y)&&Readable(z))layers[name]=(BorrowedHash(x),BorrowedHash(y),z.data<float>().ToArray());
                    };
                    inference.BypassDiagnosticObserver=(name,x,y,z)=>
                    {
                        if(layerDifferences.Count>=8||!Readable(x)||!Readable(y)||!Readable(z)||!layers.TryGetValue(name,out var expected))return;
                        var actual=z.data<float>().ToArray();if(actual.Length!=expected.Output.Length)throw new InvalidOperationException("Diagnostic output shape changed.");
                        int differing=0;double maximum=0;
                        for(int i=0;i<actual.Length;i++){if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected.Output[i]))differing++;maximum=Math.Max(maximum,Math.Abs((double)actual[i]-expected.Output[i]));}
                        if(differing>0)layerDifferences.Add(new{name,inputExact=BorrowedHash(x)==expected.Input,baseOutputExact=BorrowedHash(y)==expected.Base,inputShape=x.shape,outputShape=z.shape,differing,maxAbsoluteError=maximum});
                    };
                    model.DiagnosticObserver=(name,value)=>{if(value.is_contiguous())observed[name]=value.data<float>().ToArray();};
                    inference.DiagnosticObserver=(name,value)=>
                    {
                        if(!value.is_contiguous()||!observed.TryGetValue(name,out var expected))return;
                        var actual=value.data<float>().ToArray();int differing=0;double maximum=0;
                        for(int i=0;i<actual.Length;i++){if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))differing++;maximum=Math.Max(maximum,Math.Abs((double)actual[i]-expected[i]));}
                        comparisons.Add(new{name,elements=actual.Length,differing,maxAbsoluteError=maximum});
                    };
                    try
                    {
                        using var traceTraining=model.ForwardForTraining(input,time,context,adapters.Patches,4L*1024*1024*1024,cancellationToken,bypassMode);
                        using var traceInference=inference.Forward(input,time,context,cancellationToken);
                        progress.WriteLine(JsonSerializer.Serialize(new{phase="reload-stage-comparison",stages=comparisons,
                            observedTrainingSha256=Hash(traceTraining),observedInferenceSha256=Hash(traceInference)}));
                        progress.WriteLine(JsonSerializer.Serialize(new{phase="first-bypass-layer-differences",capturedLayers=layers.Count,layers=layerDifferences}));
                        model.BypassDiagnosticObserver=null;inference.BypassDiagnosticObserver=null;
                        model.DiagnosticObserver=null;
                        using var noAutograd=model.ForwardTrainingDiagnostic(input,time,context,adapters.Patches,4L*1024*1024*1024,cancellationToken,bypassMode,false);
                        progress.WriteLine(JsonSerializer.Serialize(new{phase="identical-training-owners-without-autograd",sha256=Hash(noAutograd),
                            matchesInference=Hash(noAutograd)==Hash(predicted),maxAbsoluteError=(noAutograd-predicted).abs().max().item<float>()}));
                        var savedFlags=adapters.Patches.Values.SelectMany(p=>p.Parameters).Select(value=>(Value:value,Enabled:value.requires_grad)).ToArray();
                        try
                        {
                            foreach(var item in savedFlags)item.Value.requires_grad_(false);
                            using var frozenOwners=model.ForwardTrainingDiagnostic(input,time,context,adapters.Patches,4L*1024*1024*1024,cancellationToken,bypassMode,false);
                            progress.WriteLine(JsonSerializer.Serialize(new{phase="identical-training-storage-with-frozen-leaves",sha256=Hash(frozenOwners),
                                matchesInference=Hash(frozenOwners)==Hash(predicted),maxAbsoluteError=(frozenOwners-predicted).abs().max().item<float>()}));
                        }
                        finally{foreach(var item in savedFlags)item.Value.requires_grad_(item.Enabled);}
                    }
                    finally{model.DiagnosticObserver=null;inference.DiagnosticObserver=null;model.BypassDiagnosticObserver=null;inference.BypassDiagnosticObserver=null;}
                    throw new InvalidOperationException("In-memory reload differs from trained prediction.");
                }
                inMemoryReload = new { targets=snapshotPlan.Bindings.Count,tensors=snapshot.Tensors.Count,predictionExact=true,predictionSha256=Hash(predicted),
                    comparison=algorithm=="LoKr"&&bypassMode?"frozen training owners versus loaded inference":"training versus loaded inference",
                    trainingPredictionSha256=Hash(trained),trainingPredictionExact=Hash(trained)==Hash(predicted),
                    trainingPredictionMaxAbsoluteDifference=(trained-predicted).abs().max().item<float>() };
                if(resumeRoundtrip)
                {
                    using var resumed=new SdTrainableAdapterSet(model.Config,rank,317,device,existing:source,algorithm:algorithm,cancellationToken:cancellationToken);
                    int resetDifferences=0,resetAlphas=0,retainedFactors=0;
                    foreach(var(name,patch)in resumed.Patches)
                    {
                        if(patch is TrainableDifferencePatch difference)
                        {
                            if(difference.Difference.count_nonzero().item<long>()!=0)throw new InvalidOperationException("Source-reset difference is nonzero.");
                            resetDifferences++;continue;
                        }
                        var lokr=patch as TrainableLokrPatch??throw new InvalidOperationException("LoKr resume selected an unexpected provider.");
                        foreach(var(key,value)in lokr.NamedParameters)
                        {
                            if(key=="alpha")
                            {
                                if(value.item<float>()!=1)throw new InvalidOperationException("Ordinary export alpha was not reset by source factory.");
                                resetAlphas++;continue;
                            }
                            if(Hash(value)!=Hash(snapshot.Tensors["diffusion_model."+name[..^7]+"."+key]))throw new InvalidOperationException("Resume changed an existing LoKr factor.");
                            retainedFactors++;
                        }
                    }
                    var resumedLeaves=resumed.Patches.SelectMany(p=>p.Value.Parameters.Select(v=>(Name:p.Key,Value:v,
                        IsAlpha:p.Value is TrainableLokrPatch k&&ReferenceEquals(v,k.NamedParameters["alpha"])))).ToArray();
                    var beforeResume=resumedLeaves.Select(p=>Hash(p.Value)).ToArray();var resumeSteps=new List<object>();
                    using var resumedOptimizer=new LoraTrainingOptimizer(resumed.Patches.Values,"SGD",.01);
                    for(int step=0;step<2;step++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();using var iteration=NewDisposeScope();
                        progress.WriteLine($"SD LoKr memory resume: {step+1}/2, {resumed.Patches.Count} targets on {requested}");
                        using var resumedPrediction=model.ForwardForTraining(input,time,context,resumed.Patches,4L*1024*1024*1024,cancellationToken);
                        using var resumedLoss=TrainingLoss.Calculate("MSE",resumedPrediction,target);resumedOptimizer.Accumulate(resumedLoss,cancellationToken);
                        int finite=0,nonzero=0,nullAlpha=0;
                        foreach(var leaf in resumedLeaves)
                        {
                            using var gradient=leaf.Value.grad;
                            if(leaf.IsAlpha){if(gradient is not null)throw new InvalidOperationException("Direct resumed LoKr alpha acquired a gradient.");nullAlpha++;continue;}
                            if(gradient is null||!gradient.isfinite().all().item<bool>())throw new ArithmeticException("Invalid resumed gradient: "+leaf.Name);
                            finite++;if(gradient.abs().sum().item<float>()>0)nonzero++;
                        }
                        resumedOptimizer.Step(cancellationToken);resumeSteps.Add(new{step=step+1,loss=resumedLoss.item<float>(),finiteGradientLeaves=finite,nonzeroGradientLeaves=nonzero,nullAlphaGradients=nullAlpha});
                    }
                    int changedAfterResume=resumedLeaves.Where((leaf,index)=>Hash(leaf.Value)!=beforeResume[index]).Count();
                    if(changedAfterResume==0)throw new InvalidOperationException("Resumed training did not update any parameter.");
                    using var resumedOutput=model.ForwardForTraining(input,time,context,resumed.Patches,4L*1024*1024*1024,cancellationToken);
                    using var resumedSnapshot=LoraTrainingState.Capture(resumed.Patches,ScalarType.Float32,cancellationToken:cancellationToken);
                    using var resumedSource=new NativeLoraTensorSource(resumedSnapshot.Tensors,cancellationToken:cancellationToken);
                    var resumedPlan=LoraFileLoader.Inspect(resumedSource,LoraModelAliases.ForUnet(model.Config),cancellationToken:cancellationToken);
                    using var resumedFrozen=LoraFileLoader.Load(resumedSource,resumedPlan,cancellationToken:cancellationToken);
                    using var resumedModel=resumedFrozen.ApplyTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:cancellationToken);
                    using var resumedReload=resumedModel.Forward(input,time,context,cancellationToken);
                    if(Hash(resumedOutput)!=Hash(resumedReload))throw new InvalidOperationException("Resumed trained prediction differs after inference reload.");
                    using var stillUnchanged=model.Forward(input,time,context,cancellationToken);
                    if(Hash(stillUnchanged)!=baseHash)throw new InvalidOperationException("Resumed training modified the base model.");
                    trainingResumeRoundtrip=new{source="in-memory native factors",resumedTargets=resumed.ResumedTargets.Count,retainedFactors,resetDifferences,resetAlphas,
                        ignoredKeys=resumed.IgnoredExistingKeys.Count,parameterBytes=resumed.ParameterBytes,changedParameterCount=changedAfterResume,steps=resumeSteps,
                        resumed.InitialCpuRandomStateSha256,resumed.InitialDeviceRandomStateSha256,baseUnchanged=true,reloadPredictionExact=true,predictionSha256=Hash(resumedReload)};
                }
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
                    else if(patch is TrainableLokrPatch lokr)
                        foreach(var (key,value) in lokr.NamedParameters)parameterHashes.Add(prefix+"."+key,Hash(value));
                }
                string trainedPrediction,referencePrediction;
                using(var prediction=model.ForwardForTraining(input,time,context,adapters.Patches,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken,bypassMode))trainedPrediction=Hash(prediction);
                if(algorithm=="LoKr"&&bypassMode)
                {
                    using var reference=FrozenAdapterEvaluation.Run(adapters.Patches.Values,()=>model.ForwardTrainingDiagnostic(input,time,context,adapters.Patches,4L*1024*1024*1024,cancellationToken,bypassMode,false));
                    referencePrediction=Hash(reference);
                }
                else referencePrediction=trainedPrediction;
                LoraTrainingFile.SaveTargetsNew(adapterPath,adapters.Patches,maxFactorBytes:adapters.ParameterBytes,cancellationToken:cancellationToken);
                using var adapterFile=new SafeTensorFile(adapterPath);
                var adapterPlan=LoraFileLoader.Inspect(adapterFile,LoraModelAliases.ForUnet(model.Config),cancellationToken:cancellationToken);
                if(adapterPlan.Bindings.Count!=adapters.Patches.Count)throw new InvalidDataException("Export lost adapter targets.");
                using var snapshots=LoraFileLoader.Load(adapterFile,adapterPlan,cancellationToken:cancellationToken);
                using var baked=bypassMode ? snapshots.ApplyBypassTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:cancellationToken)
                    : snapshots.ApplyTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:cancellationToken);
                using var reloaded=baked.Forward(input,time,context,cancellationToken);
                if(Hash(reloaded)!=referencePrediction)throw new InvalidOperationException("Reloaded mixed adapter prediction differs from its reference evaluation.");
                export=new{adapter=Path.GetFileName(adapterPath),adapterSha256=adapterFile.ComputeSha256(cancellationToken),bytes=adapterFile.FileSizeBytes,
                    tensors=adapterFile.Tensors.Count,targets=adapterPlan.Bindings.Count,parameterHashes,predictionSha256=referencePrediction,reloadPredictionExact=true,
                    trainingPredictionSha256=trainedPrediction,trainingPredictionExact=trainedPrediction==referencePrediction};
                progress.WriteLine("Mixed adapter exported and reloaded: all targets retained, prediction exactly matches its reference evaluation.");
            }
            string json = JsonSerializer.Serialize(new
            {
                status = "ok", familyQualified = false, trainingNodeQualified = false, checkpoint = Path.GetFileName(checkpoint), modelSha256,
                checkpointBytes = file.FileSizeBytes, backend = requested, dtype = "Float32", tf32Allowed = false, seed = 317, rank,
                algorithm, optimizer = "SGD", learningRate = .01, targetCount = adapters.Patches.Count, parameterCount = leaves.Length,
                alphaParameters = leaves.Count(p => p.IsAlpha), adapterParameterBytes = adapters.ParameterBytes, changedParameterCount = changed,
                adapters.InitialCpuRandomStateSha256, adapters.InitialDeviceRandomStateSha256, steps, baseUnchanged = true, export, bypassMode, inMemoryReload, trainingResumeRoundtrip,
                resume = existingFile is null ? null : new { fileSha256=resumeSha256, factorTargets=adapters.ResumedTargets.Count,
                    ignoredKeys=adapters.IgnoredExistingKeys.Count, rules="Frozen weight factory only: norm/bias reset and module.weight.alpha lookup. Filename step counter is tested separately, not used by this diagnostic." },
                inputRecipe = new { shape = new[] { 1, 4, 8, 8 }, inputSeed = 511, contextShape = new[] { 1, 3, 768 }, contextSeed = 512, targetSeed = 513, timestep = 17.25 },
                elapsedSeconds = watch.Elapsed.TotalSeconds,
                scope = "Real SD1.5 all-target selected adapter/BiasDiff gradients with synthetic miniature raw-prediction inputs. LoHa and direct LoKr alpha have no source gradient. Bypass mode and optional export are reported separately. No image-dataset, complete node, pretrained source-gradient or platform qualification."
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
