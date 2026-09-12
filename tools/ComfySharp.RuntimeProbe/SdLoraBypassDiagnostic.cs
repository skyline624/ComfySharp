using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

/// <summary>Explicit CPU pretrained bypass observation with existing checkpoint/adapter files.
/// Writes only a new small JSON report; no training, generation or model-level tolerance claim.</summary>
internal static class SdLoraBypassDiagnostic
{
    internal static int Run(string[] args, TextWriter output, TextWriter progress, CancellationToken token)
    {
        try
        {
            var options = new Dictionary<string, string>(StringComparer.Ordinal);
            for (int i = 0; i < args.Length; i += 2)
            {
                if (i + 1 == args.Length || args[i] is not ("--checkpoint" or "--adapter" or "--report"))
                    throw new ArgumentException("Usage: sd-lora-bypass --checkpoint FILE --adapter FILE --report NEW.json");
                options.Add(args[i], args[i + 1]);
            }
            string Required(string name) => options.TryGetValue(name, out var value) ? Path.GetFullPath(value) : throw new ArgumentException("Missing " + name);
            string checkpoint = Required("--checkpoint"), adapterPath = Required("--adapter"), reportPath = Required("--report");
            if (File.Exists(reportPath) || Directory.Exists(reportPath) || !Directory.Exists(Path.GetDirectoryName(reportPath))) throw new IOException("Report must be new in an existing directory.");
            var watch = Stopwatch.StartNew();
            using var file = new SafeTensorFile(checkpoint); using var adapterFile = new SafeTensorFile(adapterPath);
            var plan = Sd15CheckpointLoader.Inspect(file, new() { UnclaimedTensors = Sd15UnclaimedTensorHandling.ReportAndIgnoreOutsideComponents }, token);
            string checkpointHash = file.ComputeSha256(token), adapterHash = adapterFile.ComputeSha256(token);
            NativeRuntimeBootstrap.Initialize(); set_num_threads(16); InferenceDevice.ConfigureFloat32(CPU);
            progress.WriteLine("Loading existing SD1.5 checkpoint on CPU.");
            using var loaded = Sd15CheckpointLoader.Load(file, plan, 16L * 1024 * 1024 * 1024, token);
            using var model = loaded.CreateUnet(); using var scope = NewDisposeScope();
            using var input = NativeMath.CpuNoise([1,4,8,8],511,token); using var context = NativeMath.CpuNoise([1,3,768],512,token); using var time = tensor(new[]{17.25f});
            using var baseline = model.Forward(input,time,context,token); string baselineHash = Hash(baseline);
            NativeLoraTensorSource Capture()
            {
                using var temporary = NewDisposeScope();
                if (adapterFile.Tensors.Values.Sum(v => v.End - v.Start) > 512L * 1024 * 1024) throw new NotSupportedException("Adapter diagnostic input exceeds its snapshot allowance.");
                var values = adapterFile.Tensors.Keys.ToDictionary(k=>k,k=>adapterFile.ReadTensor(k,token));
                return new(values,cancellationToken:token);
            }
            using var source = Capture(); var loadPlan = LoraFileLoader.Inspect(source,LoraModelAliases.ForUnet(model.Config),cancellationToken:token);
            using var adapters = LoraFileLoader.Load(source,loadPlan,cancellationToken:token);
            Tensor ordinary;
            progress.WriteLine("Computing ordinary adapter prediction.");
            using(var baked=adapters.ApplyTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:token))
                ordinary=baked.Forward(input,time,context,token);
            using(ordinary)
            using(var bypass=adapters.ApplyBypassTo(model,maxPatchedWeightBytes:4L*1024*1024*1024,cancellationToken:token))
            {
                adapters.Dispose(); source.Dispose(); adapterFile.Dispose();
                var hashes=new List<string>();double maxAbsolute=0,meanAbsolute=0;
                for(int iteration=0;iteration<3;iteration++)
                {
                    progress.WriteLine($"Computing bypass prediction {iteration+1}/3.");
                    using var predicted=bypass.Forward(input,time,context,token); using var iterationScope=NewDisposeScope();
                    if(!predicted.isfinite().all().item<bool>())throw new ArithmeticException("Bypass prediction is nonfinite.");
                    hashes.Add(Hash(predicted));
                    var difference=(predicted-ordinary).abs();
                    maxAbsolute=difference.max().item<float>();meanAbsolute=difference.mean().item<float>();
                }
                using var after=model.Forward(input,time,context,token);
                if(hashes.Distinct().Count()!=1||Hash(after)!=baselineHash||hashes[0]==baselineHash)
                    throw new InvalidOperationException("Bypass repeatability, base preservation or adapter effect check failed.");
                token.ThrowIfCancellationRequested();
                var report=new{status="ok",checkpointSha256=checkpointHash,adapterSha256=adapterHash,device="cpu",threads=get_num_threads(),
                    matchedTargets=loadPlan.Bindings.Count,baselineSha256=baselineHash,ordinarySha256=Hash(ordinary),bypassSha256=hashes[0],
                    repeats=hashes.Count,repeatable=true,basePredictionUnchanged=true,adapterChangesPrediction=true,
                    comparison=new{maxAbsolute,meanAbsolute,acceptance="observation_only_no_model_tolerance_acceptance"},
                    elapsedSeconds=watch.Elapsed.TotalSeconds,familyQualified=false,
                    scope="Existing pretrained SD1.5 and trained adapter; synthetic 8x8 latent and text context. Raw CPU predictions only, no generation or training rerun."};
                using var destination=new FileStream(reportPath,FileMode.CreateNew,FileAccess.Write,FileShare.None);
                JsonSerializer.Serialize(destination,report,new JsonSerializerOptions{WriteIndented=true});
                output.WriteLine(JsonSerializer.Serialize(report));
            }
            return 0;
        }
        catch(Exception exception){output.WriteLine(JsonSerializer.Serialize(new{status="error",error=exception.Message}));return 1;}
    }
    private static string Hash(Tensor value)=>Convert.ToHexStringLower(SHA256.HashData(value.bytes));
}
