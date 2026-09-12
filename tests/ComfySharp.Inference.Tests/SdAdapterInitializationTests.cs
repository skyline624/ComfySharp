using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdAdapterInitializationTests
{
    private static Dictionary<string, Tensor> Named(TrainableWeightPatch patch) => patch switch
    {
        TrainableLoraPatch lora => new() { ["alpha"] = lora.AlphaParameter!, ["lora_up.weight"] = lora.Up, ["lora_down.weight"] = lora.Down },
        TrainableDifferencePatch difference => new() { ["bias"] = difference.Difference },
        _ => throw new InvalidOperationException()
    };
    private static string Hash(Tensor tensor) => Convert.ToHexStringLower(SHA256.HashData(tensor.bytes));
    private static Tensor Read(JsonElement e) => tensor(e.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray(), e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray()).clone();
    private static void Near(Tensor actual, JsonElement expected)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()), actual.shape);
        var values = expected.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var observed = actual.data<float>().ToArray();
        for (int i = 0; i < values.Length; i++) Assert.True(float.IsFinite(observed[i]) && Math.Abs(observed[i] - values[i]) <= 3e-5 + 3e-5 * Math.Abs(values[i]), $"Element {i}: {observed[i]} vs {values[i]}");
    }

    [Theory] [InlineData(0)] [InlineData(1)]
    public void All_targets_initialization_rng_gradients_and_updates_follow_frozen_source(int caseIndex)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope();
            using var trace = SdTrainingTrace.Open(caseIndex);
            using var corpus = TrainingReferenceCorpus.Load("adapters");
            var row = corpus.RootElement.GetProperty("cases")[caseIndex]; bool linear = row.GetProperty("linearProjection").GetBoolean();
            var config = new SdUnetConfig(32, 16, linear ? SdAttentionHeadMode.FixedSize : SdAttentionHeadMode.FixedCount, linear ? 8 : 4, linear);
            using var global = manual_seed(771); using var originalState = global.get_state();
            using var adapters = new SdTrainableAdapterSet(config, 2, 317, CPU);
            using var afterState = global.get_state(); Assert.Equal(Hash(originalState), Hash(afterState));
            Assert.Equal(row.GetProperty("randomStateSha256").GetString(), adapters.InitialCpuRandomStateSha256);
            Assert.Equal(686, adapters.Patches.Count);
            Assert.Equal(row.GetProperty("parameterCount").GetInt32(), adapters.Patches.Values.Sum(p => p.Parameters.Count));
            Assert.Equal(row.GetProperty("initial").EnumerateArray().Select(t => t.GetProperty("target").GetString()), adapters.Patches.Keys);
            foreach (var target in row.GetProperty("initial").EnumerateArray())
            {
                string name = target.GetProperty("target").GetString()!; var parameters = Named(adapters.Patches[name]);
                foreach (var parameter in target.GetProperty("parameters").EnumerateArray())
                {
                    var value = parameters[parameter.GetProperty("name").GetString()!];
                    Assert.Equal(parameter.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()), value.shape);
                    Assert.True(value.requires_grad); Assert.Equal(parameter.GetProperty("sha256").GetString(), Hash(value));
                }
            }
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            trace?.Weights(bank);
            model.DiagnosticObserver = trace is null ? null : trace.Capture;
            model.FineDiagnosticObserver = trace is null ? null : trace.Capture;
            using var latent = Read(row.GetProperty("latent")); using var time = Read(row.GetProperty("times")); using var context = Read(row.GetProperty("context")); using var targetValues = Read(row.GetProperty("target"));
            using var baseline = model.Forward(latent, time, context);
            trace?.Capture("baseline",baseline);
            trace?.Capture("latent",latent);trace?.Capture("times",time);trace?.Capture("context",context);
            using var optimizer = new LoraTrainingOptimizer(adapters.Patches.Values, "SGD", .01);
            int stepIndex=0;
            foreach (var step in row.GetProperty("steps").EnumerateArray())
            {
                using var iteration = NewDisposeScope();
                trace?.Phase("step-"+stepIndex++);
                using var output = model.ForwardForTraining(latent, time, context, adapters.Patches); trace?.Capture("output",output);Near(output, step.GetProperty("output"));
                using var loss = TrainingLoss.Calculate("MSE", output, targetValues); Assert.InRange(Math.Abs(loss.item<float>() - step.GetProperty("loss").GetSingle()), 0, 3e-5);
                optimizer.Accumulate(loss);
                foreach (var patch in adapters.Patches.Values) foreach (var parameter in patch.Parameters)
                {
                    using var gradient = parameter.grad; Assert.NotNull(gradient); Assert.True(gradient!.isfinite().all().item<bool>());
                }
                foreach (var target in step.GetProperty("gradients").EnumerateObject()) foreach (var expected in target.Value.EnumerateObject())
                {
                    using var gradient = Named(adapters.Patches[target.Name])[expected.Name].grad; trace?.Capture("gradient/"+target.Name+"/"+expected.Name,gradient!);Near(gradient!, expected.Value);
                }
                optimizer.Step();
                foreach (var target in step.GetProperty("updated").EnumerateObject()) foreach (var expected in target.Value.EnumerateObject()) Near(Named(adapters.Patches[target.Name])[expected.Name], expected.Value);
            }
            model.DiagnosticObserver=null;
            model.FineDiagnosticObserver=null;
            using var unchanged = model.Forward(latent, time, context); Assert.Equal(Hash(baseline), Hash(unchanged));
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Initialization_admission_cancellation_and_ownership_are_explicit()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
            Assert.Throws<NotSupportedException>(() => new SdTrainableAdapterSet(config, 2, 0, CPU, maxParameterBytes: 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new SdTrainableAdapterSet(config, 0, 0, CPU));
            Assert.Throws<OperationCanceledException>(() => new SdTrainableAdapterSet(config, 2, 0, CPU, cancellationToken: new(true)));
            using var adapters = new SdTrainableAdapterSet(config, 1, 0, CPU);
            using var retained = adapters.Patches["out.2.bias"].Retain(); adapters.Dispose();
            Assert.Throws<ObjectDisposedException>(() => adapters.Patches); Assert.True(retained.Parameters[0].requires_grad);
        }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
