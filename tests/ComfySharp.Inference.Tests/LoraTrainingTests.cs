using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraTrainingTests
{
    private static Tensor Read(JsonElement value) => tensor(value.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray(),
        value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray()).clone();
    private static void Compare(Tensor actual, JsonElement expected)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()), actual.shape);
        float[] values = expected.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        float[] received = actual.data<float>().ToArray();
        for (int i = 0; i < values.Length; i++) Assert.InRange(Math.Abs((double)received[i] - values[i]), 0, 3e-5 + 3e-5 * Math.Abs(values[i]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Full_graph_gradients_and_two_SGD_updates_match_the_frozen_source(bool linear)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope(); using var grad = set_grad_enabled(true);
            using var json = JsonDocument.Parse(GetType().Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora-training.reference.json")!);
            var row = json.RootElement.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("linearProjection").GetBoolean() == linear);
            var config = new SdUnetConfig(32, 16, linear ? SdAttentionHeadMode.FixedSize : SdAttentionHeadMode.FixedCount, linear ? 8 : 4, linear);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            var patches = new Dictionary<string, TrainableLoraPatch>(StringComparer.Ordinal);
            try
            {
                foreach (var property in row.GetProperty("initial").EnumerateObject())
                    patches.Add(property.Name, new(Read(property.Value.GetProperty("up")), Read(property.Value.GetProperty("down")), property.Value.GetProperty("alpha").GetDouble()));
                var latent = Read(row.GetProperty("latent")); var context = Read(row.GetProperty("context")); var times = Read(row.GetProperty("timesteps")); var target = Read(row.GetProperty("target"));
                using var original = model.Forward(latent, times, context); Assert.False(original.requires_grad);
                foreach (var step in row.GetProperty("steps").EnumerateArray())
                {
                    using var iteration = NewDisposeScope();
                    foreach (var patch in patches.Values)
                    {
                        using var u = patch.Up.grad; using var d = patch.Down.grad;
                        u?.zero_(); d?.zero_();
                    }
                    using var prediction = model.ForwardForTraining(latent, times, context, patches);
                    Assert.True(prediction.requires_grad); Compare(prediction, step.GetProperty("output"));
                    using var loss = (prediction - target).square().mean(); loss.backward();
                    Assert.InRange(Math.Abs(loss.item<float>() - step.GetProperty("loss").GetDouble()), 0, 3e-5);
                    foreach (var (name, patch) in patches)
                    {
                        using var u = patch.Up.grad!; using var d = patch.Down.grad!;
                        Assert.NotNull(u); Assert.NotNull(d);
                        Compare(u, step.GetProperty("gradients").GetProperty(name).GetProperty("up"));
                        Compare(d, step.GetProperty("gradients").GetProperty(name).GetProperty("down"));
                        using (var noGrad = no_grad())
                        {
                            patch.Up.add_(u, alpha: -row.GetProperty("learningRate").GetDouble());
                            patch.Down.add_(d, alpha: -row.GetProperty("learningRate").GetDouble());
                        }
                        Compare(patch.Up, step.GetProperty("updated").GetProperty(name).GetProperty("up"));
                        Compare(patch.Down, step.GetProperty("updated").GetProperty(name).GetProperty("down"));
                    }
                    foreach (string name in UnetWeightSchema.Describe(config).Keys) Assert.Null(bank.GetTensor(name).grad);
                }
                using var originalAgain = model.Forward(latent, times, context);
                Assert.Equal(original.data<float>().ToArray(), originalAgain.data<float>().ToArray());
                var frozen = patches.ToDictionary(p => p.Key, p => p.Value.Snapshot(), StringComparer.Ordinal);
                try
                {
                    using var baked = model.WithLora(frozen);
                    using var trained = model.ForwardForTraining(latent, times, context, patches);
                    using var inference = baked.Forward(latent, times, context);
                    using var error = (trained - inference).abs().max(); Assert.InRange(error.item<float>(), 0, 3e-5);
                    Assert.False(inference.requires_grad);
                    string path = Path.Combine(Path.GetTempPath(), "comfysharp-trained-" + Guid.NewGuid().ToString("N") + ".safetensors");
                    try
                    {
                        LoraTrainingFile.SaveNew(path, patches.ToDictionary(p => "diffusion_model." + p.Key[..^7], p => p.Value, StringComparer.Ordinal));
                        using var file = new SafeTensorFile(path);
                        var plan = LoraFileLoader.Inspect(file, LoraModelAliases.ForUnet(config));
                        using var reloaded = LoraFileLoader.Load(file, plan); using var reloadedModel = reloaded.ApplyTo(model);
                        file.Dispose();
                        using var reloadedPrediction = reloadedModel.Forward(latent, times, context);
                        Assert.Equal(inference.data<float>().ToArray(), reloadedPrediction.data<float>().ToArray());
                    }
                    finally { if (File.Exists(path)) File.Delete(path); }
                }
                finally { foreach (var patch in frozen.Values) patch.Dispose(); }
            }
            finally { foreach (var patch in patches.Values) patch.Dispose(); }
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Graph_survives_base_disposal_and_owned_leaf_snapshots_do_not_mutate_caller_factors()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope(); using var grad = set_grad_enabled(true);
            var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            var up = full(new long[] { 32, 1 }, .01f); var down = full(new long[] { 1, 36 }, .01f);
            using var patch = new TrainableLoraPatch(up, down, 1); using var retained = patch.Retain();
            up.fill_(10); down.fill_(10); Assert.All(patch.Up.data<float>().ToArray(), value => Assert.Equal(.01f, value));
            using var prediction = model.ForwardForTraining(ones(new long[] { 1, 4, 8, 8 }), tensor(new[] { 17.25f }), ones(new long[] { 1, 3, 16 }),
                new Dictionary<string, TrainableLoraPatch> { ["input_blocks.0.0.weight"] = patch });
            model.Dispose(); bank.Dispose(); patch.Dispose();
            using var loss = prediction.square().mean(); loss.backward();
            using var gradient = retained.Up.grad!; using var norm = gradient.abs().sum();
            Assert.True(norm.item<float>() > 0); Assert.True(retained.Up.is_leaf);
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Cancellation_and_rejected_targets_release_graphs_and_restore_callers_grad_mode()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope(); using var noGrad = no_grad(); using var cancel = new CancellationTokenSource();
            var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            using var patch = new TrainableLoraPatch(ones(new long[] { 32, 1 }), ones(new long[] { 1, 36 }), 1);
            var patches = new Dictionary<string, TrainableLoraPatch> { ["input_blocks.0.0.weight"] = patch };
            var x = ones(new long[] { 1, 4, 8, 8 }); var t = tensor(new[] { 17.25f }); var c = ones(new long[] { 1, 3, 16 });
            Assert.Throws<NotSupportedException>(() => model.ForwardForTraining(x, t, c, patches, maxPatchedWeightBytes: 0));
            Assert.Throws<ArgumentException>(() => model.ForwardForTraining(x, t, c, new Dictionary<string, TrainableLoraPatch>()));
            Assert.Throws<InvalidDataException>(() => model.ForwardForTraining(x, t, c, new Dictionary<string, TrainableLoraPatch> { ["missing"] = patch }));
            model.DiagnosticObserver = (_, _) => cancel.Cancel();
            Assert.ThrowsAny<OperationCanceledException>(() => model.ForwardForTraining(x, t, c, patches, cancellationToken: cancel.Token));
            Assert.False(is_grad_enabled()); Assert.Null(patch.Up.grad); Assert.Null(patch.Down.grad);
            model.DiagnosticObserver = null;
            using var result = model.ForwardForTraining(x, t, c, patches); Assert.True(result.requires_grad); Assert.False(is_grad_enabled());
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Export_refuses_existing_files_nonfinite_factors_and_budget_without_leaving_partial_files()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        string directory = Path.Combine(Path.GetTempPath(), "comfysharp-lora-export-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var scope = NewDisposeScope(); using var noGrad = no_grad();
            Directory.CreateDirectory(directory); string path = Path.Combine(directory, "adapter.safetensors");
            using var patch = new TrainableLoraPatch(ones(new long[] { 2, 1 }), ones(new long[] { 1, 3 }), 1);
            var aliases = new Dictionary<string, TrainableLoraPatch> { ["diffusion_model.layer"] = patch };
            Assert.Throws<NotSupportedException>(() => LoraTrainingFile.SaveNew(path, aliases, maxFactorBytes: 0));
            Assert.ThrowsAny<OperationCanceledException>(() => LoraTrainingFile.SaveNew(path, aliases, cancellationToken: new(true)));
            patch.Up.fill_(float.NaN);
            Assert.Throws<ArithmeticException>(() => LoraTrainingFile.SaveNew(path, aliases));
            Assert.Empty(Directory.EnumerateFiles(directory));
            File.WriteAllBytes(path, [1, 2, 3]);
            Assert.Throws<IOException>(() => LoraTrainingFile.SaveNew(path, aliases));
            Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(path));
            Assert.Single(Directory.EnumerateFiles(directory));
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
