using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdLoraObjectiveTests
{
    private static JsonDocument Corpus() => TrainingReferenceCorpus.Load("denoising");
    private static Tensor Read(JsonElement e) => tensor(e.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray(), e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray()).clone();
    private static void Compare(Tensor actual, JsonElement expected)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()), actual.shape);
        var values = expected.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray(); var received = actual.data<float>().ToArray();
        for (int i = 0; i < values.Length; i++) Assert.InRange(Math.Abs((double)received[i] - values[i]), 0, 3e-5 + 3e-5 * Math.Abs(values[i]));
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)]
    public void Denoised_latent_loss_and_gradients_match_frozen_training_sampler(int caseIndex)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope(); using var enabled = set_grad_enabled(true); using var corpus = Corpus();
            var row = corpus.RootElement.GetProperty("cases")[caseIndex]; bool linear = row.GetProperty("linearProjection").GetBoolean();
            var config = new SdUnetConfig(32, 16, linear ? SdAttentionHeadMode.FixedSize : SdAttentionHeadMode.FixedCount, linear ? 8 : 4, linear);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            using var denoiser = new SdDenoiser(model, linear ? SdPredictionKind.Velocity : SdPredictionKind.Epsilon);
            var patches = new Dictionary<string, TrainableLoraPatch>(StringComparer.Ordinal);
            try
            {
                foreach (var p in row.GetProperty("initial").EnumerateObject()) patches.Add(p.Name, new(Read(p.Value.GetProperty("up")), Read(p.Value.GetProperty("down")), 2.5));
                var latent = Read(row.GetProperty("latent")); var noise = Read(row.GetProperty("noise")); var sigma = Read(row.GetProperty("sigma")); var context = Read(row.GetProperty("context"));
                using var noisy = SdSamplingMath.NoiseScaling(noise, latent, sigma); noisy.requires_grad_();
                using var trainSigma = sigma.clone().requires_grad_();
                Compare(noisy, row.GetProperty("captures").GetProperty("noisy"));
                using var scaled = SdSamplingMath.ScaleInput(noisy, sigma); Compare(scaled, row.GetProperty("captures").GetProperty("scaled"));
                using var predicted = denoiser.DenoiseForTraining(noisy, trainSigma, context, patches);
                Assert.True(predicted.requires_grad); Compare(predicted, row.GetProperty("captures").GetProperty("denoised"));
                using var loss = TrainingLoss.Calculate("MSE", predicted, latent); using var backwardLoss = loss / 2; backwardLoss.backward();
                Assert.InRange(Math.Abs(loss.item<float>() - row.GetProperty("loss").GetDouble()), 0, 3e-5);
                Compare(noisy.grad!, row.GetProperty("noisyGradient")); Compare(trainSigma.grad!, row.GetProperty("sigmaGradient"));
                void CompareFactors()
                {
                    foreach (var (name, patch) in patches)
                    {
                        using var up = patch.Up.grad!; using var down = patch.Down.grad!;
                        Compare(up, row.GetProperty("gradients").GetProperty(name).GetProperty("up"));
                        Compare(down, row.GetProperty("gradients").GetProperty(name).GetProperty("down"));
                    }
                }
                CompareFactors();
                foreach (var patch in patches.Values) { patch.Up.grad = null; patch.Down.grad = null; }
                using var objective = SdLoraTrainingObjective.CalculateLoss(denoiser, latent, noise, sigma, context, patches, "MSE");
                Assert.InRange(Math.Abs(objective.item<float>() - row.GetProperty("loss").GetDouble()), 0, 3e-5);
                // The scalar's saved native graph remains sufficient after every model owner is released.
                model.Dispose(); bank.Dispose(); denoiser.Dispose(); using var normalized = objective / 2; normalized.backward(); CompareFactors();
                Compare(latent, row.GetProperty("latent")); Compare(noise, row.GetProperty("noise")); Compare(sigma, row.GetProperty("sigma")); Compare(context, row.GetProperty("context"));
                foreach (var input in new[] { latent, noise, sigma, context }) { Assert.False(input.requires_grad); Assert.Null(input.grad); }
            }
            finally { foreach (var patch in patches.Values) patch.Dispose(); }
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Borrowed_dataset_values_keep_existing_gradient_buffers_and_flags()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope(); using var enabled = set_grad_enabled(true);
        var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
        using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank); using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
        using var patch = new TrainableLoraPatch(ones([4, 1]) * .01, ones([1, 288]) * .01, 1);
        var inputs = new[] { ones([1, 4, 8, 8]), ones([1, 4, 8, 8]), tensor(new[] { .5f }), zeros([1, 3, 16]) };
        foreach (var input in inputs) { input.requires_grad_(); input.grad = full_like(input, .25); }
        using var loss = SdLoraTrainingObjective.CalculateLoss(denoiser, inputs[0], inputs[1], inputs[2], inputs[3], new Dictionary<string, TrainableLoraPatch> { ["out.2.weight"] = patch }, "MSE");
        loss.backward();
        foreach (var input in inputs)
        {
            Assert.True(input.requires_grad); using var gradient = input.grad!; Assert.True(gradient.eq(.25).all().item<bool>());
        }
    }

    [Fact]
    public void Training_restores_grad_mode_and_rejects_cancelled_or_mismatched_batches_without_mutation()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        using (var disabled = no_grad())
        {
            var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank); using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
            using var patch = new TrainableLoraPatch(ones([4, 1]) * .01, ones([1, 288]) * .01, 1);
            var patches = new Dictionary<string, TrainableLoraPatch> { ["out.2.weight"] = patch };
            var latent = ones([1, 4, 8, 8]); var noise = ones_like(latent); var sigma = tensor(new[] { .5f }); var context = zeros([1, 3, 16]);
            Assert.Throws<OperationCanceledException>(() => SdLoraTrainingObjective.CalculateLoss(denoiser, latent, noise, sigma, context, patches, "MSE", cancellationToken: new(true)));
            Assert.Throws<ArgumentException>(() => SdLoraTrainingObjective.CalculateLoss(denoiser, latent, ones([2, 4, 8, 8]), sigma, context, patches, "MSE"));
            Assert.Throws<ArgumentException>(() => denoiser.DenoiseForTraining(latent, sigma, zeros([2, 3, 16]), patches));
            Assert.Throws<ArgumentException>(() => SdLoraTrainingObjective.CalculateLoss(denoiser, latent, noise, sigma, context, patches, "unknown"));
            using var loss = SdLoraTrainingObjective.CalculateLoss(denoiser, latent, noise, sigma, context, patches, "MSE");
            Assert.True(loss.requires_grad); Assert.False(is_grad_enabled()); loss.backward();
            using var gradient = patch.Up.grad!; Assert.NotNull(gradient); Assert.True(gradient.isfinite().all().item<bool>());
            Assert.False(latent.requires_grad); Assert.False(sigma.requires_grad); Assert.Null(latent.grad); Assert.Null(sigma.grad);
            using var inferred = denoiser.Denoise(latent, sigma, context); Assert.False(inferred.requires_grad); Assert.False(is_grad_enabled());
        }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
