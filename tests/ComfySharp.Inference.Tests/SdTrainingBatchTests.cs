using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdTrainingBatchTests
{
    private static JsonDocument Corpus() => TrainingReferenceCorpus.Load("batches");
    private static Tensor Read(JsonElement e) => tensor(e.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray(), e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray()).clone();
    private static void Exact(Tensor actual, JsonElement expected)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()), actual.shape);
        Assert.Equal(expected.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray(), actual.data<float>().ToArray());
    }
    private static string Hash(Tensor state) => Convert.ToHexStringLower(SHA256.HashData(state.data<byte>().ToArray()));

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)]
    public void Dataset_batches_noise_sigmas_and_rng_states_match_frozen_source(int caseIndex)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        using (var corpus = Corpus())
        using (var global = manual_seed(31415))
        {
            using var originalGlobalState = global.get_state(); string globalHash = Hash(originalGlobalState);
            var row = corpus.RootElement.GetProperty("cases")[caseIndex];
            var inputs = row.GetProperty("inputs").EnumerateArray().Select(Read).ToArray();
            using var dataset = new SdTrainingDataset(inputs, row.GetProperty("bucketMode").GetBoolean());
            Assert.Equal(row.GetProperty("mode").GetString(), dataset.Mode.ToString()); Assert.Equal(row.GetProperty("count").GetInt64(), dataset.Count);
            // Inputs may be mutated or disposed after snapshot creation.
            foreach (var input in inputs) { input.fill_(99); input.Dispose(); }
            using var sampler = new SdTrainingBatchSampler(dataset, row.GetProperty("seed").GetUInt64(), row.GetProperty("batchSize").GetInt32(), CPU);
            using var initialState = sampler.CaptureRandomState(); Assert.Equal(row.GetProperty("initialStateSha256").GetString(), Hash(initialState));
            dataset.Dispose();
            foreach (var reference in row.GetProperty("batches").EnumerateArray())
            {
                using var batch = sampler.Next();
                Assert.Equal(reference.GetProperty("step").GetInt64(), batch.Index); Assert.Equal(reference.GetProperty("noiseSeed").GetUInt64(), batch.NoiseSeed);
                Assert.Equal(reference.GetProperty("lossWeightPerGroup").GetDouble(), batch.LossWeightPerGroup);
                var groups = reference.GetProperty("groups"); Assert.Equal(groups.GetArrayLength(), batch.Groups.Count);
                for (int i = 0; i < batch.Groups.Count; i++)
                {
                    Assert.Equal(groups[i].GetProperty("indices").EnumerateArray().Select(v => v.GetInt64()), batch.Groups[i].Indices);
                    Exact(batch.Groups[i].Latent, groups[i].GetProperty("latent")); Exact(batch.Groups[i].Noise, groups[i].GetProperty("noise")); Exact(batch.Groups[i].Sigmas, groups[i].GetProperty("sigma"));
                }
                using var state = sampler.CaptureRandomState(); Assert.Equal(reference.GetProperty("stateSha256").GetString(), Hash(state));
            }
            using var finalGlobalState = global.get_state(); Assert.Equal(globalHash, Hash(finalGlobalState));
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Admission_rejects_empty_invalid_and_oversized_datasets_before_sampling()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        Assert.Throws<ArgumentException>(() => new SdTrainingDataset([]));
        Assert.Throws<ArgumentException>(() => new SdTrainingDataset([ones([1, 3, 8, 8])]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SdTrainingDataset([ones([1, 4, 8, 8])], maxBytes: 4));
        using var dataset = new SdTrainingDataset([ones([1, 4, 8, 8])]);
        Assert.Throws<ArgumentOutOfRangeException>(() => new SdTrainingBatchSampler(dataset, 1, 0, CPU));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SdTrainingBatchSampler(dataset, 1, 1, CPU, maxInitialNoiseBytes: 4));
        dataset.Dispose(); Assert.Throws<ObjectDisposedException>(() => new SdTrainingBatchSampler(dataset, 1, 1, CPU));
    }

    [Fact]
    public void Cancellation_before_sampling_preserves_state_and_overflow_never_wraps_a_seed()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            using var dataset = new SdTrainingDataset([ones([2, 4, 2, 3])]);
            using var sampler = new SdTrainingBatchSampler(dataset, ulong.MaxValue, 1, CPU);
            using var state = sampler.CaptureRandomState(); string hash = Hash(state);
            Assert.Throws<OperationCanceledException>(() => sampler.Next(new(true)));
            using var unchanged = sampler.CaptureRandomState(); Assert.Equal(hash, Hash(unchanged));
            using var first = sampler.Next(); Assert.Equal(ulong.MaxValue, first.NoiseSeed);
            Assert.Throws<OverflowException>(() => sampler.Next());
            Assert.Throws<InvalidOperationException>(() => sampler.Next());
            sampler.Dispose(); Assert.Throws<ObjectDisposedException>(() => sampler.Next());
        }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
