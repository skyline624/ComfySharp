using System.Text.Json;
using Xunit;

namespace ComfySharp.Inference.Tests;

public sealed class TrainingReferenceCorpusTests
{
    [Theory]
    [InlineData("linux-x64")]
    [InlineData("osx-arm64")]
    public void Platform_oracles_preserve_source_recipes_inputs_and_thresholds(string target)
    {
        foreach (string component in new[] { "batches", "denoising", "adapters" })
        {
            using var windows = TrainingReferenceCorpus.Load(component, "win-x64");
            using var platform = TrainingReferenceCorpus.Load(component, target);
            var expected = windows.RootElement; var actual = platform.RootElement;
            Assert.Equal(expected.GetProperty("sourceHashes").GetRawText(), actual.GetProperty("sourceHashes").GetRawText());
            var left = expected.GetProperty("cases"); var right = actual.GetProperty("cases"); Assert.Equal(left.GetArrayLength(), right.GetArrayLength());
            if (component != "batches")
            {
                Assert.Equal(3e-5, actual.GetProperty("absoluteTolerance").GetDouble());
                Assert.Equal(3e-5, actual.GetProperty("relativeTolerance").GetDouble());
            }
            string[] fields = component switch
            {
                "batches" => ["name", "bucketMode", "mode", "batchSize", "seed", "count", "inputs", "latentScale", "initialStateSha256"],
                "denoising" => ["linearProjection", "predictionKind", "initial", "latent", "noise", "context", "sigma", "gradAccumulation"],
                _ => ["linearProjection", "seed", "rank", "randomStateSha256", "parameterCount", "latent", "context", "times", "target"]
            };
            for (int i = 0; i < left.GetArrayLength(); i++) foreach (string field in fields)
                Assert.Equal(left[i].GetProperty(field).GetRawText(), right[i].GetProperty(field).GetRawText());
        }
    }
}
