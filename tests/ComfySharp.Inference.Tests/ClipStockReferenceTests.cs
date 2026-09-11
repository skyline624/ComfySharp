using System.Buffers.Binary;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ComfySharp.Inference;
using TorchSharp;
using Xunit;
using Xunit.Abstractions;

namespace ComfySharp.Inference.Tests;

// Full G owns 2.59 GiB of parameters. Run alone within this testhost, never concurrently with other suites.
[CollectionDefinition("CLIP stock", DisableParallelization = true)]
public sealed class ClipStockCollection;

[Collection("CLIP stock")]
public sealed class ClipStockReferenceTests(ITestOutputHelper output)
{
    private const string CorpusHash = "edc3470a883c96f79e75d4c222b2ba8089ba4d4f78de0d5b941dc03175135f6c";

    [Fact]
    public void StockReferencesArePinnedAndRemainExplicitlySynthetic()
    {
        var raw = ClipReferenceTests.Resource("clip-stock.cpu-f32.json");
        Assert.Equal(CorpusHash, Convert.ToHexStringLower(SHA256.HashData(raw)));
        using var doc = JsonDocument.Parse(raw);
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", doc.RootElement.GetProperty("backendCommit").GetString());
        Assert.False(doc.RootElement.GetProperty("laboratory").GetProperty("modelWeightsUsed").GetBoolean());
        Assert.True(doc.RootElement.GetProperty("laboratory").GetProperty("syntheticWeights").GetBoolean());
        Assert.Equal(2, doc.RootElement.GetProperty("cases").GetArrayLength());
    }

    [Theory]
    [InlineData("stock/l", 197, 123650304L)]
    [InlineData("stock/g", 517, 694659840L)]
    public void FullStockDimensionsMatchFrozenSourceAcrossRepeatedForwards(string id, int tensorCount, long parameterCount)
    {
        using var doc = JsonDocument.Parse(ClipReferenceTests.Resource("clip-stock.cpu-f32.json"));
        var reference = doc.RootElement.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("id").GetString() == id);
        var config = ClipReferenceTests.Config(reference.GetProperty("config"));
        Assert.Equal(id == "stock/l" ? ClipTextConfig.Large : ClipTextConfig.Giant, config);
        var shapes = ClipWeightSchema.Describe(config);
        Assert.Equal(tensorCount, shapes.Count);
        Assert.Equal(parameterCount, shapes.Values.Sum(shape => shape.Aggregate(1L, (n, d) => n * d)));
        Assert.Equal(parameterCount, reference.GetProperty("parameterCount").GetInt64());
        NativeRuntimeBootstrap.Initialize();
        int threads = torch.get_num_threads();
        torch.set_num_threads(1);
        var observations = new List<long> { PrivateBytes() };
        var stopwatch = Stopwatch.StartNew();
        try
        {
            // Only input parameters use this public integer recipe. All expected tensors were computed
            // independently by the frozen Python AST and are immutable committed bytes.
            var acquired = new Dictionary<string, torch.Tensor>(StringComparer.Ordinal);
            try
            {
                foreach (var (name, shape) in shapes)
                {
                    acquired.Add(name, Parameter(name, shape));
                    if (acquired.Count % 16 == 0) GC.Collect(2, GCCollectionMode.Optimized, blocking: true);
                }
                using var weights = ClipWeightSet.FromOwnedTensors(config, acquired);
                acquired.Clear();
                using var encoder = new ClipTextEncoder(weights);
                observations.Add(PrivateBytes());
                output.WriteLine($"{id}: parameters={parameterCount}, learnedTensors={tensorCount}, constructionMs={stopwatch.ElapsedMilliseconds}, synthetic=true.");
                var inputs = ClipReferenceTests.Rows(reference.GetProperty("tokens"));
                var opts = reference.GetProperty("options");
                var options = new ClipForwardOptions
                {
                    IntermediateLayer = opts.GetProperty("intermediate_output").GetInt32(),
                    NormalizeIntermediate = opts.GetProperty("final_layer_norm_intermediate").GetBoolean(),
                    TokenCounts = opts.GetProperty("num_tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray()
                };
                var comparison = new ClipReferenceTests(output);
                var expected = reference.GetProperty("outputs");
                for (int iteration = 0; iteration < 3; iteration++)
                {
                    long before = stopwatch.ElapsedMilliseconds;
                    using (var actual = encoder.Forward(inputs, options))
                    {
                        string key = $"{id}/repeat-{iteration}";
                        comparison.Compare(key, "final", actual.FinalHidden, expected.GetProperty("final"));
                        comparison.Compare(key, "intermediate", actual.IntermediateHidden, expected.GetProperty("intermediate"));
                        comparison.Compare(key, "projected", actual.ProjectedPooled, expected.GetProperty("projected"));
                        comparison.Compare(key, "pooled", actual.Pooled, expected.GetProperty("pooled"));
                    }
                    observations.Add(PrivateBytes());
                    output.WriteLine($"{id}/repeat-{iteration}: forwardAndCompareMs={stopwatch.ElapsedMilliseconds - before}.");
                }
            }
            finally { foreach (var value in acquired.Values) value.Dispose(); }
        }
        finally
        {
            torch.set_num_threads(threads);
            observations.Add(PrivateBytes());
            output.WriteLine($"{id}: processPrivateBytes=[{string.Join(',', observations)}]; allocator/process observations only, not a zero-leak or model-family qualification claim.");
        }
    }

    private static torch.Tensor Parameter(string name, IReadOnlyList<long> shape)
    {
        var digest = SHA256.HashData(Encoding.ASCII.GetBytes(name));
        long seed = BinaryPrimitives.ReadUInt32LittleEndian(digest) % 65521;
        long stride = 1 + BinaryPrimitives.ReadUInt32LittleEndian(digest.AsSpan(4)) % 251;
        int count = checked((int)shape.Aggregate(1L, (n, d) => checked(n * d)));
        var data = new float[count];
        float divisor = shape.Count > 1 ? 4096 : 8192;
        bool norm = name.EndsWith(".weight", StringComparison.Ordinal) && name.Contains("layer_norm", StringComparison.Ordinal);
        for (int i = 0; i < count; i++) data[i] = ((i * stride + seed) % 257 - 128) / divisor + (norm ? 1 : 0);
        using var flat = torch.tensor(data, dtype: torch.ScalarType.Float32);
        return flat.reshape(shape.ToArray()).DetachFromDisposeScope();
    }

    private static long PrivateBytes()
    {
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        return process.PrivateMemorySize64;
    }
}
