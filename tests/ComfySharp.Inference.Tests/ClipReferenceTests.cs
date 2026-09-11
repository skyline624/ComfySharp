using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using ComfySharp.Tokenization;
using TorchSharp;
using Xunit;
using Xunit.Abstractions;

namespace ComfySharp.Inference.Tests;

/// <summary>Independent frozen-source outputs, including actual nontrivial learned-parameter operations.</summary>
public sealed class ClipReferenceTests(ITestOutputHelper output)
{
    private const string CorpusHash = "80a36961770ef3b86065f95f73899d75bb6a4d3c95dbb266e4941f4d286e1250";

    internal static byte[] Resource(string suffix)
    {
        var assembly = Assembly.GetExecutingAssembly();
        var name = assembly.GetManifestResourceNames().Single(n => n.EndsWith(suffix, StringComparison.Ordinal));
        using var stream = assembly.GetManifestResourceStream(name)!;
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        return buffer.ToArray();
    }

    public static IEnumerable<object[]> CaseIds()
    {
        using var doc = JsonDocument.Parse(Resource("clip-encoders.cpu-f32.json"));
        return doc.RootElement.GetProperty("cases").EnumerateArray()
            .Select(c => new object[] { c.GetProperty("id").GetString()! }).ToArray();
    }

    [Fact]
    public void FixtureAndSyntheticInputsHavePinnedIndependentProvenance()
    {
        var raw = Resource("clip-encoders.cpu-f32.json");
        Assert.Equal(CorpusHash, Convert.ToHexStringLower(SHA256.HashData(raw)));
        using var doc = JsonDocument.Parse(raw);
        var root = doc.RootElement;
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", root.GetProperty("backendCommit").GetString());
        Assert.Equal(46, root.GetProperty("cases").GetArrayLength());
        Assert.Equal("2.13.0+cu130", root.GetProperty("laboratory").GetProperty("torch").GetString());
        Assert.False(root.GetProperty("laboratory").GetProperty("modelWeightsUsed").GetBoolean());
        Assert.True(root.GetProperty("laboratory").GetProperty("syntheticWeights").GetBoolean());
        Assert.Equal(3e-5, root.GetProperty("comparison").GetProperty("absoluteTolerance").GetDouble());
        Assert.Equal(3e-5, root.GetProperty("comparison").GetProperty("relativeTolerance").GetDouble());
        foreach (var entry in root.GetProperty("assets").EnumerateObject())
        {
            var asset = entry.Value;
            var bytes = Resource(asset.GetProperty("file").GetString()!);
            Assert.Equal(asset.GetProperty("bytes").GetInt32(), bytes.Length);
            Assert.Equal(asset.GetProperty("sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(bytes)));
        }
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void NativeEncoderMatchesFrozenComfyUi(string id)
    {
        using var doc = JsonDocument.Parse(Resource("clip-encoders.cpu-f32.json"));
        var root = doc.RootElement;
        var reference = root.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("id").GetString() == id);
        var expected = reference.GetProperty("outputs");
        // NativeRuntimeBootstrap must be first even when this suite is the only test in a fresh process.
        NativeRuntimeBootstrap.Initialize();
        using var scope = torch.NewDisposeScope();
        if (reference.GetProperty("kind").GetString() == "sdxl")
        {
            using var l = Load(root, "l");
            using var g = Load(root, "g");
            using var combined = new ComfySdxlEncoder(l, g);
            using var actual = combined.Encode(Pairs(reference.GetProperty("l")), Pairs(reference.GetProperty("g")),
                new ClipSdxlConditioningOptions { L = ConditioningOptions(reference.GetProperty("lOptions")), G = ConditioningOptions(reference.GetProperty("gOptions")) });
            Compare(id, "hidden", actual.Hidden, expected.GetProperty("hidden"));
            Compare(id, "pooled", actual.Pooled, expected.GetProperty("pooled"));
            Assert.Null(actual.AttentionMask);
            return;
        }

        using var encoder = Load(root, reference.GetProperty("model").GetString()!);
        if (reference.GetProperty("kind").GetString() == "graph")
        {
            var p = reference.GetProperty("options");
            bool all = p.TryGetProperty("intermediate_output", out var layer) && layer.ValueKind == JsonValueKind.String;
            using var actual = encoder.Forward(Rows(reference.GetProperty("tokens")), new ClipForwardOptions
            {
                AllIntermediateLayers = all,
                IntermediateLayer = layer.ValueKind == JsonValueKind.Number ? layer.GetInt32() : null,
                NormalizeIntermediate = Bool(p, "final_layer_norm_intermediate") ?? true,
                AttentionMask = p.TryGetProperty("attention_mask", out var mask) ? Rows(mask) : null,
                TokenCounts = p.TryGetProperty("num_tokens", out var counts) ? counts.EnumerateArray().Select(c => c.GetInt32()).ToArray() : null
            });
            Compare(id, "final", actual.FinalHidden, expected.GetProperty("final"));
            Compare(id, "intermediate", actual.IntermediateHidden, expected.GetProperty("intermediate"));
            Compare(id, "projected", actual.ProjectedPooled, expected.GetProperty("projected"));
            Compare(id, "pooled", actual.Pooled, expected.GetProperty("pooled"));
        }
        else
        {
            var profile = reference.GetProperty("profile").GetString() switch
            {
                "sd1-l" => ClipProfile.Sd1L, "sdxl-l" => ClipProfile.SdXlL, "sdxl-g" => ClipProfile.SdXlG,
                _ => throw new InvalidDataException("Unknown fixture profile.")
            };
            using var wrapper = new ComfyClipEncoder(encoder, profile);
            using var actual = wrapper.Encode(Pairs(reference.GetProperty("tokenWeights")), ConditioningOptions(reference.GetProperty("options")));
            Compare(id, "hidden", actual.Hidden, expected.GetProperty("hidden"));
            Compare(id, "pooled", actual.Pooled, expected.GetProperty("pooled"));
            if (expected.TryGetProperty("mask", out var mask)) Compare(id, "mask", actual.AttentionMask, mask);
            else Assert.Null(actual.AttentionMask);
        }
    }

    private static ClipTextEncoder Load(JsonElement root, string model)
    {
        var asset = root.GetProperty("assets").GetProperty(model);
        var bytes = Resource(asset.GetProperty("file").GetString()!);
        var path = Path.Combine(Path.GetTempPath(), "comfysharp-clip-reference-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            File.WriteAllBytes(path, bytes);
            using var file = new SafeTensorFile(path);
            var config = Config(root.GetProperty("configs").GetProperty(model));
            var plan = ClipCheckpointLoader.Inspect(file, config, ClipCheckpointLayout.Canonical);
            using var weights = ClipCheckpointLoader.Load(file, plan);
            return new ClipTextEncoder(weights);
        }
        finally { File.Delete(path); }
    }

    internal static ClipTextConfig Config(JsonElement config) => new(
        config.GetProperty("hidden_size").GetInt32(), config.GetProperty("intermediate_size").GetInt32(),
        config.GetProperty("num_hidden_layers").GetInt32(), config.GetProperty("num_attention_heads").GetInt32(),
        config.GetProperty("hidden_act").GetString() switch
        {
            "quick_gelu" => ClipActivation.QuickGelu, "gelu" => ClipActivation.Gelu,
            "gelu_pytorch_tanh" => ClipActivation.GeluTanh, _ => throw new InvalidDataException("Unknown activation.")
        });

    private static bool? Bool(JsonElement p, string name) => p.TryGetProperty(name, out var v) ? v.GetBoolean() : null;
    internal static int[][] Rows(JsonElement element) => element.EnumerateArray()
        .Select(row => row.EnumerateArray().Select(t => t.GetInt32()).ToArray()).ToArray();
    private static ClipTokenWeight[][] Pairs(JsonElement element) => element.EnumerateArray()
        .Select(row => row.EnumerateArray().Select(t => new ClipTokenWeight(t[0].GetInt32(),
            BitConverter.UInt64BitsToDouble(Convert.ToUInt64(t[1].GetString(), 16)))).ToArray()).ToArray();

    private static ClipConditioningOptions ConditioningOptions(JsonElement p) => new()
    {
        HiddenSelection = p.TryGetProperty("selection", out var s) ? s.GetString() switch
        {
            "last" => ClipHiddenSelection.Last, "hidden" => ClipHiddenSelection.Hidden, "all" => ClipHiddenSelection.All,
            _ => throw new InvalidDataException("Unknown hidden selection.")
        } : ClipHiddenSelection.ProfileDefault,
        IntermediateLayer = p.TryGetProperty("layer", out var layer) ? layer.GetInt32() : null,
        NormalizeIntermediate = Bool(p, "normalize"), ProjectPooled = Bool(p, "project"),
        EnableAttentionMasks = Bool(p, "mask") ?? false, ZeroOutMasked = Bool(p, "zero") ?? false,
        ReturnAttentionMasks = Bool(p, "returnMask") ?? false,
        SpecialTokens = p.TryGetProperty("specialTokens", out var spec) ? new ClipSpecialTokens(
            spec.GetProperty("start").ValueKind == JsonValueKind.Null ? null : spec.GetProperty("start").GetInt32(),
            spec.GetProperty("end").ValueKind == JsonValueKind.Null ? null : spec.GetProperty("end").GetInt32(), spec.GetProperty("pad").GetInt32()) : null
    };

    internal void Compare(string id, string name, torch.Tensor? actual, JsonElement expected)
    {
        if (expected.ValueKind == JsonValueKind.Null) { Assert.Null(actual); return; }
        Assert.NotNull(actual);
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(n => n.GetInt64()).ToArray(), actual.shape);
        Assert.Equal("cpu", actual.device.ToString());
        Assert.False(actual.requires_grad);
        var raw = Convert.FromBase64String(expected.GetProperty("littleEndianBase64").GetString()!);
        Assert.Equal(expected.GetProperty("bytesSha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(raw)));
        Assert.True(BitConverter.IsLittleEndian);
        using var contiguous = actual.contiguous();
        if (expected.GetProperty("dtype").GetString() == "int64")
        {
            Assert.Equal(torch.ScalarType.Int64, actual.dtype);
            var values = contiguous.data<long>().ToArray();
            Assert.Equal(raw.Length / sizeof(long), values.Length);
            for (int i = 0; i < values.Length; i++) Assert.Equal(BitConverter.ToInt64(raw, i * sizeof(long)), values[i]);
            output.WriteLine($"{id}/{name}: exact Int64 mask.");
            return;
        }
        Assert.Equal(torch.ScalarType.Float32, actual.dtype);
        var floats = contiguous.data<float>().ToArray();
        Assert.Equal(raw.Length / sizeof(float), floats.Length);
        double maxAbsolute = 0, maxToleranceFraction = 0;
        int nonfinite = 0;
        for (int i = 0; i < floats.Length; i++)
        {
            float e = BitConverter.ToSingle(raw, i * sizeof(float)), a = floats[i];
            if (!float.IsFinite(e))
            {
                nonfinite++;
                Assert.True(float.IsNaN(e) ? float.IsNaN(a) : e.Equals(a), $"{id}/{name}[{i}] nonfinite classification differs.");
                continue;
            }
            Assert.True(float.IsFinite(a), $"{id}/{name}[{i}] became nonfinite.");
            double error = Math.Abs((double)a - e), tolerance = 3e-5 + 3e-5 * Math.Abs(e);
            Assert.True(error <= tolerance, $"{id}/{name}[{i}]: actual={a:R}, expected={e:R}, error={error:R}, tolerance={tolerance:R}.");
            maxAbsolute = Math.Max(maxAbsolute, error);
            maxToleranceFraction = Math.Max(maxToleranceFraction, error / tolerance);
        }
        output.WriteLine(FormattableString.Invariant($"{id}/{name}: maxAbsolute={maxAbsolute:R}, maxToleranceFraction={maxToleranceFraction:R}, nonfinite={nonfinite}; fixed clip-basic-cpu-f32-v1."));
    }
}
