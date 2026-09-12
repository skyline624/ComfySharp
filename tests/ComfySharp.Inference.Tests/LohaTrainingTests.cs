using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LohaTrainingTests
{
    public static IEnumerable<object[]> Cases => Enumerable.Range(0, 16).Select(i => new object[] { i });
    private static JsonDocument Fixture()
    {
        var bytes = ClipReferenceTests.Resource("loha-training.reference.json");
        Assert.Equal("cfa3149c52cb10031e92cb648543a82d143aa9a81bb94f8110f83e3b61c288db", Convert.ToHexStringLower(SHA256.HashData(bytes)));
        return JsonDocument.Parse(bytes);
    }
    private static Tensor Read(JsonElement e) => tensor(e.GetProperty("values").EnumerateArray().Select(x => x.GetSingle()).ToArray(),
        e.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray());
    private static TrainableLohaPatch Create(JsonElement initial) => new(
        Read(initial.GetProperty("hada_w1_a")), Read(initial.GetProperty("hada_w1_b")),
        Read(initial.GetProperty("hada_w2_a")), Read(initial.GetProperty("hada_w2_b")),
        initial.GetProperty("alpha").GetProperty("values")[0].GetSingle(),
        initial.TryGetProperty("hada_t1", out var t1) ? Read(t1) : null,
        initial.TryGetProperty("hada_t2", out var t2) ? Read(t2) : null);
    private static void Near(Tensor actual, JsonElement expected, string name)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()), actual.shape);
        var values = expected.GetProperty("values").EnumerateArray().Select(x => x.GetSingle()).ToArray();
        var observed = actual.data<float>().ToArray();
        for (int i = 0; i < values.Length; i++)
            Assert.True(float.IsFinite(observed[i]) && Math.Abs(observed[i] - values[i]) <= 3e-5 + 3e-5 * Math.Abs(values[i]),
                $"{name}[{i}]: {observed[i]} vs {values[i]}");
    }

    [Theory] [MemberData(nameof(Cases))]
    public void Every_first_order_gradient_and_optimizer_update_matches_frozen_LohaDiff(int index)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            using var document = Fixture(); var root = document.RootElement;
            Assert.Equal(3e-5, root.GetProperty("absoluteTolerance").GetDouble());
            Assert.Equal(3e-5, root.GetProperty("relativeTolerance").GetDouble());
            var row = root.GetProperty("cases")[index]; var initial = row.GetProperty("initial");
            using var patch = Create(initial); using var retained = patch.Retain();
            Assert.Equal(initial.EnumerateObject().Select(p => p.Name), retained.NamedParameters.Keys);
            using var optimizer = new LoraTrainingOptimizer(new[] { patch }, row.GetProperty("optimizer").GetString()!, .003);
            patch.Dispose(); Assert.Throws<ObjectDisposedException>(() => patch.Parameters);
            using var weight = Read(row.GetProperty("weight")); byte[] original = weight.bytes.ToArray();
            using var target = Read(row.GetProperty("target"));
            foreach (var step in row.GetProperty("steps").EnumerateArray())
            {
                using var iteration = NewDisposeScope(); using var output = retained.Apply(weight);
                Near(output, step.GetProperty("output"), "output");
                using var loss = TrainingLoss.Calculate("MSE", output, target);
                Assert.InRange(Math.Abs(loss.item<float>() - step.GetProperty("loss").GetSingle()), 0, 3e-5);
                optimizer.Accumulate(loss);
                foreach (var (name, parameter) in retained.NamedParameters)
                {
                    Assert.True(parameter.requires_grad); using var gradient = parameter.grad;
                    var expected = step.GetProperty("gradients").GetProperty(name);
                    if (expected.ValueKind == JsonValueKind.Null) Assert.Null(gradient);
                    else { Assert.NotNull(gradient); Near(gradient!, expected, "gradient/" + name); }
                }
                optimizer.Step();
                foreach (var (name, parameter) in retained.NamedParameters)
                    Near(parameter, step.GetProperty("updated").GetProperty(name), "updated/" + name);
                Assert.Equal(1.75f, retained.NamedParameters["alpha"].item<float>());
            }
            Assert.Equal(original, weight.bytes.ToArray());
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory] [InlineData(0)] [InlineData(8)]
    public void Snapshots_and_files_keep_hada_keys_independent_of_training(int index)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        string path = Path.Combine(Path.GetTempPath(), "comfysharp-loha-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            using var scope = NewDisposeScope(); using var document = Fixture();
            var initial = document.RootElement.GetProperty("cases")[index].GetProperty("initial");
            using var patch = Create(initial); var targets = new Dictionary<string, TrainableLohaPatch> { ["layer.weight"] = patch };
            long bytes = patch.Parameters.Sum(p => p.numel()) * 4;
            Assert.Throws<NotSupportedException>(() => LoraTrainingState.Capture(targets, ScalarType.Float32, maxSnapshotBytes: bytes - 1));
            Assert.Throws<NotSupportedException>(() => LoraTrainingFile.SaveTargetsNew(path, targets, maxFactorBytes: bytes - 1));
            Assert.False(File.Exists(path));
            using var state = LoraTrainingState.Capture(targets, ScalarType.Float32, maxSnapshotBytes: bytes);
            using var bf16 = LoraTrainingState.Capture(targets, ScalarType.BFloat16, maxSnapshotBytes: bytes / 2);
            LoraTrainingFile.SaveTargetsNew(path, targets, maxFactorBytes: bytes);
            Assert.Throws<IOException>(() => LoraTrainingFile.SaveTargetsNew(path, targets));
            using (var noGrad = no_grad()) foreach (var value in patch.Parameters) value.fill_(99);
            patch.Dispose(); using var file = new SafeTensorFile(path);
            foreach (var entry in initial.EnumerateObject())
            {
                string key = "diffusion_model.layer." + entry.Name;
                using var fromFile = file.ReadTensor(key); Near(fromFile, entry.Value, key);
                Assert.Equal(state.Tensors[key].bytes.ToArray(), fromFile.bytes.ToArray());
                using var cast = fromFile.to_type(ScalarType.BFloat16);
                Assert.Equal(cast.bytes.ToArray(), bf16.Tensors[key].bytes.ToArray());
                Assert.False(state.Tensors[key].requires_grad);
            }
        }
        finally { if (File.Exists(path)) File.Delete(path); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Unet_weight_bank_trains_LoHa_and_rejects_source_unsupported_bypass(bool tucker)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope(); var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            using var patch = tucker
                ? new TrainableLohaPatch(ones([2, 4]) * .1, ones([2, 32]) * .1, ones([2, 4]) * .2, ones([2, 32]) * .2,
                    t1: ones([2, 2, 3, 3]) * .1, t2: ones([2, 2, 3, 3]) * .2)
                : new TrainableLohaPatch(ones([4, 2]) * .1, ones([2, 288]) * .1, ones([4, 2]) * .2, ones([2, 288]) * .2);
            var patches = new Dictionary<string, TrainableWeightPatch> { ["out.2.weight"] = patch };
            using var input = NativeMath.CpuNoise([1, 4, 8, 8], 511); using var context = NativeMath.CpuNoise([1, 3, 16], 512);
            using var time = tensor(new[] { 17.25f }); using var baseline = model.Forward(input, time, context);
            using var optimizer = new LoraTrainingOptimizer(patches.Values, "SGD", .01);
            using var prediction = model.ForwardForTraining(input, time, context, patches);
            using var loss = prediction.square().mean(); optimizer.Accumulate(loss);
            foreach (var (name, parameter) in patch.NamedParameters)
            {
                using var gradient = parameter.grad;
                if (name == "alpha") Assert.Null(gradient);
                else { Assert.NotNull(gradient); Assert.True(gradient!.isfinite().all().item<bool>()); Assert.True(gradient.abs().sum().item<float>() > 0); }
            }
            optimizer.Step(); using var changed = model.ForwardForTraining(input, time, context, patches);
            Assert.NotEqual(prediction.bytes.ToArray(), changed.bytes.ToArray());
            Assert.Throws<NotSupportedException>(() => model.ForwardForTraining(input, time, context, patches, bypassMode: true));
            using var unchanged = model.Forward(input, time, context); Assert.Equal(baseline.bytes.ToArray(), unchanged.bytes.ToArray());
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Invalid_factor_geometry_and_cancelled_execution_release_resources()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            Assert.Throws<ArgumentException>(() => new TrainableLohaPatch(ones([4, 2]), ones([3, 5]), ones([4, 2]), ones([2, 5])));
            Assert.Throws<ArgumentException>(() => new TrainableLohaPatch(ones([4, 2]), ones([2, 5]), ones([4, 2]), ones([2, 5]), t1: ones([2, 2])));
            Assert.Throws<ArgumentException>(() => new TrainableLohaPatch(full([4, 2], float.NaN), ones([2, 5]), ones([4, 2]), ones([2, 5])));
            Assert.Throws<NotSupportedException>(() => new TrainableLohaPatch(ones([2, 4]), ones([2, 5]), ones([3, 4]), ones([2, 5]),
                t1: ones([2, 2, 3]), t2: ones([3, 2, 3])));
            using var patch = new TrainableLohaPatch(ones([4, 2]), ones([2, 5]), ones([4, 2]), ones([2, 5]));
            Assert.Throws<ArgumentException>(() => patch.Apply(ones([5, 4])));
            Assert.Throws<OperationCanceledException>(() => patch.Apply(ones([4, 5]), new(true)));
            Assert.Throws<ArgumentException>(() => new LoraTrainingOptimizer(new[] { patch, patch }, "SGD", .01));
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Complete_LoHa_factory_matches_source_values_and_rng(bool linear)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            byte[] compressed = ClipReferenceTests.Resource("loha-factory.reference.json.gz");
            Assert.Equal("8d42daf7c40d691677580f148b1afce5772befe1c2bf8fd449dce2a73717768a", Convert.ToHexStringLower(SHA256.HashData(compressed)));
            using var input = new MemoryStream(compressed); using var gzip = new GZipStream(input, CompressionMode.Decompress);
            using var decoded = new MemoryStream(); gzip.CopyTo(decoded);
            Assert.Equal("d81c773f5d714c6e5e5e32bf683ee6d53c71ca8afaae080cb422d5262b2fcd9d", Convert.ToHexStringLower(SHA256.HashData(decoded.ToArray())));
            using var document = JsonDocument.Parse(decoded.ToArray()); var row = document.RootElement;
            var config = new SdUnetConfig(32, 16, linear ? SdAttentionHeadMode.FixedSize : SdAttentionHeadMode.FixedCount, linear ? 8 : 4, linear);
            long bytes = row.GetProperty("parameterBytes").GetInt64();
            Assert.Throws<NotSupportedException>(() => new SdTrainableAdapterSet(config, 2, 317, CPU, bytes - 1, algorithm: "LoHa"));
            using var global = manual_seed(117); using var beforeRandom = global.get_state();
            using var adapters = new SdTrainableAdapterSet(config, 2, 317, CPU, bytes, algorithm: "LoHa");
            using var afterRandom = global.get_state(); Assert.Equal(beforeRandom.bytes.ToArray(), afterRandom.bytes.ToArray());
            Assert.Equal(row.GetProperty("randomStateSha256").GetString(), adapters.InitialCpuRandomStateSha256);
            Assert.Equal(686, adapters.Patches.Count); Assert.Equal(1814, adapters.Patches.Values.Sum(p => p.Parameters.Count));
            Assert.Equal(bytes, adapters.ParameterBytes); Assert.Empty(adapters.ResumedTargets);
            Assert.Equal(row.GetProperty("targets").EnumerateObject().Select(p => p.Name), adapters.Patches.Keys);
            foreach (var (name, patch) in adapters.Patches)
            {
                var expected = row.GetProperty("targets").GetProperty(name);
                if (patch is TrainableLohaPatch loha)
                {
                    Assert.Equal(expected.EnumerateObject().Select(p => p.Name), loha.NamedParameters.Keys);
                    foreach (var (key, value) in loha.NamedParameters) Near(value, expected.GetProperty(key), name + "/" + key);
                }
                else Near(Assert.IsType<TrainableDifferencePatch>(patch).Difference, expected.GetProperty("bias"), name);
            }
        }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
