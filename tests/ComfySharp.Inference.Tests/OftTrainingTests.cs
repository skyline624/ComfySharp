using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class OftTrainingTests
{
    private static JsonDocument Fixture()
    {
        var bytes = ClipReferenceTests.Resource("oft-training.reference.json.gz");
        Assert.Equal("8f2b11934ca8da8a7c5fa8371af28e19f2baf4165c8dd8e1b43a5d9f5b205eed", Convert.ToHexStringLower(SHA256.HashData(bytes)));
        using var input = new MemoryStream(bytes); using var gzip = new GZipStream(input, CompressionMode.Decompress);
        using var decoded = new MemoryStream(); gzip.CopyTo(decoded); var raw = decoded.ToArray();
        Assert.Equal("655c6c2a200c407252dde2362db5eca93131e61bb4a9b047f8b1d7d65634e74a", Convert.ToHexStringLower(SHA256.HashData(raw)));
        return JsonDocument.Parse(raw);
    }
    private static Tensor Read(JsonElement e) => tensor(e.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray(), e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray());
    private static void Near(Tensor actual, JsonElement expected, string name)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()), actual.shape);
        using var cast = actual.to_type(ScalarType.Float32); using var contiguous = cast.contiguous();
        var values = expected.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray(); var observed = contiguous.data<float>().ToArray();
        for (int i = 0; i < values.Length; i++)
            Assert.True(float.IsFinite(observed[i]) && Math.Abs(observed[i] - values[i]) <= 3e-5 + 3e-5 * Math.Abs(values[i]), $"{name}[{i}]: {observed[i]} vs {values[i]}");
    }
    public static IEnumerable<object[]> Cases => Enumerable.Range(0, 192).Select(i => new object[] { i });
    [Theory] [MemberData(nameof(Cases))]
    public void Frozen_rotations_gradients_updates_and_captured_constraint(int index)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            using var doc = Fixture(); var root = doc.RootElement; var row = root.GetProperty("cases")[index];
            Assert.Equal(3e-5, root.GetProperty("absoluteTolerance").GetDouble()); Assert.Equal(3e-5, root.GetProperty("relativeTolerance").GetDouble());
            var initial = row.GetProperty("initial"); using var blocks = Read(initial.GetProperty("oft_blocks"));
            using var scale = initial.TryGetProperty("rescale", out var s) ? Read(s) : null;
            using var original = new TrainableOftPatch(blocks, row.GetProperty("alpha").GetDouble(), scale);
            using var patch = original.Retain(); original.Dispose(); Assert.Throws<ObjectDisposedException>(() => original.Parameters);
            Assert.Equal(initial.EnumerateObject().Select(p => p.Name), patch.NamedParameters.Keys);
            // The constructor must own storage independently from its caller.
            using (no_grad()) { blocks.fill_(17); scale?.fill_(19); }
            foreach (var (name, value) in patch.NamedParameters) Near(value, initial.GetProperty(name), name);
            var dtype = row.GetProperty("dtype").GetString() switch { "torch.float16" => ScalarType.Float16, "torch.bfloat16" => ScalarType.BFloat16, _ => ScalarType.Float32 };
            var input = Read(row.GetProperty("input")).to_type(dtype);
            if (input.dim() > 2) input = input.transpose(-1, -2).contiguous().transpose(-1, -2);
            input = input.detach().requires_grad_(); var inputBytes = input.contiguous().bytes.ToArray();
            using var target = Read(row.GetProperty("target")); bool weight = row.GetProperty("mode").GetString() == "weight";
            Tensor Forward() => weight ? patch.Apply(input) : patch.ApplyOutput(input + zeros_like(input), row.GetProperty("dims").GetInt32() > 0, row.GetProperty("multiplier").GetDouble());
            using var optimizer = new LoraTrainingOptimizer(new[] { patch }, "SGD", .003);
            using var output = Forward(); Near(output, row.GetProperty("output"), "output");
            Assert.Equal(dtype, output.dtype);
            using var loss = (output.to_type(ScalarType.Float32) - target).square().mean();
            Assert.True(Math.Abs(loss.item<float>() - row.GetProperty("loss").GetSingle()) <= 3e-5);
            optimizer.Accumulate(loss);
            using var inputGradient = input.grad; Assert.NotNull(inputGradient); Near(inputGradient!, row.GetProperty("inputGradient"), "input gradient");
            foreach (var (name, value) in patch.NamedParameters)
            {
                using var gradient = value.grad; var expected = row.GetProperty("gradients").GetProperty(name);
                if (name == "alpha") { Assert.Equal(JsonValueKind.Null, expected.ValueKind); Assert.Null(gradient); }
                else { Assert.NotNull(gradient); Near(gradient!, expected, name + " gradient"); }
            }
            optimizer.Step(); foreach (var (name, value) in patch.NamedParameters) Near(value, row.GetProperty("updated").GetProperty(name), name + " updated");
            using (no_grad()) patch.NamedParameters["alpha"].fill_(123);
            using var after = Forward(); Near(after, row.GetProperty("afterAlphaMutation"), "captured constraint");
            Assert.Equal(inputBytes, input.contiguous().bytes.ToArray());
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Invalid_inputs_cancellation_and_ownership_leave_no_native_resources()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            Assert.Throws<ArgumentException>(() => new TrainableOftPatch(ones([2, 3, 4])));
            Assert.Throws<ArgumentException>(() => new TrainableOftPatch(ones([0, 3, 3])));
            Assert.Throws<ArgumentException>(() => new TrainableOftPatch(full([2, 3, 3], float.NaN)));
            Assert.Throws<ArgumentException>(() => new TrainableOftPatch(ones([2, 3, 3], dtype: ScalarType.Float16)));
            Assert.Throws<ArgumentOutOfRangeException>(() => new TrainableOftPatch(ones([2, 3, 3]), double.PositiveInfinity));
            using var patch = new TrainableOftPatch(zeros([2, 3, 3]));
            Assert.Throws<ArgumentException>(() => patch.Apply(ones([5, 4])));
            Assert.Throws<ArgumentException>(() => patch.ApplyOutput(ones([2, 5]), false));
            Assert.Throws<ArgumentException>(() => patch.ApplyOutput(ones([2, 6]), true));
            Assert.Throws<NotSupportedException>(() => patch.ApplyOutput(ones([2, 6], dtype: ScalarType.Float16), false));
            Assert.Throws<OperationCanceledException>(() => patch.Apply(ones([6, 4]), new(true)));
            Assert.Throws<OperationCanceledException>(() => patch.ApplyOutput(ones([2, 6]), false, cancellationToken: new(true)));
            Assert.Throws<ArgumentException>(() => new LoraTrainingOptimizer(new[] { patch, patch }, "SGD", .01));
            using var badScale = new TrainableOftPatch(zeros([2, 3, 3]), rescale: ones([7]));
            Assert.ThrowsAny<Exception>(() => badScale.Apply(ones([6, 4])));
            Assert.ThrowsAny<Exception>(() => badScale.ApplyOutput(ones([2, 6]), false));
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory] [InlineData(ScalarType.Float32)] [InlineData(ScalarType.BFloat16)]
    public void State_and_file_export_preserve_names_dtype_and_independent_storage(ScalarType dtype)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        string path = Path.Combine(Path.GetTempPath(), "comfysharp-oft-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            using var scope = NewDisposeScope();
            using var patch = new TrainableOftPatch(arange(18, dtype: ScalarType.Float32).reshape(2, 3, 3) / 31, -.1, ones([6, 1]));
            var targets = new Dictionary<string, TrainableWeightPatch> { { "layer.weight", patch } };
            using var state = LoraTrainingState.Capture(targets, dtype, maxSnapshotBytes: 25 * (dtype == ScalarType.Float32 ? 4 : 2));
            Assert.Throws<OperationCanceledException>(() => LoraTrainingState.Capture(targets, dtype, cancellationToken: new(true)));
            Assert.Throws<NotSupportedException>(() => LoraTrainingFile.SaveTargetsNew(path, targets, maxFactorBytes: 99));
            Assert.False(File.Exists(path));
            LoraTrainingFile.SaveTargetsNew(path, targets, maxFactorBytes: 100);
            using var file = new SafeTensorFile(path);
            Assert.Equal(state.Tensors.Keys.Order(), file.Tensors.Keys.Order());
            foreach (var (name, value) in patch.NamedParameters)
            {
                string key = "diffusion_model.layer." + name;
                using var disk = file.ReadTensor(key); Assert.Equal(ScalarType.Float32, disk.dtype);
                Assert.Equal(value.bytes.ToArray(), disk.bytes.ToArray());
                var saved = state.Tensors[key]; Assert.Equal(dtype, saved.dtype); Assert.False(saved.requires_grad);
                Assert.Equal(value.to_type(dtype).bytes.ToArray(), saved.bytes.ToArray());
                using (no_grad()) value.fill_(42);
                Assert.Equal(disk.to_type(dtype).bytes.ToArray(), saved.bytes.ToArray());
            }
            Assert.Throws<IOException>(() => LoraTrainingFile.SaveTargetsNew(path, targets));
            patch.Dispose(); Assert.Equal(3, state.Tensors.Count);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Unet_trains_owned_rotation_in_weight_and_bypass_modes(bool bypass)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope(); var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            using var patch = new TrainableOftPatch(zeros([1, 4, 4]), .1);
            var patches = new Dictionary<string, TrainableWeightPatch> { { "out.2.weight", patch } };
            using var input = NativeMath.CpuNoise([1, 4, 8, 8], 511); using var context = NativeMath.CpuNoise([1, 3, 16], 512); using var time = tensor(new[] { 17.25f });
            using var baseline = model.Forward(input, time, context); using var optimizer = new LoraTrainingOptimizer(patches.Values, "SGD", .01);
            using var prediction = model.ForwardForTraining(input, time, context, patches, maxPatchedWeightBytes: bypass ? 0 : 8192, bypassMode: bypass);
            Assert.True(allclose(baseline, prediction, rtol: 3e-5, atol: 3e-5));
            using var target = NativeMath.CpuNoise(prediction.shape, 513); using var loss = (prediction - target).square().mean(); optimizer.Accumulate(loss);
            using var gradient = patch.NamedParameters["oft_blocks"].grad; Assert.NotNull(gradient);
            Assert.True(gradient!.isfinite().all().item<bool>()); Assert.True(gradient.abs().sum().item<float>() > 0);
            using var alphaGradient = patch.NamedParameters["alpha"].grad; Assert.Null(alphaGradient);
            optimizer.Step(); using var changed = model.ForwardForTraining(input, time, context, patches, maxPatchedWeightBytes: bypass ? 0 : 8192, bypassMode: bypass);
            Assert.NotEqual(prediction.contiguous().bytes.ToArray(), changed.contiguous().bytes.ToArray());
            using var state = LoraTrainingState.Capture(patches, ScalarType.Float32, maxSnapshotBytes: 68);
            Assert.Equal(2, state.Tensors.Count); Assert.Equal(patch.NamedParameters["oft_blocks"].bytes.ToArray(), state.Tensors["diffusion_model.out.2.oft_blocks"].bytes.ToArray());
            Assert.Throws<NotSupportedException>(() => LoraTrainingState.Capture(patches, ScalarType.Float32, maxSnapshotBytes: 67));
            byte[] snapshot = state.Tensors["diffusion_model.out.2.oft_blocks"].bytes.ToArray();
            using (no_grad()) patch.NamedParameters["oft_blocks"].fill_(7);
            Assert.Equal(snapshot, state.Tensors["diffusion_model.out.2.oft_blocks"].bytes.ToArray());
            using var unchanged = model.Forward(input, time, context); Assert.Equal(baseline.bytes.ToArray(), unchanged.bytes.ToArray());
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
