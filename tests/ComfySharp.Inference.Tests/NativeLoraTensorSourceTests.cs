using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class NativeLoraTensorSourceTests
{
    [Theory]
    [InlineData(".lora_up.weight", ".lora_down.weight")]
    [InlineData("_lora.up.weight", "_lora.down.weight")]
    [InlineData(".lora_B.weight", ".lora_A.weight")]
    [InlineData(".lora.up.weight", ".lora.down.weight")]
    [InlineData(".lora_B", ".lora_A")]
    [InlineData(".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight")]
    [InlineData(".lora_B.default.weight", ".lora_A.default.weight")]
    public void All_factor_spellings_use_the_same_plan_and_result_as_files(string up, string down)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        string path = Path.Combine(Path.GetTempPath(), "comfysharp-native-lora-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            using var scope = NewDisposeScope();
            var values = new Dictionary<string, Tensor>
            {
                ["layer" + up] = tensor(new[] { 1f, 2f }).reshape(2, 1),
                ["layer" + down] = tensor(new[] { 3f, 4f, 5f }).reshape(1, 3),
                ["layer.alpha"] = tensor(0.5), ["layer.diff_b"] = tensor(new[] { .1f, -.2f })
            };
            using (var output = File.Create(path)) SafeTensorWriter.Write(output, values);
            using var file = new SafeTensorFile(path); using var native = new NativeLoraTensorSource(values);
            LoraAlias[] aliases = [new("layer", new("model", "layer.weight", new long[] { 2, 3 }))];
            var left = LoraFileLoader.Inspect(file, aliases); var right = LoraFileLoader.Inspect(native, aliases);
            Assert.Equal(left.Bindings.Select(b => (b.Prefix, b.Target.Weight, b.Up, b.Down, b.Difference)),
                right.Bindings.Select(b => (b.Prefix, b.Target.Weight, b.Up, b.Down, b.Difference)));
            Assert.Equal(left.ResidentFactorBytes, right.ResidentFactorBytes);
            using var fromFile = LoraFileLoader.Load(file, left); using var fromNative = LoraFileLoader.Load(native, right);
            file.Dispose(); native.Dispose();
            foreach (var binding in right.Bindings)
            {
                using var weight = ones(binding.Target.Shape.ToArray());
                using var expected = fromFile.Apply("model", binding.Target.Weight, weight);
                using var actual = fromNative.Apply("model", binding.Target.Weight, weight);
                Assert.Equal(expected.bytes.ToArray(), actual.bytes.ToArray());
            }
        }
        finally { File.Delete(path); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory]
    [InlineData(ScalarType.Float32, "F32")]
    [InlineData(ScalarType.Float16, "F16")]
    [InlineData(ScalarType.BFloat16, "BF16")]
    public void Snapshot_and_reads_have_independent_storage_and_dtype(ScalarType dtype, string name)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            var input = ones(new long[] { 3, 2 }, dtype: dtype).transpose(0, 1);
            using var source = new NativeLoraTensorSource(new Dictionary<string, Tensor> { ["value"] = input });
            Assert.Equal(name, source.Tensors["value"].DType);
            Assert.Equal(new long[] { 2, 3 }, source.Tensors["value"].Shape);
            input.fill_(0); input.Dispose();
            using var first = source.ReadTensor("value"); first.fill_(7);
            using var second = source.ReadTensor("value"); source.Dispose();
            Assert.Equal(dtype, second.dtype); Assert.True(second.is_contiguous());
            using var expected = ones(new long[] { 2, 3 }, dtype: dtype);
            Assert.Equal(expected.bytes.ToArray(), second.bytes.ToArray());
            Assert.Throws<ObjectDisposedException>(() => source.ReadTensor("value"));
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Plans_cannot_cross_sources_and_failed_reads_or_loads_release_their_copies()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            var values = new Dictionary<string, Tensor> { ["layer.diff"] = ones(2) };
            using var first = new NativeLoraTensorSource(values); using var second = new NativeLoraTensorSource(values);
            var plan = LoraFileLoader.Inspect(first, [new("layer", new("model", "layer.weight", new long[] { 2 }))]);
            long borrowed = Tensor.TotalCount;
            Assert.Throws<ArgumentException>(() => LoraFileLoader.Load(second, plan));
            Assert.Throws<OperationCanceledException>(() => first.ReadTensor("layer.diff", new(true)));
            Assert.Throws<OperationCanceledException>(() => LoraFileLoader.Load(first, plan, cancellationToken: new(true)));
            Assert.Throws<NotSupportedException>(() => new NativeLoraTensorSource(values, maxSnapshotBytes: 1));
            Assert.Throws<OperationCanceledException>(() => new NativeLoraTensorSource(values, cancellationToken: new(true)));
            Assert.Equal(borrowed, Tensor.TotalCount);
        }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
