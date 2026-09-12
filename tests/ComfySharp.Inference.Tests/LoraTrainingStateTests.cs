using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Inference;
using ComfySharp.Nodes.Tensor;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraTrainingStateTests
{
    [Theory]
    [InlineData(0, ScalarType.BFloat16)]
    [InlineData(1, ScalarType.Float32)]
    public void Final_cast_matches_frozen_source_bits_and_detaches_from_training(int index, ScalarType dtype)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            using var resource = typeof(LoraTrainingStateTests).Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora-training-state.json")!;
            using var buffer = new MemoryStream(); resource.CopyTo(buffer);
            byte[] fixture = buffer.ToArray();
            Assert.Equal("f066e79f9052d9ec7d629834a9a82a418ea446d76cf209bb430b8110ae495ae3", Convert.ToHexStringLower(SHA256.HashData(fixture)));
            using var document = JsonDocument.Parse(fixture);
            var row = document.RootElement.GetProperty("cases")[index];
            using var patch = new TrainableDifferencePatch(tensor(row.GetProperty("input").EnumerateArray().Select(n => n.GetSingle()).ToArray()));
            using var state = LoraTrainingState.Capture(new Dictionary<string, TrainableDifferencePatch> { ["norm.weight"] = patch }, dtype);
            var value = state.Tensors["diffusion_model.norm.diff"];
            Assert.Equal(row.GetProperty("outputHex").GetString(), Convert.ToHexStringLower(value.bytes));
            Assert.Equal(row.GetProperty("outputSha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(value.bytes)));
            Assert.False(value.requires_grad); Assert.True(patch.Difference.requires_grad);
            using (var noGrad = no_grad()) patch.Difference.fill_(99);
            patch.Dispose();
            Assert.Equal(row.GetProperty("outputHex").GetString(), Convert.ToHexStringLower(value.bytes));
            state.Dispose(); Assert.Throws<ObjectDisposedException>(() => state.Tensors);
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory]
    [InlineData(ScalarType.Float32)]
    [InlineData(ScalarType.BFloat16)]
    public void All_686_targets_publish_1250_independent_state_values(ScalarType dtype)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            using var adapters = new SdTrainableAdapterSet(new(32, 16, SdAttentionHeadMode.FixedCount, 4, false), 2, 317, CPU);
            using var state = LoraTrainingState.Capture(adapters.Patches, dtype);
            Assert.Equal(1250, state.Tensors.Count);
            Assert.Equal(282, state.Tensors.Keys.Count(k => k.EndsWith(".alpha", StringComparison.Ordinal)));
            Assert.Equal(109, state.Tensors.Keys.Count(k => k.EndsWith(".diff", StringComparison.Ordinal)));
            Assert.Equal(295, state.Tensors.Keys.Count(k => k.EndsWith(".diff_b", StringComparison.Ordinal)));
            adapters.Dispose();
            foreach (var tensor in state.Tensors.Values)
            {
                Assert.Equal(dtype, tensor.dtype); Assert.False(tensor.requires_grad);
                Assert.True(tensor.is_contiguous()); Assert.Equal(CPU.type, tensor.device_type);
            }
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    private sealed class MemoryStore : IStreamingFileStore
    {
        public byte[]? Bytes;
        public IReadOnlyList<string> PrepareDirectory(string type, string subfolder, CancellationToken cancellationToken = default) => [];
        public ValueTask WriteAsync(ImageFileDescriptor file, ReadOnlyMemory<byte> bytes, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public ValueTask WriteAtomicAsync(ImageFileDescriptor file, Action<Stream, CancellationToken> write, CancellationToken cancellationToken = default)
        {
            using var stream = new MemoryStream(); write(stream, cancellationToken); Bytes = stream.ToArray(); return ValueTask.CompletedTask;
        }
    }

    [Fact]
    public async Task Trained_state_survives_producer_and_training_disposal_then_reaches_SaveLoRA()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var consumer = new RuntimeNodeContext())
        {
            RuntimeValue retained;
            using (var producer = new RuntimeNodeContext())
            using (var scope = NewDisposeScope())
            using (var patch = new TrainableDifferencePatch(tensor(new[] { 1f, 2f })))
            using (var optimizer = new LoraTrainingOptimizer(new[] { patch }, "SGD", .1))
            {
                using var loss = patch.Difference.square().sum(); optimizer.Accumulate(loss); optimizer.Step();
                var value = TrainingNodeValues.CaptureAdapters(producer, new Dictionary<string, TrainableDifferencePatch> { ["layer.bias"] = patch }, ScalarType.Float32);
                retained = consumer.Retain(value);
                using var noGrad = no_grad(); patch.Difference.fill_(99);
            }
            var result = retained.Properties["diffusion_model.layer.diff_b"].GetNative<Tensor>();
            Assert.Equal(new[] { .8f, 1.6f }, result.data<float>().ToArray()); Assert.False(result.requires_grad);
            using var saveContext = new RuntimeNodeContext(); var store = new MemoryStore();
            await new SaveLoraNode(store).ExecuteAsync(saveContext, new Dictionary<string, RuntimeValue>
            {
                ["lora"] = retained, ["prefix"] = saveContext.Json(JsonValue.Create("trained"))
            }, default);
            byte[] encoded = store.Bytes!;
            int payload = 8 + checked((int)BinaryPrimitives.ReadUInt64LittleEndian(encoded));
            Assert.Equal(result.bytes.ToArray(), encoded[payload..]);
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Failed_publication_and_admission_do_not_leak_or_take_training_ownership()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        using (var patch = new TrainableDifferencePatch(ones(2)))
        using (var closed = new RuntimeNodeContext())
        {
            var targets = new Dictionary<string, TrainableDifferencePatch> { ["layer.weight"] = patch };
            long borrowed = Tensor.TotalCount;
            Assert.Throws<ArgumentException>(() => LoraTrainingState.Capture(targets, ScalarType.Float16));
            Assert.Throws<NotSupportedException>(() => LoraTrainingState.Capture(targets, maxSnapshotBytes: 1));
            Assert.Throws<OperationCanceledException>(() => LoraTrainingState.Capture(targets, cancellationToken: new(true)));
            Assert.Throws<ArgumentException>(() => LoraTrainingState.Capture(new Dictionary<string, TrainableDifferencePatch> { ["invalid"] = patch }));
            closed.Dispose();
            Assert.Throws<ObjectDisposedException>(() => TrainingNodeValues.CaptureAdapters(closed, targets));
            Assert.Equal(borrowed, Tensor.TotalCount);
            Assert.True(patch.Difference.requires_grad);
        }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
