using System.Runtime.InteropServices;
using System.Security.Cryptography;
using ComfySharp.RuntimeProbe;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdSyntheticWeightBuilderTests
{
    [Theory]
    [InlineData("norm.weight", true, 1)]
    [InlineData("norm.bias", true, 1)]
    [InlineData("conv.weight", true, 4)]
    [InlineData("linear.weight", true, 2)]
    [InlineData("case/latent", false, 4)]
    public void ChunksMatchExistingInputRecipeBitForBit(string name, bool parameter, int rank)
    {
        long[] shape = rank switch { 1 => [129], 2 => [17, 31], _ => [2, 3, 3, 3] };
        float[] expected = SdSyntheticInputs.Values(name, shape, parameter);
        var descriptor = SdSyntheticRecipe.Describe(name, shape, parameter);
        foreach (int chunk in new[] { 1, 2, 7, 16, 64, 127, 128, 129, 262144 })
        {
            float[] actual = new float[expected.Length];
            for (int start = 0; start < actual.Length; start += chunk)
                SdSyntheticRecipe.Fill(actual.AsSpan(start, Math.Min(chunk, actual.Length - start)), start, descriptor);
            Assert.Equal(MemoryMarshal.AsBytes(expected.AsSpan()).ToArray(), MemoryMarshal.AsBytes(actual.AsSpan()).ToArray());
        }
    }

    [Fact]
    public void GlobalIndexWrapAndRangeChecksDoNotRestartTheRecipeAtChunks()
    {
        var descriptor = SdSyntheticRecipe.Describe("wrap.weight", [(1L << 32) + 50], true);
        var actual = new float[64];
        long start = (1L << 32) - 17;
        SdSyntheticRecipe.Fill(actual.AsSpan(0, 19), start, descriptor);
        SdSyntheticRecipe.Fill(actual.AsSpan(19), start + 19, descriptor);
        for (int i = 0; i < actual.Length; i++)
        {
            uint word = unchecked((uint)(start + i) * 1664525U + descriptor.Seed);
            float expected = 1 + ((int)(word >> 16) - 32768) * MathF.ScaleB(1, -18);
            Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual[i]));
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => SdSyntheticRecipe.Fill(new float[1], -1, descriptor));
        Assert.Throws<ArgumentOutOfRangeException>(() => SdSyntheticRecipe.Fill(new float[2], descriptor.Elements - 1, descriptor));
    }

    [Fact]
    public void MetadataRejectsInvalidShapesAndBudgetBeforeAllocation()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SdSyntheticRecipe.Describe("x", [0], true));
        Assert.Throws<OverflowException>(() => SdSyntheticRecipe.Describe("x", [long.MaxValue, 2], true));
        Assert.Throws<NotSupportedException>(() => SdSyntheticWeightBuilder.Describe(Schema(("huge", [int.MaxValue]))));
        Assert.Throws<ArgumentException>(() => SdSyntheticWeightBuilder.Describe(Schema()));
        var plan = SdSyntheticWeightBuilder.Describe(Schema(("tiny", [4])));
        int allocations = 0;
        Tensor Allocate(long[] shape) { allocations++; throw new InvalidOperationException(); }
        Assert.Throws<InvalidOperationException>(() => SdSyntheticWeightBuilder.Create(plan, new(15), FakeBank.Take, allocate: Allocate));
        Assert.Throws<ArgumentOutOfRangeException>(() => SdSyntheticWeightBuilder.Create(plan, new(16, 0), FakeBank.Take, allocate: Allocate));
        Assert.Throws<OperationCanceledException>(() => SdSyntheticWeightBuilder.Create(plan, new(16), FakeBank.Take, new(true), Allocate));
        Assert.Equal(0, allocations);
    }

    [Fact]
    public void ReducedUnetUsesAlignedNativeStorageAndRetainsExactInputHashes()
    {
        var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
        var plan = SdSyntheticWeightBuilder.DescribeUnet(config);
        NativeRuntimeBootstrap.Initialize();
        Dictionary<string, Tensor>? originals = null;
        SyntheticBank<UnetWeightSet> generated;
        using (var scope = NewDisposeScope())
            generated = SdSyntheticWeightBuilder.Create(plan, new(plan.ResidentBytes, 257), values =>
            {
                originals = new(values, StringComparer.Ordinal);
                return UnetWeightSet.FromOwnedTensors(config, values);
            });
        using (generated)
        {
            Assert.Equal(686, generated.Parameters.Count);
            using var retained = generated.Weights.Retain();
            foreach (var record in generated.Parameters)
            {
                var tensor = retained.GetTensor(record.Name);
                Assert.Same(originals![record.Name], tensor);
                Assert.True(CpuModelWeightBank.IsAligned(tensor));
                Assert.Equal(record.Shape, tensor.shape);
                Assert.False(tensor.requires_grad);
                string actualHash = Convert.ToHexStringLower(SHA256.HashData(tensor.bytes));
                var expected = SdSyntheticInputs.Values(record.Name, record.Shape, parameter: true);
                Assert.Equal(record.Sha256, actualHash);
                Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(expected.AsSpan()))), actualHash);
            }
        }
    }

    [Fact]
    public void AlignedTransferKeepsTheSameWrappersAndLifetime()
    {
        NativeRuntimeBootstrap.Initialize();
        var plan = SdSyntheticWeightBuilder.Describe(Schema(("a.weight", [17]), ("b.weight", [3, 5])));
        var allocated = new List<Tensor>();
        using (var result = SdSyntheticWeightBuilder.Create(plan, new(plan.ResidentBytes, 4), values =>
        {
            Assert.Equal(allocated, values.Values);
            return FakeBank.Take(values);
        }, allocate: shape =>
        {
            var tensor = empty(shape, dtype: ScalarType.Float32, device: CPU).DetachFromDisposeScope();
            allocated.Add(tensor);
            return tensor;
        }))
        {
            Assert.All(allocated, tensor => Assert.False(tensor.IsInvalid));
            Assert.Equal(2, result.Parameters.Count);
        }
        Assert.All(allocated, tensor => Assert.True(tensor.IsInvalid));
    }

    [Theory]
    [InlineData("after-allocation")]
    [InlineData("middle-first-tensor")]
    [InlineData("after-first-tensor")]
    [InlineData("transfer-throws")]
    [InlineData("schema-rejected")]
    [InlineData("transfer-null")]
    [InlineData("transfer-cancels")]
    public void FailureAtEveryOwnershipBoundaryReleasesDetachedAllocations(string stage)
    {
        NativeRuntimeBootstrap.Initialize();
        var plan = SdSyntheticWeightBuilder.Describe(Schema(("a.weight", [17]), ("b.weight", [3, 5])));
        using var cancellation = new CancellationTokenSource();
        var allocated = new List<Tensor>();
        FakeBank? committed = null;
        var exception = Record.Exception(() => SdSyntheticWeightBuilder.Create(plan, new(plan.ResidentBytes, 4), values =>
        {
            if (stage == "transfer-throws") throw new InvalidDataException("synthetic test");
            if (stage == "transfer-null") return null!;
            if (stage == "schema-rejected")
            {
                using var invalid = UnetWeightSet.FromOwnedTensors(new(32, 16, SdAttentionHeadMode.FixedCount, 4, false), values);
                throw new InvalidOperationException("An incomplete schema was unexpectedly accepted.");
            }
            committed = FakeBank.Take(values);
            cancellation.Cancel();
            return committed;
        }, cancellation.Token, shape =>
        {
            var tensor = empty(shape, dtype: ScalarType.Float32, device: CPU).DetachFromDisposeScope();
            allocated.Add(tensor);
            if (stage == "after-allocation") cancellation.Cancel();
            return tensor;
        }, (name, end) =>
        {
            if (stage == "middle-first-tensor" && name == "a.weight" && end == 4
                || stage == "after-first-tensor" && name == "b.weight") cancellation.Cancel();
        }));
        Assert.NotNull(exception);
        if (stage is "transfer-throws" or "schema-rejected" or "transfer-null") Assert.IsType<InvalidDataException>(exception);
        else Assert.IsAssignableFrom<OperationCanceledException>(exception);
        Assert.NotEmpty(allocated);
        Assert.All(allocated, tensor => Assert.True(tensor.IsInvalid));
        if (committed is not null) Assert.True(committed.Disposed);
    }

    [Theory]
    [InlineData("dtype")]
    [InlineData("shape")]
    [InlineData("alignment")]
    public void MalformedDetachedAllocationIsDisposed(string problem)
    {
        NativeRuntimeBootstrap.Initialize();
        Tensor? allocated = null;
        var plan = SdSyntheticWeightBuilder.Describe(Schema(("a.weight", [17])));
        Assert.ThrowsAny<Exception>(() => SdSyntheticWeightBuilder.Create(plan, new(plan.ResidentBytes), FakeBank.Take,
            allocate: shape =>
            {
                if (problem == "alignment")
                {
                    using var backing = empty([18], dtype: ScalarType.Float32, device: CPU);
                    allocated = backing.narrow(0, 1, 17).DetachFromDisposeScope();
                }
                else allocated = empty(problem == "shape" ? [1] : shape,
                    dtype: problem == "dtype" ? ScalarType.Float64 : ScalarType.Float32, device: CPU).DetachFromDisposeScope();
                return allocated;
            }));
        Assert.NotNull(allocated);
        Assert.True(allocated.IsInvalid);
    }

    private static IReadOnlyDictionary<string, IReadOnlyList<long>> Schema(params (string Name, long[] Shape)[] values)
        => values.ToDictionary(item => item.Name, item => (IReadOnlyList<long>)item.Shape, StringComparer.Ordinal);

    private sealed class FakeBank(Tensor[] tensors) : IDisposable
    {
        internal bool Disposed { get; private set; }
        internal static FakeBank Take(IReadOnlyDictionary<string, Tensor> values) => new(values.Values.ToArray());
        public void Dispose() { foreach (var tensor in tensors) tensor.Dispose(); Disposed = true; }
    }
}
