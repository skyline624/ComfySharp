using System.Security.Cryptography;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Only the prospective aligned-input diagnostic owns these copies.</summary>
internal sealed class SdAlignedDiagnosticInputs : IDisposable
{
    internal const string Policy = "sd-native-aligned-inputs-v1";
    private readonly Dictionary<string, Tensor> originals;
    internal Dictionary<string, Tensor> Values { get; } = new(StringComparer.Ordinal);
    internal Dictionary<string, InputLayout> OriginalLayouts { get; } = new(StringComparer.Ordinal);
    private readonly List<Dictionary<string, InputLayout>> observations = [];
    internal sealed record InputLayout(string dtype, long[] shape, long[] stride, long storageOffset,
        int addressModulo64, bool aligned64, string sha256);

    internal SdAlignedDiagnosticInputs(Dictionary<string, Tensor> borrowed)
    {
        originals = borrowed;
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        try
        {
            foreach (var (name, input) in originals)
            {
                var original = Layout(input);
                Assert.Equal(0L, original.storageOffset);
                OriginalLayouts.Add(name, original);
                // Unconditional, one allocation per input. Never branch on original alignment.
                var copy = empty(input.shape, dtype: ScalarType.Float32, device: CPU);
                copy.copy_(input);
                var prepared = Layout(copy);
                Assert.True(SameContentAndLayout(original, prepared));
                Assert.Equal(0, prepared.addressModulo64);
                Assert.Equal(0L, prepared.storageOffset);
                Assert.All(Values.Values, previous => Assert.False(SameAddress(previous, copy)));
                Assert.False(SameAddress(input, copy));
                Values.Add(name, copy.DetachFromDisposeScope());
            }
        }
        catch { Dispose(); throw; }
    }

    internal static unsafe InputLayout Layout(Tensor value)
    {
        Assert.Equal(ScalarType.Float32, value.dtype);
        Assert.Equal(DeviceType.CPU, value.device_type);
        Assert.False(value.is_sparse);
        Assert.True(value.is_contiguous());
        Assert.True(value.numel() > 0);
        int residue;
        fixed (byte* pointer = value.bytes) residue = (int)((nuint)pointer & 63);
        return new("float32", value.shape, value.stride(), value.storage_offset(), residue, residue == 0,
            Convert.ToHexStringLower(SHA256.HashData(value.bytes)));
    }

    private static unsafe bool SameAddress(Tensor first, Tensor second)
    {
        fixed (byte* a = first.bytes)
        fixed (byte* b = second.bytes) return a == b;
    }

    private static bool SameContentAndLayout(InputLayout first, InputLayout second) =>
        first.dtype == second.dtype && first.sha256 == second.sha256 && first.storageOffset == second.storageOffset &&
        first.shape.SequenceEqual(second.shape) && first.stride.SequenceEqual(second.stride);

    // Called immediately before and after the actual forward closure; no native allocation.
    internal void Observe()
    {
        var snapshot = new Dictionary<string, InputLayout>(StringComparer.Ordinal);
        foreach (var (name, value) in Values)
        {
            var current = Layout(value);
            Assert.True(SameContentAndLayout(OriginalLayouts[name], current));
            Assert.Equal(0, current.addressModulo64);
            Assert.True(SameContentAndLayout(OriginalLayouts[name], Layout(originals[name])));
            snapshot.Add(name, current);
        }
        observations.Add(snapshot);
    }

    internal object Evidence()
    {
        Assert.Equal(6, observations.Count);
        return new { policy = Policy, preparationOrder = Values.Keys.ToArray(), allocation = "nativeEmptyThenCopy",
            originalLayouts = OriginalLayouts, observations, observationOrder = new[] { "before0", "after0", "before1", "after1", "before2", "after2" },
            independentBuffers = true, originalInputsUnchanged = true };
    }

    public void Dispose()
    {
        foreach (var value in Values.Values) value.Dispose();
        Values.Clear();
    }
}
