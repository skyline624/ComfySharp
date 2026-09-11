using System.Security.Cryptography;
using System.Text.Json;
using TorchSharp;
using Xunit;
using Xunit.Sdk;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Ownership and integrity tests for diagnostic input copies only.
/// No model forward, source oracle or numerical acceptance profile is involved.</summary>
[Collection("Classical VAE")]
public sealed class SdAlignedDiagnosticInputsTests
{
    public SdAlignedDiagnosticInputsTests() => NativeRuntimeBootstrap.Initialize();

    [Fact]
    public void AlreadyAlignedAliasedInputsStillReceiveIndependentOwnedCopiesWithoutGradients()
    {
        using var scope = NewDisposeScope();
        using var callerGrad = set_grad_enabled(true);
        using var original = full(new long[] { 2, 3 }, 2.5, dtype: ScalarType.Float32, requires_grad: true);
        Assert.True(SdAlignedDiagnosticInputs.Layout(original).aligned64);
        byte[] expected = original.bytes.ToArray();
        using var prepared = new SdAlignedDiagnosticInputs(new(StringComparer.Ordinal)
            { ["latent"] = original, ["context"] = original });
        var first = prepared.Values["latent"];
        var second = prepared.Values["context"];
        Assert.False(SameAddress(original, first));
        Assert.False(SameAddress(original, second));
        Assert.False(SameAddress(first, second));
        foreach (var copy in prepared.Values.Values)
        {
            Assert.Equal(expected, copy.bytes.ToArray());
            Assert.Equal(original.shape, copy.shape);
            Assert.Equal(original.stride(), copy.stride());
            Assert.Equal(0L, copy.storage_offset());
            Assert.True(SdAlignedDiagnosticInputs.Layout(copy).aligned64);
            Assert.False(copy.requires_grad);
        }
        Assert.True(original.requires_grad);
        Assert.True(is_grad_enabled());
        first.fill_(9);
        Assert.Equal(expected, original.bytes.ToArray());
        Assert.Equal(expected, second.bytes.ToArray());
        Assert.NotEqual(expected, first.bytes.ToArray());
    }

    [Fact]
    public void OwnedCopiesSurviveAmbientScopeAndDisposeWithoutTakingBorrowedInputs()
    {
        using var original = full(new long[] { 2, 3 }, 1.25, dtype: ScalarType.Float32);
        byte[] expected = original.bytes.ToArray();
        SdAlignedDiagnosticInputs? prepared = null;
        try
        {
            Tensor copy;
            using (var scope = NewDisposeScope())
            {
                prepared = new(new(StringComparer.Ordinal) { ["latent"] = original });
                copy = prepared.Values["latent"];
            }
            Assert.False(copy.IsInvalid);
            Assert.Equal(expected, copy.bytes.ToArray());
            prepared.Observe();
            prepared.Dispose();
            Assert.True(copy.IsInvalid);
            Assert.Empty(prepared.Values);
            prepared.Dispose();
            Assert.False(original.IsInvalid);
            Assert.Equal(expected, original.bytes.ToArray());
        }
        finally { prepared?.Dispose(); }
    }

    [Fact]
    public void ContiguousNonzeroOffsetIsRejectedWithoutChangingTheBorrowedView()
    {
        using var scope = NewDisposeScope();
        using var allocation = arange(7, dtype: ScalarType.Float32);
        using var view = allocation.narrow(0, 1, 6);
        Assert.True(view.is_contiguous());
        Assert.Equal(1L, view.storage_offset());
        byte[] expected = view.bytes.ToArray();
        Assert.ThrowsAny<XunitException>(() =>
        {
            using var unexpected = new SdAlignedDiagnosticInputs(new(StringComparer.Ordinal) { ["latent"] = view });
        });
        Assert.False(view.IsInvalid);
        Assert.Equal(1L, view.storage_offset());
        Assert.Equal(expected, view.bytes.ToArray());
    }

    [Fact]
    public void NoncontiguousInputIsRejectedWithoutNormalizingItsStorage()
    {
        using var scope = NewDisposeScope();
        using var original = arange(6, dtype: ScalarType.Float32).reshape(2, 3);
        using var view = original.transpose(0, 1);
        var stride = view.stride();
        byte[] expected = original.bytes.ToArray();
        Assert.False(view.is_contiguous());
        Assert.ThrowsAny<XunitException>(() =>
        {
            using var unexpected = new SdAlignedDiagnosticInputs(new(StringComparer.Ordinal) { ["latent"] = view });
        });
        Assert.False(view.IsInvalid);
        Assert.False(view.is_contiguous());
        Assert.Equal(stride, view.stride());
        Assert.Equal(expected, original.bytes.ToArray());
    }

    [Fact]
    public void LaterInputFailureReleasesTheEarlierDetachedCopyImmediately()
    {
        using var scope = NewDisposeScope();
        using var valid = full(new long[] { 2, 3 }, 4, dtype: ScalarType.Float32);
        using var invalid = valid.transpose(0, 1);
        var borrowed = new Dictionary<string, Tensor>(StringComparer.Ordinal)
            { ["first-valid"] = valid, ["second-invalid"] = invalid };
        // Warm the successful path before measuring native tensor wrappers. The
        // invalid second entry then fails after one copy has left the local scope.
        using (var warm = new SdAlignedDiagnosticInputs(new(StringComparer.Ordinal) { ["first-valid"] = valid }))
            warm.Observe();
        long before = Tensor.TotalCount;
        for (int attempt = 0; attempt < 3; attempt++)
        {
            Assert.ThrowsAny<XunitException>(() =>
            {
                using var unexpected = new SdAlignedDiagnosticInputs(borrowed);
            });
            Assert.Equal(before, Tensor.TotalCount);
            Assert.False(valid.IsInvalid);
            Assert.False(invalid.IsInvalid);
        }
        Assert.All(valid.data<float>().ToArray(), value => Assert.Equal(4f, value));
    }

    [Fact]
    public void ObserveRejectsMutationOfTheActuallyPreparedCopy()
    {
        using var scope = NewDisposeScope();
        using var original = full(new long[] { 3 }, 1, dtype: ScalarType.Float32);
        byte[] expected = original.bytes.ToArray();
        using var prepared = new SdAlignedDiagnosticInputs(new(StringComparer.Ordinal) { ["latent"] = original });
        prepared.Observe();
        prepared.Values["latent"].fill_(2);
        Assert.ThrowsAny<XunitException>(prepared.Observe);
        Assert.Equal(expected, original.bytes.ToArray());
        prepared.Values["latent"].copy_(original);
        for (int i = 0; i < 5; i++) prepared.Observe();
        using var evidence = JsonDocument.Parse(JsonSerializer.Serialize(prepared.Evidence()));
        Assert.Equal(6, evidence.RootElement.GetProperty("observations").GetArrayLength());
    }

    [Fact]
    public void ObserveRejectsMutationOfAnOriginalEvenWhenThePreparedCopyIsUnchanged()
    {
        using var scope = NewDisposeScope();
        using var original = full(new long[] { 3 }, 1, dtype: ScalarType.Float32);
        byte[] expected = original.bytes.ToArray();
        using var prepared = new SdAlignedDiagnosticInputs(new(StringComparer.Ordinal) { ["latent"] = original });
        prepared.Observe();
        original.fill_(2);
        Assert.ThrowsAny<XunitException>(prepared.Observe);
        Assert.Equal(expected, prepared.Values["latent"].bytes.ToArray());
        original.fill_(1);
        for (int i = 0; i < 5; i++) prepared.Observe();
        using var evidence = JsonDocument.Parse(JsonSerializer.Serialize(prepared.Evidence()));
        Assert.Equal(6, evidence.RootElement.GetProperty("observations").GetArrayLength());
    }

    [Fact]
    public void EvidenceRequiresExactlySixOrderedSnapshotsOfEveryActualInput()
    {
        using var scope = NewDisposeScope();
        using var latent = full(new long[] { 2, 3 }, 1, dtype: ScalarType.Float32);
        using var timestep = full(new long[] { 1 }, 0.5, dtype: ScalarType.Float32);
        var originals = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["latent"] = latent, ["timesteps"] = timestep };
        using var prepared = new SdAlignedDiagnosticInputs(originals);
        Assert.ThrowsAny<XunitException>(() => prepared.Evidence());
        for (int i = 0; i < 5; i++) prepared.Observe();
        Assert.ThrowsAny<XunitException>(() => prepared.Evidence());
        prepared.Observe();
        using var evidence = JsonDocument.Parse(JsonSerializer.Serialize(prepared.Evidence()));
        var root = evidence.RootElement;
        Assert.Equal("sd-native-aligned-inputs-v1", root.GetProperty("policy").GetString());
        Assert.Equal("nativeEmptyThenCopy", root.GetProperty("allocation").GetString());
        Assert.True(root.GetProperty("independentBuffers").GetBoolean());
        Assert.True(root.GetProperty("originalInputsUnchanged").GetBoolean());
        Assert.Equal(new[] { "latent", "timesteps" }, root.GetProperty("preparationOrder").EnumerateArray().Select(v => v.GetString()));
        Assert.Equal(new[] { "before0", "after0", "before1", "after1", "before2", "after2" },
            root.GetProperty("observationOrder").EnumerateArray().Select(v => v.GetString()));
        var observations = root.GetProperty("observations");
        Assert.Equal(6, observations.GetArrayLength());
        foreach (var observation in observations.EnumerateArray())
        {
            Assert.Equal(originals.Count, observation.EnumerateObject().Count());
            foreach (var (name, original) in originals)
            {
                var record = observation.GetProperty(name);
                Assert.Equal("float32", record.GetProperty("dtype").GetString());
                Assert.Equal(original.shape, record.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()));
                Assert.Equal(original.stride(), record.GetProperty("stride").EnumerateArray().Select(v => v.GetInt64()));
                Assert.Equal(0L, record.GetProperty("storageOffset").GetInt64());
                Assert.Equal(0, record.GetProperty("addressModulo64").GetInt32());
                Assert.True(record.GetProperty("aligned64").GetBoolean());
                Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(original.bytes)), record.GetProperty("sha256").GetString());
            }
        }
        prepared.Observe();
        Assert.ThrowsAny<XunitException>(() => prepared.Evidence());
    }

    private static unsafe bool SameAddress(Tensor first, Tensor second)
    {
        fixed (byte* a = first.bytes)
        fixed (byte* b = second.bytes) return a == b;
    }
}
