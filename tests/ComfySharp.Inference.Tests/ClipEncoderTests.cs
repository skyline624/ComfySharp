using System.Collections;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class ClipEncoderTests
{
    [Theory]
    [InlineData(ClipActivation.QuickGelu)]
    [InlineData(ClipActivation.Gelu)]
    [InlineData(ClipActivation.GeluTanh)]
    public void OutputsSurviveModelAndAmbientScopeAndAreFrozen(ClipActivation activation)
    {
        NativeRuntimeBootstrap.Initialize();
        using var callerGradMode = set_grad_enabled(true);
        ClipForwardResult result;
        float[] expected;
        using (var scope = NewDisposeScope())
        using (var bank = SyntheticWeights(activation: activation))
        using (var encoder = new ClipTextEncoder(bank))
        {
            result = encoder.Forward(Rows(), new() { AllIntermediateLayers = true });
            Assert.True(is_grad_enabled());
            expected = result.FinalHidden.data<float>().ToArray();
            Assert.Equal(new long[] { 1, 77, 4 }, result.FinalHidden.shape);
            Assert.Equal(new long[] { 1, 3, 77, 4 }, result.IntermediateHidden!.shape);
            Assert.Equal(new long[] { 1, 4 }, result.Pooled.shape);
            Assert.Equal(new long[] { 1, 4 }, result.ProjectedPooled!.shape);
            Assert.All(new[] { result.FinalHidden, result.IntermediateHidden, result.Pooled, result.ProjectedPooled },
                value => { Assert.False(value.requires_grad); Assert.Equal(ScalarType.Float32, value.dtype); });
        }
        using (result)
        {
            Assert.Equal(expected, result.FinalHidden.data<float>().ToArray());
            Assert.All(result.Pooled.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
        }
        Assert.True(result.FinalHidden.IsInvalid);
        Assert.True(result.IntermediateHidden!.IsInvalid);
        Assert.True(result.ProjectedPooled!.IsInvalid);
        Assert.True(result.Pooled.IsInvalid);
    }

    [Fact]
    public void ScalarNegativeAndAllLayersAgreeAndFinalNormIsOptionalOnlyForIntermediate()
    {
        using var bank = SyntheticWeights();
        using var encoder = new ClipTextEncoder(bank);
        using var all = encoder.Forward(Rows(), new() { AllIntermediateLayers = true, NormalizeIntermediate = false });
        using var scalar = encoder.Forward(Rows(), new() { IntermediateLayer = -2, NormalizeIntermediate = false });
        using var positive = encoder.Forward(Rows(), new() { IntermediateLayer = 1, NormalizeIntermediate = false });
        using var first = encoder.Forward(Rows(), new() { IntermediateLayer = -3, NormalizeIntermediate = false });
        using var normalized = encoder.Forward(Rows(), new() { IntermediateLayer = -1 });
        using var selected = all.IntermediateHidden!.select(1, 1);
        using var firstSelected = all.IntermediateHidden.select(1, 0);
        Assert.Equal(selected.data<float>().ToArray(), scalar.IntermediateHidden!.data<float>().ToArray());
        Assert.Equal(selected.data<float>().ToArray(), positive.IntermediateHidden!.data<float>().ToArray());
        Assert.Equal(firstSelected.data<float>().ToArray(), first.IntermediateHidden!.data<float>().ToArray());
        Assert.Equal(normalized.FinalHidden.data<float>().ToArray(), normalized.IntermediateHidden!.data<float>().ToArray());
        Assert.Equal(all.Pooled.data<float>().ToArray(), scalar.Pooled.data<float>().ToArray());
        Assert.Equal(all.FinalHidden.data<float>().ToArray(), normalized.FinalHidden.data<float>().ToArray());
    }

    [Fact]
    public void PoolingUsesFirstEosOrZeroAndExplicitCountsKeepZeroAndLeftPaddingQuirks()
    {
        using var bank = SyntheticWeights();
        using var encoder = new ClipTextEncoder(bank);
        var first = Enumerable.Range(0, 77).Select(i => 100 + i).ToArray();
        first[7] = first[12] = 49407;
        var second = Enumerable.Range(0, 77).Select(i => 200 + i).ToArray();
        IReadOnlyList<IReadOnlyList<int>> rows = new[] { first, second };
        using var fallback = encoder.Forward(rows);
        using var counts = encoder.Forward(rows, new() { TokenCounts = new[] { 0, 4 } });
        using var row0 = fallback.FinalHidden.select(0, 0);
        using var row1 = fallback.FinalHidden.select(0, 1);
        using var eos = row0.select(0, 7);
        using var noEos = row1.select(0, 0);
        using var last = row0.select(0, 76);
        using var count = row1.select(0, 3);
        Assert.Equal(eos.data<float>().ToArray().Concat(noEos.data<float>().ToArray()), fallback.Pooled.data<float>().ToArray());
        Assert.Equal(last.data<float>().ToArray().Concat(count.data<float>().ToArray()), counts.Pooled.data<float>().ToArray());
    }

    [Fact]
    public void CausalAndKeyMasksBlockChangesInInvisibleTokens()
    {
        using var bank = SyntheticWeights();
        using var encoder = new ClipTextEncoder(bank);
        var original = (int[])Rows()[0];
        var changed = (int[])original.Clone();
        changed[60] = 893;
        using var a = encoder.Forward(new[] { original });
        using var b = encoder.Forward(new[] { changed });
        using var earlyA = a.FinalHidden.narrow(1, 0, 60);
        using var earlyB = b.FinalHidden.narrow(1, 0, 60);
        Assert.Equal(earlyA.data<float>().ToArray(), earlyB.data<float>().ToArray());

        changed = (int[])original.Clone();
        changed[4] = 893;
        var mask = Enumerable.Repeat(1, 77).ToArray();
        mask[4] = 0;
        var options = new ClipForwardOptions { AttentionMask = new[] { mask } };
        using var maskedA = encoder.Forward(new[] { original }, options);
        using var maskedB = encoder.Forward(new[] { changed }, options);
        using var visibleA = maskedA.FinalHidden.narrow(1, 5, 72);
        using var visibleB = maskedB.FinalHidden.narrow(1, 5, 72);
        Assert.Equal(visibleA.data<float>().ToArray(), visibleB.data<float>().ToArray());
    }

    [Fact]
    public void RejectsInvalidInputsAndMissingRequestedProjection()
    {
        using var bank = SyntheticWeights(projection: false);
        using var encoder = new ClipTextEncoder(bank);
        Assert.False(encoder.HasProjection);
        Assert.Throws<InvalidOperationException>(() => encoder.Forward(Rows()));
        using var valid = encoder.Forward(Rows(), new() { ProjectPooled = false });
        Assert.Null(valid.ProjectedPooled);
        Assert.Throws<ArgumentException>(() => encoder.Forward(Array.Empty<IReadOnlyList<int>>()));
        Assert.Throws<ArgumentException>(() => encoder.Forward(new[] { new int[76] }));
        Assert.Throws<ArgumentException>(() => encoder.Forward(Rows(), new() { AllIntermediateLayers = true, IntermediateLayer = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Forward(Rows(), new() { IntermediateLayer = 3 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Forward(Rows(), new() { IntermediateLayer = -4 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Forward(Rows(), new() { TokenCounts = new[] { -1 } }));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Forward(Rows(), new() { TokenCounts = new[] { 78 } }));
        Assert.Throws<ArgumentException>(() => encoder.Forward(Rows(), new() { AttentionMask = new[] { new int[76] } }));
        Assert.Throws<ArgumentException>(() => encoder.Forward(Rows(), new() { AttentionMask = new[] { Enumerable.Repeat(2, 77).ToArray() } }));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Forward(new[] { Enumerable.Repeat(49408, 77).ToArray() }));
    }

    [Fact]
    public async Task RunningForwardRetainsWeightsWhenBothModelOwnersAreDisposed()
    {
        using var bank = SyntheticWeights();
        using var encoder = new ClipTextEncoder(bank);
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var rows = new InterceptedRows(Rows(), () =>
        {
            entered.Set();
            Assert.True(release.Wait(TimeSpan.FromSeconds(20)));
        });
        var running = Task.Run(() => encoder.Forward(rows));
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(20)));
            encoder.Dispose();
            bank.Dispose();
        }
        finally { release.Set(); }
        using var result = await running;
        Assert.All(result.FinalHidden.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
        Assert.Throws<ObjectDisposedException>(() => encoder.Retain());
        Assert.Throws<ObjectDisposedException>(() => encoder.Forward(Rows()));
    }

    [Fact]
    public void RetainedEncoderAndSubsequentForwardSurviveCancellation()
    {
        using var bank = SyntheticWeights();
        using var original = new ClipTextEncoder(bank);
        using var retained = original.Retain();
        original.Dispose();
        bank.Dispose();
        Assert.Throws<OperationCanceledException>(() => retained.Forward(Rows(), cancellationToken: new(true)));
        using var cancellation = new CancellationTokenSource();
        var rows = new InterceptedRows(Rows(), cancellation.Cancel);
        Assert.Throws<OperationCanceledException>(() => retained.Forward(rows, cancellationToken: cancellation.Token));
        using var a = retained.Forward(Rows());
        using var b = retained.Forward(Rows());
        Assert.Equal(a.FinalHidden.data<float>().ToArray(), b.FinalHidden.data<float>().ToArray());
        a.Dispose();
        Assert.All(b.FinalHidden.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
    }

    private sealed class InterceptedRows(IReadOnlyList<IReadOnlyList<int>> rows, Action intercept) : IReadOnlyList<IReadOnlyList<int>>
    {
        public int Count => rows.Count;
        public IReadOnlyList<int> this[int index] { get { intercept(); return rows[index]; } }
        public IEnumerator<IReadOnlyList<int>> GetEnumerator() => rows.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    internal static IReadOnlyList<IReadOnlyList<int>> Rows() =>
        new[] { Enumerable.Range(0, 77).Select(i => i == 10 ? 49407 : 100 + i).ToArray() };

    // Explicit nondegenerate synthetic parameters support structural/lifetime checks only.
    // Numerical reference values are supplied independently by the frozen Python AST corpus.
    internal static ClipWeightSet SyntheticWeights(bool projection = true, ClipActivation activation = ClipActivation.QuickGelu)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var config = new ClipTextConfig(4, 7, 3, 2, activation);
        var tensors = new Dictionary<string, Tensor>();
        int ordinal = 0;
        foreach (var (name, shape) in ClipWeightSchema.Describe(config, projection))
        {
            int count = checked((int)shape.Aggregate(1L, (a, b) => a * b));
            float offset = name.Contains("norm") && name.EndsWith(".weight") ? 0.8f : 0;
            int seed = ++ordinal;
            var values = Enumerable.Range(0, count).Select(i => offset + (float)(0.13 * Math.Sin(i * 0.71 + seed))).ToArray();
            tensors.Add(name, tensor(values, dtype: ScalarType.Float32, device: CPU).reshape(shape.ToArray()));
        }
        return ClipWeightSet.FromOwnedTensors(config, tensors, projection);
    }
}
