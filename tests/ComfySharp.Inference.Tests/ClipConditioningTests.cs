using ComfySharp.Tokenization;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class ClipConditioningTests
{
    [Fact]
    public void ProcessTokensPreservesLeadingPaddingInclusiveEosAndUnadjustedCounts()
    {
        var special = ClipSpecialTokens.ForProfile(ClipProfile.Sd1L);
        IReadOnlyList<IReadOnlyList<int>> rows = [
            new[] { 49407, 49407, 49406, 42, 49407, 8, 49407 },
            new[] { 49407, 49407, 49407 },
            new[] { 49406, 2, 3 }
        ];
        var result = ComfyClipEncoder.ProcessTokens(rows, special);
        Assert.Equal(new[] { 0, 0, 1, 1, 1, 0, 0 }, result.Masks[0]);
        Assert.Equal(new[] { 0, 0, 0 }, result.Masks[1]);
        Assert.Equal(new[] { 1, 1, 1 }, result.Masks[2]);
        Assert.Equal(new[] { 3, 0, 3 }, result.Counts);
        // The first row pools at count-1 == 2, not its absolute EOS index == 4.
    }

    [Fact]
    public void ProcessTokensWithoutEosExcludesTheTerminatingPad()
    {
        var result = ComfyClipEncoder.ProcessTokens([new[] { 0, 0, 49406, 4, 0, 7 }], new(49406, null, 0));
        Assert.Equal(new[] { 0, 0, 1, 1, 0, 0 }, result.Masks[0]);
        Assert.Equal(2, result.Counts[0]);
    }

    [Fact]
    public void ProfileDefaultsAndFirstSectionPoolingMatchGraphSelection()
    {
        using var bank = Bank();
        using var graph = new ClipTextEncoder(bank);
        var sections = Sections(Row(17), Row(29));
        var ids = Ids(sections);
        foreach (var profile in Enum.GetValues<ClipProfile>())
        {
            using var wrapper = new ComfyClipEncoder(graph, profile);
            using var result = wrapper.Encode(sections, new() { ReturnAttentionMasks = true });
            var processed = ComfyClipEncoder.ProcessTokens(ids, ClipSpecialTokens.ForProfile(profile));
            using var raw = graph.Forward(ids, new()
            {
                IntermediateLayer = profile == ClipProfile.Sd1L ? null : -2,
                NormalizeIntermediate = profile == ClipProfile.Sd1L,
                ProjectPooled = profile != ClipProfile.Sd1L,
                TokenCounts = processed.Counts
            });
            using var selected = (profile == ClipProfile.Sd1L ? raw.FinalHidden : raw.IntermediateHidden!).reshape(1, 154, 4);
            using var firstPooled = (profile == ClipProfile.Sd1L ? raw.Pooled : raw.ProjectedPooled!)[0];
            using var pooled = firstPooled.unsqueeze(0);
            AssertClose(selected, result.Hidden);
            AssertClose(pooled, result.Pooled);
            Assert.Equal(new long[] { 1, 154 }, result.AttentionMask!.shape);
            Assert.Equal(processed.Masks.SelectMany(x => x).Select(x => (long)x), result.AttentionMask.data<long>().ToArray());
            Assert.False(result.Hidden.requires_grad);
        }
    }

    [Fact]
    public void WeightZeroUsesRightPaddedEmptyBaselineAndLeavesPoolUnchanged()
    {
        using var bank = Bank();
        using var graph = new ClipTextEncoder(bank);
        using var wrapper = new ComfyClipEncoder(graph, ClipProfile.Sd1L);
        var left = Enumerable.Repeat(new ClipTokenWeight(49407, 1), 77).ToArray();
        left[73] = new(49406, 1); left[74] = new(17, 1); left[75] = new(31, 1);
        var weighted = left.ToArray();
        weighted[74] = weighted[74] with { Weight = 0 };
        using var plain = wrapper.Encode(Sections(left));
        using var result = wrapper.Encode(Sections(weighted), new() { ReturnAttentionMasks = true });
        using var empty = wrapper.Encode(Sections(EmptyRow(49407)));
        using var actualPosition = result.Hidden[0, 74];
        using var expectedPosition = empty.Hidden[0, 74];
        AssertClose(expectedPosition, actualPosition);
        AssertClose(plain.Pooled, result.Pooled);
        Assert.Equal(new long[] { 1, 77 }, result.AttentionMask!.shape);
        using var maskSum = result.AttentionMask.sum();
        Assert.Equal(4L, maskSum.item<long>());
        using var untouched = result.Hidden[0, 75];
        using var untouchedExpected = plain.Hidden[0, 75];
        AssertClose(untouchedExpected, untouched);
    }

    [Fact]
    public void AllLayerWeightingUsesLayerIndexedTokenWeightsAndConcatenatesSequenceAxis()
    {
        using var bank = Bank();
        using var graph = new ClipTextEncoder(bank);
        using var wrapper = new ComfyClipEncoder(graph, ClipProfile.Sd1L);
        var row = Row(17);
        row[0] = row[0] with { Weight = 0 };
        // In the source's rank-four path a token weight after the last layer is never applied.
        row[70] = row[70] with { Weight = double.NaN };
        var options = new ClipConditioningOptions { HiddenSelection = ClipHiddenSelection.All };
        using var result = wrapper.Encode(Sections(row, Row(31)), options);
        using var empty = wrapper.Encode(Sections(EmptyRow(49407)), options);
        using var plain = wrapper.Encode(Sections(Row(17)), options);
        Assert.Equal(new long[] { 1, 2, 154, 4 }, result.Hidden.shape);
        using var firstLayer = result.Hidden[0, 0];
        using var first = firstLayer.slice(0, 0, 77, 1);
        using var baseline = empty.Hidden[0, 0];
        AssertClose(baseline, first);
        using var secondLayer = result.Hidden[0, 1];
        using var second = secondLayer.slice(0, 0, 77, 1);
        using var untouched = plain.Hidden[0, 1];
        AssertClose(untouched, second);
    }

    [Fact]
    public void AllLayerZeroMaskRejectsSourceBroadcastMismatchAndAllowsEqualBatchLayers()
    {
        using var bank = Bank();
        using var graph = new ClipTextEncoder(bank);
        using var wrapper = new ComfyClipEncoder(graph, ClipProfile.Sd1L);
        var options = new ClipConditioningOptions { HiddenSelection = ClipHiddenSelection.All, ZeroOutMasked = true };
        Assert.Throws<ArgumentException>(() => wrapper.Encode(Sections(Row(17), Row(21), Row(31)), options));
        using var result = wrapper.Encode(Sections(Row(17), Row(21)), options);
        Assert.Equal(new long[] { 1, 2, 154, 4 }, result.Hidden.shape);
    }

    [Fact]
    public void SdxlCutsSequenceAndConcatenatesLThenGWithGProjectedPool()
    {
        using var lBank = Bank(projection: false);
        using var gBank = Bank(offset: 0.7f);
        using var lGraph = new ClipTextEncoder(lBank);
        using var gGraph = new ClipTextEncoder(gBank);
        using var sdxl = new ComfySdxlEncoder(lGraph, gGraph);
        using var lWrapper = new ComfyClipEncoder(lGraph, ClipProfile.SdXlL);
        using var gWrapper = new ComfyClipEncoder(gGraph, ClipProfile.SdXlG);
        var lTokens = Sections(Row(17), Row(31));
        var gTokens = Sections(Row(29, 0));
        using var result = sdxl.Encode(lTokens, gTokens);
        using var lResult = lWrapper.Encode(lTokens, new() { ProjectPooled = false });
        using var gResult = gWrapper.Encode(gTokens);
        Assert.Equal(new long[] { 1, 77, 8 }, result.Hidden.shape);
        using var actualL = result.Hidden.slice(-1, 0, 4, 1);
        using var expectedL = lResult.Hidden.slice(1, 0, 77, 1);
        using var actualG = result.Hidden.slice(-1, 4, 8, 1);
        AssertClose(expectedL, actualL);
        AssertClose(gResult.Hidden, actualG);
        AssertClose(gResult.Pooled, result.Pooled);
        Assert.Null(result.AttentionMask);
        Assert.Throws<InvalidOperationException>(() => lWrapper.Encode(lTokens));
        Assert.Throws<ArgumentException>(() => sdxl.Encode(lTokens, gTokens, new() { G = new() { ProjectPooled = false } }));
    }

    [Fact]
    public void OwnershipOutlivesModelsAndAmbientScopeAndInvalidInputsAreDiagnosed()
    {
        ClipConditioningResult result;
        ComfyClipEncoder retained;
        NativeRuntimeBootstrap.Initialize();
        using (var scope = NewDisposeScope())
        using (var bank = Bank())
        using (var graph = new ClipTextEncoder(bank))
        using (var wrapper = new ComfyClipEncoder(graph, ClipProfile.Sd1L))
        {
            retained = wrapper.Retain();
            result = wrapper.Encode(Sections(Row(17)));
            Assert.Throws<ArgumentException>(() => wrapper.Encode(Array.Empty<IReadOnlyList<ClipTokenWeight>>()));
            Assert.Throws<ArgumentException>(() => wrapper.Encode(Sections(new ClipTokenWeight[79])));
            Assert.Throws<ArgumentException>(() => wrapper.Encode(Sections(Row(17)), new() { HiddenSelection = ClipHiddenSelection.Hidden }));
            Assert.Throws<OperationCanceledException>(() => wrapper.Encode(Sections(Row(17)), cancellationToken: new(true)));
        }
        using (retained)
        using (result)
        using (var repeated = retained.Encode(Sections(Row(17))))
        {
            AssertClose(result.Hidden, repeated.Hidden);
            Assert.False(result.Hidden.IsInvalid);
        }
        Assert.True(result.Hidden.IsInvalid);
        Assert.Throws<ObjectDisposedException>(() => retained.Encode(Sections(Row(17))));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SdxlRejectsBranchMaskExtrasBeforeExecutingEitherBranch(bool requestOnL)
    {
        using var bank = Bank();
        using var graph = new ClipTextEncoder(bank);
        using var sdxl = new ComfySdxlEncoder(graph, graph);
        var requested = new ClipConditioningOptions { ReturnAttentionMasks = true };
        var options = requestOnL ? new ClipSdxlConditioningOptions { L = requested }
            : new ClipSdxlConditioningOptions { G = requested };
        var error = Assert.Throws<ArgumentException>(() => sdxl.Encode(Sections(Row(17)), Sections(Row(29, 0)), options));
        Assert.Contains("attention-mask extras", error.Message);
        // Invalid sections are a sentinel: neither branch may even begin token processing.
        var early = Assert.Throws<ArgumentException>(() => sdxl.Encode(Sections(), Sections(), options));
        Assert.Contains("attention-mask extras", early.Message);
    }

    private static IReadOnlyList<IReadOnlyList<ClipTokenWeight>> Sections(params ClipTokenWeight[][] rows) => rows;
    private static IReadOnlyList<IReadOnlyList<int>> Ids(IReadOnlyList<IReadOnlyList<ClipTokenWeight>> rows) =>
        rows.Select(row => (IReadOnlyList<int>)row.Select(x => x.Id).ToArray()).ToArray();
    private static ClipTokenWeight[] EmptyRow(int pad)
    {
        var row = Enumerable.Repeat(new ClipTokenWeight(pad, 1), 77).ToArray();
        row[0] = new(49406, 1); row[1] = new(49407, 1);
        return row;
    }
    private static ClipTokenWeight[] Row(int token, int pad = 49407)
    {
        var row = EmptyRow(pad);
        row[1] = new(token, 1); row[2] = new(token + 1, 1); row[3] = new(49407, 1);
        return row;
    }

    private static void AssertClose(Tensor expected, Tensor actual)
    {
        Assert.Equal(expected.shape, actual.shape);
        using var leftTensor = expected.contiguous();
        using var rightTensor = actual.contiguous();
        var left = leftTensor.data<float>().ToArray();
        var right = rightTensor.data<float>().ToArray();
        for (int i = 0; i < left.Length; i++)
            Assert.True(Math.Abs(left[i] - right[i]) <= 3e-5 + 3e-5 * Math.Abs(left[i]), $"Value {i}: expected {left[i]}, actual {right[i]}.");
    }

    // Deliberately synthetic small graph; numerical source qualification uses the independent root corpus.
    private static ClipWeightSet Bank(bool projection = true, float offset = 0)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var config = new ClipTextConfig(4, 7, 2, 2, ClipActivation.QuickGelu);
        var tensors = new Dictionary<string, Tensor>();
        void Add(string name, params long[] shape)
        {
            int size = checked((int)shape.Aggregate(1L, (a, b) => a * b));
            int salt = name.Sum(c => c);
            var values = Enumerable.Range(0, size).Select(i =>
                (float)(Math.Sin((i + salt) * 0.17) * 0.08 + offset * Math.Cos(i * 0.3)
                + (name.Contains("norm", StringComparison.Ordinal) && name.EndsWith("weight", StringComparison.Ordinal) ? 1 : 0))).ToArray();
            tensors.Add(name, tensor(values, dtype: ScalarType.Float32, device: CPU).reshape(shape));
        }
        Add("text_model.embeddings.token_embedding.weight", 49408, 4);
        Add("text_model.embeddings.position_embedding.weight", 77, 4);
        Add("text_model.final_layer_norm.weight", 4); Add("text_model.final_layer_norm.bias", 4);
        if (projection) Add("text_projection.weight", 4, 4);
        for (int i = 0; i < 2; i++)
        {
            string p = $"text_model.encoder.layers.{i}";
            foreach (string norm in new[] { "layer_norm1", "layer_norm2" })
            { Add(p + "." + norm + ".weight", 4); Add(p + "." + norm + ".bias", 4); }
            foreach (string linear in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
            { Add(p + ".self_attn." + linear + ".weight", 4, 4); Add(p + ".self_attn." + linear + ".bias", 4); }
            Add(p + ".mlp.fc1.weight", 7, 4); Add(p + ".mlp.fc1.bias", 7);
            Add(p + ".mlp.fc2.weight", 4, 7); Add(p + ".mlp.fc2.bias", 4);
        }
        return ClipWeightSet.FromOwnedTensors(config, tensors, projection);
    }
}
