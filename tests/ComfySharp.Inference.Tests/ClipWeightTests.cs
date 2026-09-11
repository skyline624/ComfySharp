using System.Buffers.Binary;
using System.Text.Json;
using ComfySharp.Inference;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class ClipWeightTests
{
    private static readonly ClipTextConfig Tiny = new(2, 3, 2, 1, ClipActivation.QuickGelu);
    private sealed record Entry(string Name, long[] Shape, string DType = "F32");

    private static List<Entry> Canonical(bool projection = true) => ClipWeightSchema.Describe(Tiny, projection)
        .Select(p => new Entry(p.Key, p.Value.ToArray())).ToList();

    private static string Fixture(IEnumerable<Entry> entries)
    {
        string path = Path.GetTempFileName();
        var header = new Dictionary<string, object>();
        var payloads = new List<byte[]>();
        long offset = 0;
        foreach (var entry in entries)
        {
            int elements = checked((int)entry.Shape.Aggregate(1L, (a, b) => a * b));
            int width = entry.DType is "F16" or "BF16" ? 2 : entry.DType is "F64" or "I64" ? 8 : 4;
            byte[] bytes = new byte[elements * width];
            for (int i = 0; i < elements; i++)
            {
                // Exactly representable nontrivial values, including asymmetric projection and Q/K/V rows.
                float value = i % 13 + 1;
                if (entry.DType == "F32") BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * width), value);
                if (entry.DType == "F16") BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * width), BitConverter.HalfToUInt16Bits((Half)value));
                if (entry.DType == "BF16") BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * width), (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16));
            }
            header.Add(entry.Name, new { dtype = entry.DType, shape = entry.Shape, data_offsets = new[] { offset, offset + bytes.Length } });
            offset += bytes.Length; payloads.Add(bytes);
        }
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
        using var file = File.Create(path);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)json.Length);
        file.Write(prefix); file.Write(json);
        foreach (var bytes in payloads) file.Write(bytes);
        return path;
    }

    private static void WithFile(IEnumerable<Entry> entries, Action<SafeTensorFile> action)
    {
        string path = Fixture(entries);
        try { using var file = new SafeTensorFile(path); action(file); }
        finally { File.Delete(path); }
    }

    private static List<Entry> OpenClip(bool suffixedProjection = false)
    {
        var result = new List<Entry>();
        foreach (var entry in Canonical())
        {
            string name = entry.Name;
            if (name.Contains("self_attn.q_proj.", StringComparison.Ordinal) || name.Contains("self_attn.k_proj.", StringComparison.Ordinal) || name.Contains("self_attn.v_proj.", StringComparison.Ordinal)) continue;
            name = name.Replace("text_model.embeddings.position_embedding.weight", "positional_embedding", StringComparison.Ordinal)
                .Replace("text_model.embeddings.token_embedding.weight", "token_embedding.weight", StringComparison.Ordinal)
                .Replace("text_model.final_layer_norm.", "ln_final.", StringComparison.Ordinal)
                .Replace("text_model.encoder.layers.", "transformer.resblocks.", StringComparison.Ordinal)
                .Replace("layer_norm1.", "ln_1.", StringComparison.Ordinal).Replace("layer_norm2.", "ln_2.", StringComparison.Ordinal)
                .Replace("mlp.fc1.", "mlp.c_fc.", StringComparison.Ordinal).Replace("mlp.fc2.", "mlp.c_proj.", StringComparison.Ordinal)
                .Replace("self_attn.out_proj.", "attn.out_proj.", StringComparison.Ordinal);
            if (!suffixedProjection && name == ClipWeightSchema.Projection) name = "text_projection";
            result.Add(entry with { Name = name });
        }
        for (int i = 0; i < Tiny.LayerCount; i++)
        {
            result.Add(new($"transformer.resblocks.{i}.attn.in_proj_weight", [6, 2]));
            result.Add(new($"transformer.resblocks.{i}.attn.in_proj_bias", [6]));
        }
        return result;
    }

    [Fact]
    public void StockSchemasHaveExactSourceCountsAndParameterEstimates()
    {
        foreach (var (config, count, parameters) in new[] { (ClipTextConfig.Large, 197, 123650304L), (ClipTextConfig.Giant, 517, 694659840L) })
        {
            var shapes = ClipWeightSchema.Describe(config);
            Assert.Equal(count, shapes.Count);
            Assert.Equal(parameters, shapes.Values.Sum(s => s.Aggregate(1L, (a, b) => a * b)));
        }
    }

    [Theory]
    [InlineData(ClipCheckpointLayout.Canonical, "", false)]
    [InlineData(ClipCheckpointLayout.ClipL, "clip_l.transformer.", false)]
    [InlineData(ClipCheckpointLayout.ClipG, "clip_g.transformer.", false)]
    [InlineData(ClipCheckpointLayout.Sd1, "cond_stage_model.transformer.", false)]
    [InlineData(ClipCheckpointLayout.Sd1, "cond_stage_model.transformer.", true)]
    [InlineData(ClipCheckpointLayout.SdxlL, "conditioner.embedders.0.transformer.", false)]
    public void ExplicitNamespacesAndSd1LegacyKeysMapExactly(ClipCheckpointLayout layout, string prefix, bool old)
    {
        var entries = Canonical().Select(e => e with { Name = prefix + (old ? e.Name.Replace("text_model.", "", StringComparison.Ordinal) : e.Name) }).ToList();
        if (prefix.Length != 0) entries.Add(new("model.diffusion_model.unrelated", [1], "I64"));
        entries.Add(new(prefix + (old ? "embeddings.position_ids" : "text_model.embeddings.position_ids"), [1, 77], "I64"));
        WithFile(entries, file =>
        {
            var plan = ClipCheckpointLoader.Inspect(file, Tiny, layout);
            Assert.Equal(37, plan.Mappings.Count);
            Assert.Single(plan.IgnoredEncoderKeys);
            Assert.All(plan.Mappings, m => Assert.Equal(ClipWeightTransform.Identity, m.Transform));
            Assert.Equal(ClipWeightSchema.Describe(Tiny).Values.Sum(s => s.Aggregate(1L, (a, b) => a * b)) * 4, plan.ResidentBytes);
            Assert.True(plan.TemporaryBytes > 0);
        });
    }

    [Fact]
    public void RejectsMissingUnknownCollisionWrongShapeAndDtypeBeforeLoad()
    {
        var cases = new List<List<Entry>>();
        var missing = Canonical(); missing.RemoveAt(0); cases.Add(missing);
        var unknown = Canonical(); unknown.Add(new("text_model.unknown.weight", [1])); cases.Add(unknown);
        var collision = Canonical(); collision.Add(new("text_projection", [2, 2])); cases.Add(collision);
        var shape = Canonical(); shape[0] = shape[0] with { Shape = [1, 2] }; cases.Add(shape);
        foreach (string dtype in new[] { "F64", "I64", "I32" })
        { var typed = Canonical(); typed[0] = typed[0] with { DType = dtype }; cases.Add(typed); }
        var extraLayer = Canonical(); extraLayer.Add(new("text_model.encoder.layers.2.layer_norm1.weight", [2])); cases.Add(extraLayer);
        foreach (var entries in cases)
            WithFile(entries, file => Assert.Throws<InvalidDataException>(() => ClipCheckpointLoader.Inspect(file, Tiny, ClipCheckpointLayout.Canonical)));
        WithFile(OpenClip().Append(new("text_model.encoder.layers.0.self_attn.q_proj.weight", [2, 2])),
            file => Assert.Contains("collision", Assert.Throws<InvalidDataException>(() => ClipCheckpointLoader.Inspect(file, Tiny, ClipCheckpointLayout.OpenClip)).Message));
    }

    [Fact]
    public void MissingUnusedProjectionAllowedButGAndRequestedProjectionRequireIt()
    {
        WithFile(Canonical(false), file =>
        {
            var plan = ClipCheckpointLoader.Inspect(file, Tiny, ClipCheckpointLayout.Canonical, requireProjection: false);
            Assert.False(plan.HasProjection);
            Assert.Throws<InvalidDataException>(() => ClipCheckpointLoader.Inspect(file, Tiny, ClipCheckpointLayout.Canonical));
            using var bank = ClipCheckpointLoader.Load(file, plan);
            Assert.False(bank.HasProjection);
            Assert.Throws<InvalidOperationException>(() => bank.GetTensor(ClipWeightSchema.Projection));
        });
        WithFile(Canonical(false).Select(e => e with { Name = "clip_g.transformer." + e.Name }), file =>
        {
            Assert.Throws<ArgumentException>(() => ClipCheckpointLoader.Inspect(file, Tiny, ClipCheckpointLayout.ClipG, false));
            Assert.Throws<InvalidDataException>(() => ClipCheckpointLoader.Inspect(file, Tiny, ClipCheckpointLayout.ClipG));
        });
    }

    [Theory]
    [InlineData("F32", false, false)]
    [InlineData("F16", false, false)]
    [InlineData("BF16", false, false)]
    [InlineData("F32", true, false)]
    [InlineData("F32", false, true)]
    public void OpenClipSplitsQkvConvertsStorageAndUsesSourceProjectionOrientation(string dtype, bool suffixed, bool sdxl)
    {
        var entries = OpenClip(suffixed).Select(e => e with { DType = dtype, Name = (sdxl ? "conditioner.embedders.1.model." : "") + e.Name });
        WithFile(entries, file =>
        {
            var plan = ClipCheckpointLoader.Inspect(file, Tiny, sdxl ? ClipCheckpointLayout.SdxlG : ClipCheckpointLayout.OpenClip);
            using var bank = ClipCheckpointLoader.Load(file, plan);
            string block = "text_model.encoder.layers.0.self_attn.";
            Assert.Equal(new float[] { 1, 2, 3, 4 }, bank.GetTensor(block + "q_proj.weight").data<float>().ToArray());
            Assert.Equal(new float[] { 5, 6, 7, 8 }, bank.GetTensor(block + "k_proj.weight").data<float>().ToArray());
            Assert.Equal(new float[] { 9, 10, 11, 12 }, bank.GetTensor(block + "v_proj.weight").data<float>().ToArray());
            Assert.Equal(new float[] { 5, 6 }, bank.GetTensor(block + "v_proj.bias").data<float>().ToArray());
            Assert.Equal(suffixed ? new float[] { 1, 2, 3, 4 } : [1, 3, 2, 4], bank.GetTensor(ClipWeightSchema.Projection).data<float>().ToArray());
            Assert.All(plan.Mappings, m => Assert.Equal(ScalarType.Float32, bank.GetTensor(m.CanonicalName).dtype));
        });
    }

    [Fact]
    public void BankAndRetainedOwnerOutliveReaderAndAmbientScope()
    {
        NativeRuntimeBootstrap.Initialize();
        ClipWeightSet? bank = null;
        WithFile(Canonical(), file =>
        {
            using var scope = NewDisposeScope();
            bank = ClipCheckpointLoader.Load(file, ClipCheckpointLoader.Inspect(file, Tiny, ClipCheckpointLayout.Canonical));
        });
        using var retained = bank!.Retain();
        var tensor = retained.GetTensor(ClipWeightSchema.Projection);
        bank.Dispose(); bank.Dispose();
        Assert.Throws<ObjectDisposedException>(() => bank.Retain());
        Assert.False(tensor.IsInvalid);
        Assert.Equal(new float[] { 1, 2, 3, 4 }, tensor.data<float>().ToArray());
        retained.Dispose(); Assert.True(tensor.IsInvalid);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CancellationAndFailureAfterPartialMaterializationDisposeAllTransferredTensors(bool cancel)
    {
        WithFile(OpenClip(), file =>
        {
            var plan = ClipCheckpointLoader.Inspect(file, Tiny, ClipCheckpointLayout.OpenClip);
            using var cancellation = new CancellationTokenSource();
            Tensor[] acquired = [];
            void Check(IReadOnlyDictionary<string, Tensor> loaded)
            {
                acquired = loaded.Values.ToArray();
                if (loaded.Count != 8) return;
                if (cancel) cancellation.Cancel(); else throw new InvalidOperationException("Injected partial load failure");
            }
            if (cancel) Assert.Throws<OperationCanceledException>(() => ClipCheckpointLoader.Load(file, plan, cancellation.Token, Check));
            else Assert.Throws<InvalidOperationException>(() => ClipCheckpointLoader.Load(file, plan, cancellation.Token, Check));
            Assert.Equal(8, acquired.Length);
            Assert.All(acquired, t => Assert.True(t.IsInvalid));
            using var valid = ClipCheckpointLoader.Load(file, plan);
            Assert.False(valid.GetTensor(ClipWeightSchema.Projection).IsInvalid);
        });
    }

    [Fact]
    public void FactoryValidationFailureLeavesCallerOwnershipAndSuccessDetachesAll()
    {
        NativeRuntimeBootstrap.Initialize();
        Dictionary<string, Tensor> weights;
        ClipWeightSet bank;
        using (var scope = NewDisposeScope())
        {
            weights = ClipWeightSchema.Describe(Tiny).ToDictionary(p => p.Key, p => zeros(p.Value.ToArray()));
            weights.Add("bad", zeros(1));
            Assert.Throws<InvalidDataException>(() => ClipWeightSet.FromOwnedTensors(Tiny, weights));
            Assert.All(weights.Values, t => Assert.False(t.IsInvalid));
            weights.Remove("bad");
            bank = ClipWeightSet.FromOwnedTensors(Tiny, weights);
            weights.Clear(); // Collection is snapshotted.
        }
        var tensor = bank.GetTensor(ClipWeightSchema.Projection);
        Assert.False(tensor.IsInvalid);
        bank.Dispose(); Assert.True(tensor.IsInvalid);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void FactoryRejectsUnfrozenAndWrongComputeDtypeWithoutTakingOwnership(bool gradients)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var weights = ClipWeightSchema.Describe(Tiny).ToDictionary(p => p.Key, p => zeros(p.Value.ToArray()));
        weights[ClipWeightSchema.Projection] = zeros([2, 2], dtype: gradients ? ScalarType.Float32 : ScalarType.Float16, requires_grad: gradients);
        Assert.Throws<InvalidDataException>(() => ClipWeightSet.FromOwnedTensors(Tiny, weights));
        Assert.All(weights.Values, t => Assert.False(t.IsInvalid));
        Assert.Equal(gradients, weights[ClipWeightSchema.Projection].requires_grad);
    }

    [Fact]
    public async Task DisposeAndRetainRaceNeverProducesInvalidRetainedOwner()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var weights = ClipWeightSchema.Describe(Tiny).ToDictionary(p => p.Key, p => zeros(p.Value.ToArray()));
        var bank = ClipWeightSet.FromOwnedTensors(Tiny, weights);
        using var barrier = new Barrier(2);
        var retain = Task.Run(() =>
        {
            barrier.SignalAndWait();
            try { using var owner = bank.Retain(); Assert.False(owner.GetTensor(ClipWeightSchema.Projection).IsInvalid); }
            catch (ObjectDisposedException) { }
        });
        barrier.SignalAndWait(); bank.Dispose();
        await retain;
        Assert.All(weights.Values, t => Assert.True(t.IsInvalid));
    }

    [Fact]
    public void PlanCannotLoadDifferentReaderAndPrecancelledLoadDoesNoMaterialization()
    {
        WithFile(Canonical(), file =>
        {
            var plan = ClipCheckpointLoader.Inspect(file, Tiny, ClipCheckpointLayout.Canonical);
            WithFile(Canonical(), other => Assert.Throws<ArgumentException>(() => ClipCheckpointLoader.Load(other, plan)));
            Assert.Throws<OperationCanceledException>(() => ClipCheckpointLoader.Load(file, plan, new CancellationToken(true), _ => Assert.Fail("Materialized a cancelled load")));
        });
    }
}
