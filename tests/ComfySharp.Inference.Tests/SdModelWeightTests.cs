using System.Buffers.Binary;
using System.Text.Json;
using System.Text.RegularExpressions;
using ComfySharp.Inference;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdModelWeightTests
{
    private static readonly SdUnetConfig Unet = new(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
    private static readonly ClassicalVaeConfig Vae = new(32);
    private sealed record Entry(string Name, long[] Shape, string DType = "F32");
    private static IReadOnlyDictionary<string, IReadOnlyList<long>> Schema(bool vae) => vae ? ClassicalVaeWeightSchema.Describe(Vae) : UnetWeightSchema.Describe(Unet);
    private static List<Entry> Entries(bool vae) => Schema(vae).Select(p => new Entry(p.Key, p.Value.ToArray())).ToList();

    // Header and zero payload are generated without Torch, including for malformed metadata tests.
    // A short asymmetric prefix is enough to verify dtype conversion / matrix reshape orientation.
    private static string Fixture(IEnumerable<Entry> entries)
    {
        string path = Path.GetTempFileName();
        var header = new Dictionary<string, object>();
        var starts = new List<(long Start, long Elements, string DType)>();
        long offset = 0;
        foreach (var entry in entries)
        {
            long elements = entry.Shape.Aggregate(1L, (n, d) => checked(n * d));
            int width = entry.DType is "F16" or "BF16" ? 2 : entry.DType is "F64" or "I64" ? 8 : 4;
            long end = checked(offset + elements * width);
            header.Add(entry.Name, new { dtype = entry.DType, shape = entry.Shape, data_offsets = new[] { offset, end } });
            starts.Add((offset, elements, entry.DType)); offset = end;
        }
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
        using var stream = File.Create(path);
        Span<byte> prefix = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)json.Length);
        stream.Write(prefix); stream.Write(json); stream.SetLength(8 + json.Length + offset);
        foreach (var (start, elements, dtype) in starts)
        {
            int width = dtype is "F16" or "BF16" ? 2 : dtype is "F64" or "I64" ? 8 : 4;
            byte[] values = new byte[Math.Min(elements, 16) * width];
            for (int i = 0; i < values.Length / width; i++)
            {
                float value = (i - 7) * 0.25f;
                if (dtype == "F32") BinaryPrimitives.WriteSingleLittleEndian(values.AsSpan(i * width), value);
                if (dtype == "F16") BinaryPrimitives.WriteUInt16LittleEndian(values.AsSpan(i * width), BitConverter.HalfToUInt16Bits((Half)value));
                if (dtype == "BF16") BinaryPrimitives.WriteUInt16LittleEndian(values.AsSpan(i * width), (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16));
            }
            stream.Position = 8 + json.Length + start; stream.Write(values);
        }
        return path;
    }

    private static void WithFile(IEnumerable<Entry> entries, Action<SafeTensorFile> action)
    {
        string path = Fixture(entries);
        try { using var file = new SafeTensorFile(path); action(file); }
        finally { File.Delete(path); }
    }

    // Independent names following the frozen conversion maps; no production mapper is called here.
    private static string DiffusersVae(string name, bool legacyAttention)
    {
        name = Regex.Replace(name, @"encoder\.down\.(\d)\.block\.(\d)\.", "encoder.down_blocks.$1.resnets.$2.");
        name = Regex.Replace(name, @"decoder\.up\.(\d)\.block\.(\d)\.", m => $"decoder.up_blocks.{3 - int.Parse(m.Groups[1].Value)}.resnets.{m.Groups[2].Value}.");
        name = Regex.Replace(name, @"encoder\.down\.(\d)\.downsample\.", "encoder.down_blocks.$1.downsamplers.0.");
        name = Regex.Replace(name, @"decoder\.up\.(\d)\.upsample\.", m => $"decoder.up_blocks.{3 - int.Parse(m.Groups[1].Value)}.upsamplers.0.");
        name = name.Replace(".nin_shortcut.", ".conv_shortcut.").Replace(".norm_out.", ".conv_norm_out.")
            .Replace(".mid.block_1.", ".mid_block.resnets.0.").Replace(".mid.block_2.", ".mid_block.resnets.1.");
        if (name.Contains(".mid.attn_1."))
        {
            name = name.Replace(".mid.attn_1.", ".mid_block.attentions.0.").Replace(".norm.", ".group_norm.");
            foreach (var (from, modern, legacy) in new[] { ("q", "to_q", "query"), ("k", "to_k", "key"), ("v", "to_v", "value"), ("proj_out", "to_out.0", "proj_attn") })
                name = name.Replace("." + from + ".", "." + (legacyAttention ? legacy : modern) + ".");
        }
        return name;
    }

    private static List<Entry> DiffusersVaeEntries(bool legacy = false) => Entries(true).Select(e => e with
    {
        Name = DiffusersVae(e.Name, legacy),
        Shape = e.Name.Contains(".mid.attn_1.") && e.Shape.Length == 4 ? e.Shape[..2] : e.Shape
    }).ToList();

    private static string DiffusersUnet(string name)
    {
        foreach (var (canonical, alias) in new[] { ("time_embed.0.", "time_embedding.linear_1."), ("time_embed.2.", "time_embedding.linear_2."),
            ("input_blocks.0.0.", "conv_in."), ("out.0.", "conv_norm_out."), ("out.2.", "conv_out.") })
            if (name.StartsWith(canonical)) return alias + name[canonical.Length..];
        var match = Regex.Match(name, @"^(input_blocks|output_blocks|middle_block)\.(\d+)\.(.+)$");
        Assert.True(match.Success);
        int index = int.Parse(match.Groups[2].Value);
        string tail = match.Groups[3].Value, prefix;
        bool residual;
        if (match.Groups[1].Value == "middle_block")
        {
            residual = index != 1;
            prefix = residual ? $"mid_block.resnets.{index / 2}." : "mid_block.attentions.0.";
        }
        else
        {
            int module = tail[0] - '0'; tail = tail[2..];
            if (match.Groups[1].Value == "input_blocks")
            {
                if (index % 3 == 0) return $"down_blocks.{index / 3 - 1}.downsamplers.0.conv." + tail[3..];
                residual = module == 0;
                prefix = $"down_blocks.{(index - 1) / 3}.{(residual ? "resnets" : "attentions")}.{(index - 1) % 3}.";
            }
            else
            {
                if (tail.StartsWith("conv.")) return $"up_blocks.{index / 3}.upsamplers.0." + tail;
                residual = module == 0;
                prefix = $"up_blocks.{index / 3}.{(residual ? "resnets" : "attentions")}.{index % 3}.";
            }
        }
        if (residual)
            foreach (var (canonical, alias) in new[] { ("in_layers.0.", "norm1."), ("in_layers.2.", "conv1."), ("emb_layers.1.", "time_emb_proj."),
                ("out_layers.0.", "norm2."), ("out_layers.3.", "conv2."), ("skip_connection.", "conv_shortcut.") })
                if (tail.StartsWith(canonical)) return prefix + alias + tail[canonical.Length..];
        Assert.False(residual);
        return prefix + tail;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AllDiffusersUnetNamesMapExactlyWithoutGuessingProjectionRank(bool linear)
    {
        var config = linear ? Unet with { UseLinearProjection = true, HeadMode = SdAttentionHeadMode.FixedSize, HeadParameter = 8 } : Unet;
        var schema = UnetWeightSchema.Describe(config);
        var entries = schema.Select(e => new Entry(DiffusersUnet(e.Key), e.Value.ToArray())).ToList();
        WithFile(entries, file =>
        {
            var plan = UnetCheckpointLoader.Inspect(file, config, UnetCheckpointLayout.Diffusers);
            Assert.Equal(686, plan.Mappings.Count);
            foreach (var map in plan.Mappings)
            {
                Assert.Equal(DiffusersUnet(map.CanonicalName), map.SourceName);
                Assert.Equal(SdModelWeightTransform.Identity, map.Transform);
            }
            int width = linear ? 2 : 4;
            Assert.Equal(width, plan.Mappings.Single(m => m.CanonicalName == "input_blocks.1.1.proj_in.weight").SourceShape.Count);
            Assert.Throws<InvalidDataException>(() => UnetCheckpointLoader.Inspect(file, config with { UseLinearProjection = !linear }, UnetCheckpointLayout.Diffusers));
        });
        entries.Add(new("up_blocks.3.upsamplers.0.conv.weight", [32, 32, 3, 3]));
        WithFile(entries, file => Assert.Throws<InvalidDataException>(() => UnetCheckpointLoader.Inspect(file, config, UnetCheckpointLayout.Diffusers)));
    }

    [Fact]
    public void CompleteStockSchemasHavePinnedTensorCountsAndParameterBytes()
    {
        foreach (var (config, parameters) in new[] { (SdUnetConfig.Sd15, 859520964L), (SdUnetConfig.Sd2, 865910724L) })
        {
            var schema = UnetWeightSchema.Describe(config);
            Assert.Equal(686, schema.Count);
            Assert.Equal(parameters, schema.Values.Sum(s => s.Aggregate(1L, (n, d) => n * d)));
            Assert.Equal(config.UseLinearProjection ? new long[] { 320, 320 } : [320, 320, 1, 1], schema["input_blocks.1.1.proj_in.weight"]);
            Assert.Equal(new long[] { 1280, 1920, 3, 3 }, schema["output_blocks.5.0.in_layers.2.weight"]);
            Assert.Equal(new long[] { 640, 1920, 3, 3 }, schema["output_blocks.6.0.in_layers.2.weight"]);
            Assert.Equal(new long[] { 320, config.ContextSize }, schema["input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"]);
            Assert.DoesNotContain("input_blocks.10.1.norm.weight", schema.Keys);
            Assert.DoesNotContain("output_blocks.0.1.norm.weight", schema.Keys);
            Assert.DoesNotContain("input_blocks.1.1.transformer_blocks.0.attn1.to_q.bias", schema.Keys);
        }
        var vae = ClassicalVaeWeightSchema.Describe(ClassicalVaeConfig.Stock);
        Assert.Equal(248, vae.Count);
        Assert.Equal(83653863L, vae.Values.Sum(s => s.Aggregate(1L, (n, d) => n * d)));
        Assert.Equal(106, vae.Keys.Count(k => k.StartsWith("encoder.")));
        Assert.Equal(138, vae.Keys.Count(k => k.StartsWith("decoder.")));
        Assert.Equal(new long[] { 256, 512, 1, 1 }, vae["decoder.up.1.block.0.nin_shortcut.weight"]);
        Assert.Equal(new long[] { 8, 8, 1, 1 }, vae["quant_conv.weight"]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SchemaAndMappingsAreImmutableAndNamespaceInspectionIsMetadataOnly(bool vae)
    {
        var schema = Schema(vae);
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, IReadOnlyList<long>>)schema).Clear());
        Assert.Throws<NotSupportedException>(() => ((IList<long>)schema.First().Value)[0] = 1);
        string prefix = vae ? "first_stage_model." : "model.diffusion_model.";
        var entries = Entries(vae).Select(e => e with { Name = prefix + e.Name }).Append(new("unrelated.encoder.weight", [3], "I64"));
        WithFile(entries, file =>
        {
            var maps = vae ? ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.FirstStageModel).Mappings
                : UnetCheckpointLoader.Inspect(file, Unet, UnetCheckpointLayout.ModelDiffusionModel).Mappings;
            Assert.Equal(schema.Count, maps.Count);
            Assert.All(maps, m => Assert.Equal(SdModelWeightTransform.Identity, m.Transform));
            Assert.Throws<NotSupportedException>(() => ((IList<SdModelWeightMapping>)maps).Clear());
            Assert.Throws<NotSupportedException>(() => ((IList<long>)maps[0].SourceShape)[0] = 0);
            Assert.Throws<NotSupportedException>(() => ((IList<long>)maps[0].CanonicalShape)[0] = 0);
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MissingUnknownRankDtypeExtraLayerAndUnsupportedArchitectureAreRejected(bool vae)
    {
        var missing = Entries(vae); missing.RemoveAt(0);
        var wrong = Entries(vae); wrong[0] = wrong[0] with { Shape = [wrong[0].Shape.Aggregate(1L, (a, b) => a * b)] };
        var dtype = Entries(vae); dtype[0] = dtype[0] with { DType = "I64" };
        var extra = Entries(vae); extra.Add(new(vae ? "encoder.down.4.block.0.norm1.weight" : "input_blocks.1.1.transformer_blocks.1.norm1.weight", [32]));
        var architecture = Entries(vae); int index = architecture.FindIndex(e => e.Name == (vae ? "post_quant_conv.weight" : "input_blocks.0.0.weight"));
        architecture[index] = architecture[index] with { Shape = vae ? [8, 4, 1, 1] : [32, 9, 3, 3] };
        foreach (var entries in new[] { missing, wrong, dtype, extra, architecture })
            WithFile(entries, file => Assert.Throws<InvalidDataException>(() => Inspect(file, vae)));
    }

    private static void Inspect(SafeTensorFile file, bool vae)
    {
        if (vae) ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.Canonical);
        else UnetCheckpointLoader.Inspect(file, Unet, UnetCheckpointLayout.Canonical);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DiffusersVaeAliasesAndDecoderOrderRequireDenseAttentionMatrices(bool legacy)
    {
        WithFile(DiffusersVaeEntries(legacy), file =>
        {
            var plan = ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.Diffusers);
            Assert.Equal(248, plan.Mappings.Count);
            Assert.Equal(8, plan.Mappings.Count(m => m.Transform == SdModelWeightTransform.Reshape));
            Assert.Equal("decoder.up.3.block.0.conv1.weight", plan.Mappings.Single(m => m.SourceName == "decoder.up_blocks.0.resnets.0.conv1.weight").CanonicalName);
            Assert.Equal("decoder.up.0.block.2.norm2.bias", plan.Mappings.Single(m => m.SourceName == "decoder.up_blocks.3.resnets.2.norm2.bias").CanonicalName);
        });
        var wrongRank = DiffusersVaeEntries(legacy); int i = wrongRank.FindIndex(e => e.Name.Contains("attentions.0.") && e.Shape.Length == 2);
        wrongRank[i] = wrongRank[i] with { Shape = [128, 128, 1, 1] };
        WithFile(wrongRank, file => Assert.Throws<InvalidDataException>(() => ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.Diffusers)));
    }

    [Theory]
    [InlineData("F32")]
    [InlineData("F16")]
    [InlineData("BF16")]
    public void VaeDenseAttentionIsReshapedWithoutTransposeAndStorageConvertsToF32(string dtype)
    {
        WithFile(DiffusersVaeEntries().Select(e => e with { DType = dtype }), file =>
        {
            var plan = ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.Diffusers);
            Assert.Equal(plan.ResidentBytes * (dtype == "F32" ? 1 : 0.5), (double)plan.SourceBytes);
            Assert.True(plan.TemporaryBytes > 0);
            using var weights = ClassicalVaeCheckpointLoader.Load(file, plan);
            using var scope = NewDisposeScope();
            var q = weights.GetTensor("encoder.mid.attn_1.q.weight");
            Assert.Equal(ScalarType.Float32, q.dtype);
            Assert.Equal(new long[] { 128, 128, 1, 1 }, q.shape);
            Assert.Equal(-1.5f, q.flatten()[1].item<float>());
            Assert.Equal(0f, q.flatten()[128].item<float>());
            Assert.False(q.requires_grad);
        });
    }

    [Fact]
    public void VaeNestedQuantAliasesAreAcceptedButCollisionsAndMissingPostQuantAreRejected()
    {
        var entries = Entries(true).Select(e => e with { Name = e.Name.StartsWith("quant_conv.") ? "encoder." + e.Name : e.Name.StartsWith("post_quant_conv.") ? "decoder." + e.Name : e.Name }).ToList();
        WithFile(entries, file => Assert.Equal(248, ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.Canonical).Mappings.Count));
        entries.Add(new("quant_conv.weight", [8, 8, 1, 1]));
        WithFile(entries, file => Assert.Contains("collision", Assert.Throws<InvalidDataException>(() => ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.Canonical)).Message));
        WithFile(Entries(true).Where(e => !e.Name.StartsWith("post_quant_conv.")), file => Assert.Throws<InvalidDataException>(() => ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.Canonical)));
        var aliases = DiffusersVaeEntries(); aliases.Add(new("encoder.mid_block.attentions.0.query.weight", [128, 128]));
        WithFile(aliases, file => Assert.Contains("collision", Assert.Throws<InvalidDataException>(() => ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.Diffusers)).Message));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoadBindsReaderAndCancellationAndPartialFailureReleaseEveryMaterializedTensor(bool vae)
    {
        WithFile(Entries(vae), file =>
        {
            var seen = new List<Tensor>();
            using var cts = new CancellationTokenSource();
            if (vae)
            {
                var plan = ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.Canonical);
                WithFile(Entries(true), other => Assert.Throws<ArgumentException>(() => ClassicalVaeCheckpointLoader.Load(other, plan)));
                Assert.Throws<OperationCanceledException>(() => ClassicalVaeCheckpointLoader.Load(file, plan, new(true), _ => Assert.Fail("Precancelled load materialized")));
                Assert.Throws<OperationCanceledException>(() => ClassicalVaeCheckpointLoader.Load(file, plan, cts.Token, d => { seen.Add(d.Last().Value); if (d.Count == 3) cts.Cancel(); }));
                Assert.All(seen, t => Assert.True(t.IsInvalid)); seen.Clear();
                Assert.Throws<InvalidOperationException>(() => ClassicalVaeCheckpointLoader.Load(file, plan, default, d => { seen.Add(d.Last().Value); if (d.Count == 3) throw new InvalidOperationException("Injected failure"); }));
            }
            else
            {
                var plan = UnetCheckpointLoader.Inspect(file, Unet, UnetCheckpointLayout.Canonical);
                WithFile(Entries(false), other => Assert.Throws<ArgumentException>(() => UnetCheckpointLoader.Load(other, plan)));
                Assert.Throws<OperationCanceledException>(() => UnetCheckpointLoader.Load(file, plan, new(true), _ => Assert.Fail("Precancelled load materialized")));
                Assert.Throws<OperationCanceledException>(() => UnetCheckpointLoader.Load(file, plan, cts.Token, d => { seen.Add(d.Last().Value); if (d.Count == 3) cts.Cancel(); }));
                Assert.All(seen, t => Assert.True(t.IsInvalid)); seen.Clear();
                Assert.Throws<InvalidOperationException>(() => UnetCheckpointLoader.Load(file, plan, default, d => { seen.Add(d.Last().Value); if (d.Count == 3) throw new InvalidOperationException("Injected failure"); }));
            }
            Assert.Equal(3, seen.Count); Assert.All(seen, t => Assert.True(t.IsInvalid));
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OwnedBanksSnapshotDetachRetainAndFinallyDisposeUniqueWrappers(bool vae)
    {
        NativeRuntimeBootstrap.Initialize();
        Dictionary<string, Tensor> values;
        UnetWeightSet? unet = null; ClassicalVaeWeightSet? autoencoder = null;
        using (var scope = NewDisposeScope())
        {
            // Shared shape-identical wrappers deliberately verify deduplicated last-owner disposal.
            var byShape = new Dictionary<string, Tensor>();
            values = Schema(vae).ToDictionary(p => p.Key, p =>
            {
                string shape = string.Join(',', p.Value);
                if (!byShape.TryGetValue(shape, out var tensor)) byShape.Add(shape, tensor = zeros(p.Value.ToArray(), dtype: ScalarType.Float32));
                return tensor;
            });
            values.Add("unknown", zeros(1, dtype: ScalarType.Float32));
            if (vae) Assert.Throws<InvalidDataException>(() => ClassicalVaeWeightSet.FromOwnedTensors(Vae, values));
            else Assert.Throws<InvalidDataException>(() => UnetWeightSet.FromOwnedTensors(Unet, values));
            Assert.All(values.Values, t => Assert.False(t.IsInvalid)); values.Remove("unknown");
            if (vae) autoencoder = ClassicalVaeWeightSet.FromOwnedTensors(Vae, values);
            else unet = UnetWeightSet.FromOwnedTensors(Unet, values);
        }
        var transferred = values.Values.ToArray();
        string firstKey = values.First().Key;
        values.Clear(); // Mutating the caller's dictionary cannot change the retained bank.
        Assert.All(transferred, t => Assert.False(t.IsInvalid));
        if (vae)
        {
            using var retained = autoencoder!.Retain(); autoencoder.Dispose();
            Assert.Throws<ObjectDisposedException>(() => autoencoder.Retain());
            Assert.False(retained.GetTensor(firstKey).IsInvalid);
        }
        else
        {
            using var retained = unet!.Retain(); unet.Dispose();
            Assert.Throws<ObjectDisposedException>(() => unet.Retain());
            Assert.False(retained.GetTensor(firstKey).IsInvalid);
        }
        Assert.All(transferred, t => Assert.True(t.IsInvalid));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompleteLoadOutlivesReaderAndAmbientScopeWithMixedStorage(bool vae)
    {
        NativeRuntimeBootstrap.Initialize();
        string prefix = vae ? "first_stage_model." : "model.diffusion_model.";
        string[] dtypes = ["F32", "F16", "BF16"];
        var entries = Entries(vae).Select((e, i) => e with { Name = prefix + e.Name, DType = dtypes[i % 3] });
        string path = Fixture(entries.Append(new("unrelated.weight", [1], "I64")));
        UnetWeightSet? unet = null; ClassicalVaeWeightSet? autoencoder = null;
        try
        {
            using (var scope = NewDisposeScope())
            using (var file = new SafeTensorFile(path))
            {
                if (vae) autoencoder = ClassicalVaeCheckpointLoader.Load(file,
                    ClassicalVaeCheckpointLoader.Inspect(file, Vae, ClassicalVaeCheckpointLayout.FirstStageModel));
                else unet = UnetCheckpointLoader.Load(file,
                    UnetCheckpointLoader.Inspect(file, Unet, UnetCheckpointLayout.ModelDiffusionModel));
            }
            using var readScope = NewDisposeScope();
            foreach (string key in Schema(vae).Keys)
            {
                var value = vae ? autoencoder!.GetTensor(key) : unet!.GetTensor(key);
                Assert.False(value.IsInvalid);
                Assert.Equal(ScalarType.Float32, value.dtype);
                using var flat = value.flatten();
                using var sample = flat[1];
                Assert.Equal(-1.5f, sample.item<float>());
            }
        }
        finally { unet?.Dispose(); autoencoder?.Dispose(); File.Delete(path); }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void FactoryRejectsStorageDtypesAndTrainableTensorsWithoutTakingOwnership(bool vae, bool gradients)
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        var entry = Schema(vae).First();
        var value = zeros(entry.Value.ToArray(), dtype: gradients ? ScalarType.Float32 : ScalarType.Float16, requires_grad: gradients);
        var values = new Dictionary<string, Tensor> { [entry.Key] = value };
        var error = vae ? Assert.Throws<InvalidDataException>(() => ClassicalVaeWeightSet.FromOwnedTensors(Vae, values))
            : Assert.Throws<InvalidDataException>(() => UnetWeightSet.FromOwnedTensors(Unet, values));
        Assert.Contains(gradients ? "frozen" : "Float32", error.Message);
        Assert.False(value.IsInvalid);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MisalignedStorageNormalizesAtomicallyPreservingBitsAndDeduplicatedLifetime(bool vae)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var schema = Schema(vae);
        var values = schema.ToDictionary(p => p.Key, p => zeros(p.Value.ToArray(), dtype: ScalarType.Float32));
        var vectorShape = schema.First(p => p.Value.Count == 1).Value;
        string[] aliases = schema.Where(p => p.Value.SequenceEqual(vectorShape)).Take(2).Select(p => p.Key).ToArray();
        Assert.Equal(2, aliases.Length);
        // A one-float offset is deterministic regardless of managed GC placement.
        long length = schema[aliases[0]].Aggregate(1L, (a, b) => a * b);
        var allocation = arange(length + 1, dtype: ScalarType.Float32);
        Assert.True(CpuModelWeightBank.IsAligned(allocation));
        var shifted = allocation.narrow(0, 1, length).reshape(schema[aliases[0]].ToArray());
        Assert.False(CpuModelWeightBank.IsAligned(shifted));
        byte[] bits = shifted.bytes.ToArray();
        foreach (string alias in aliases) { values[alias].Dispose(); values[alias] = shifted; }
        var materialized = new List<Tensor>();
        Assert.Throws<InvalidOperationException>(() => CpuModelWeightBank.Create(schema, values, clone =>
        {
            materialized.Add(clone);
            throw new InvalidOperationException("Injected normalization failure");
        }));
        Assert.Single(materialized);
        Assert.True(materialized[0].IsInvalid);
        Assert.All(values.Values, tensor => Assert.False(tensor.IsInvalid));
        Assert.Equal(bits, shifted.bytes.ToArray());
        int clones = 0;
        using var bank = CpuModelWeightBank.Create(schema, values, _ => clones++);
        Assert.Equal(1, clones);
        Assert.True(shifted.IsInvalid);
        var normalized = bank.GetTensor(aliases[0]);
        Assert.True(CpuModelWeightBank.IsAligned(normalized));
        Assert.Equal(bits, normalized.bytes.ToArray());
        Assert.All(aliases, alias => Assert.Same(normalized, bank.GetTensor(alias)));
        string untouched = schema.Keys.First(key => !aliases.Contains(key));
        Assert.Same(values[untouched], bank.GetTensor(untouched));
        using var retained = bank.Retain();
        bank.Dispose();
        Assert.False(normalized.IsInvalid);
        retained.Dispose();
        Assert.True(normalized.IsInvalid);
        Assert.True(values[untouched].IsInvalid);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ConcurrentRetainAndDisposeNeverReturnsInvalidRetainedWeights(bool vae)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var values = Schema(vae).ToDictionary(p => p.Key, p => zeros(p.Value.ToArray(), dtype: ScalarType.Float32));
        using var barrier = new Barrier(2);
        if (vae)
        {
            using var bank = ClassicalVaeWeightSet.FromOwnedTensors(Vae, values);
            var task = Task.Run(() => { barrier.SignalAndWait(); try { using var owner = bank.Retain(); Assert.False(owner.GetTensor(values.First().Key).IsInvalid); } catch (ObjectDisposedException) { } });
            barrier.SignalAndWait(); bank.Dispose(); await task;
        }
        else
        {
            using var bank = UnetWeightSet.FromOwnedTensors(Unet, values);
            var task = Task.Run(() => { barrier.SignalAndWait(); try { using var owner = bank.Retain(); Assert.False(owner.GetTensor(values.First().Key).IsInvalid); } catch (ObjectDisposedException) { } });
            barrier.SignalAndWait(); bank.Dispose(); await task;
        }
        Assert.All(values.Values, t => Assert.True(t.IsInvalid));
    }
}
