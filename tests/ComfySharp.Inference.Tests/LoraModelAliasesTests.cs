using System.Text.Json;
using ComfySharp.Tokenization;
using Xunit;

namespace ComfySharp.Inference.Tests;

public sealed class LoraModelAliasesTests
{
    private static JsonDocument Reference() => JsonDocument.Parse(typeof(LoraModelAliasesTests).Assembly
        .GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora-aliases.reference.json")!);

    private static void Compare(string name, IReadOnlyList<LoraAlias> actual, bool shapes = true, bool includeVectors = false)
    {
        using var reference = includeVectors ? JsonDocument.Parse(typeof(LoraModelAliasesTests).Assembly
            .GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora-all-aliases.reference.json")!) : Reference();
        // This historical source corpus explicitly covers rank>=2 weights only.
        // One-dimensional adapter aliases have a separate source corpus.
        if (!includeVectors) actual = actual.Where(a => a.Target.Shape.Count >= 2).ToArray();
        var row = reference.RootElement.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("name").GetString() == name);
        var expected = row.GetProperty("aliasesByTarget");
        var groups = actual.GroupBy(a => a.Target.Weight).ToDictionary(g => g.Key, g => g.ToArray());
        Assert.Equal(expected.EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal), groups.Keys.Order(StringComparer.Ordinal));
        Assert.Equal(actual.Count, actual.Select(a => a.Prefix).Distinct(StringComparer.Ordinal).Count());
        foreach (var property in expected.EnumerateObject())
        {
            Assert.Equal(property.Value.EnumerateArray().Select(v => v.GetString()), groups[property.Name].Select(a => a.Prefix));
            if (shapes)
                foreach (var alias in groups[property.Name])
                    Assert.Equal(row.GetProperty("shapes").GetProperty(property.Name).EnumerateArray().Select(v => v.GetInt64()), alias.Target.Shape);
        }
    }

    [Fact]
    public void All_weight_aliases_including_normalizations_match_frozen_source()
    {
        Compare("sd15-checkpoint", LoraModelAliases.ForUnet(SdUnetConfig.Sd15), includeVectors: true);
        Compare("clip-l-checkpoint", LoraModelAliases.ForClip(ClipTextConfig.Large, ClipProfile.Sd1L, false), includeVectors: true);
        Compare("clip_l-33-layer-metadata", LoraModelAliases.ForClip(new(4,8,33,1,ClipActivation.Gelu), ClipProfile.Sd1L), includeVectors: true);
        Compare("clip_g-33-layer-metadata", LoraModelAliases.ForClip(new(4,8,33,1,ClipActivation.Gelu), ClipProfile.SdXlG), includeVectors: true);
    }

    [Fact]
    public void Every_SD15_alias_and_per_target_priority_matches_frozen_source_on_the_real_checkpoint_header() =>
        Compare("sd15-checkpoint", LoraModelAliases.ForUnet(SdUnetConfig.Sd15));

    [Fact]
    public void SD2_keeps_the_plain_topology_aliases_with_its_own_weight_shapes()
    {
        var aliases = LoraModelAliases.ForUnet(SdUnetConfig.Sd2, "sd2");
        Compare("sd15-checkpoint", aliases, shapes: false);
        Assert.All(aliases, alias => Assert.Equal("sd2", alias.Target.Component));
        Assert.Equal(new long[] { 320, 320 }, aliases.First(a => a.Target.Weight == "input_blocks.1.1.proj_in.weight").Target.Shape);
        Assert.Equal(new long[] { 320, 1024 }, aliases.First(a => a.Target.Weight == "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight").Target.Shape);
    }

    [Theory]
    [InlineData(ClipProfile.Sd1L)]
    [InlineData(ClipProfile.SdXlL)]
    public void Standalone_CLIP_L_without_projection_matches_checkpoint_source(ClipProfile profile) =>
        Compare("clip-l-checkpoint", LoraModelAliases.ForClip(ClipTextConfig.Large, profile, includeProjection: false));

    [Theory]
    [InlineData(ClipProfile.Sd1L, "clip_l-33-layer-metadata")]
    [InlineData(ClipProfile.SdXlG, "clip_g-33-layer-metadata")]
    public void Generic_aliases_continue_after_source_legacy_layer_limit_and_projection_is_mapped(ClipProfile profile, string name)
    {
        var aliases = LoraModelAliases.ForClip(new(4, 8, 33, 1, ClipActivation.Gelu), profile);
        Compare(name, aliases);
        Assert.Single(aliases, a => a.Target.Weight == "text_model.encoder.layers.32.mlp.fc1.weight");
        Assert.DoesNotContain(aliases, a => a.Prefix.Contains("text_encoder_2.", StringComparison.Ordinal));
    }

    [Fact]
    public void Results_are_readonly_and_invalid_arguments_fail_before_native_work()
    {
        var aliases = LoraModelAliases.ForUnet(SdUnetConfig.Sd15);
        Assert.Throws<NotSupportedException>(() => ((IList<LoraAlias>)aliases).Clear());
        Assert.Throws<NotSupportedException>(() => ((IList<long>)aliases[0].Target.Shape)[0] = 1);
        Assert.Throws<ArgumentException>(() => LoraModelAliases.ForUnet(SdUnetConfig.Sd15, " "));
        Assert.Throws<ArgumentOutOfRangeException>(() => LoraModelAliases.ForClip(ClipTextConfig.Large, (ClipProfile)77));
        Assert.Throws<ArgumentNullException>(() => LoraModelAliases.ForUnet(null!));
    }
}
