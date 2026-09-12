using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdKarrasScheduleTests
{
    [Fact]
    public void Partial_schedules_and_maximum_noise_match_frozen_source_cases()
    {
        using var file = typeof(SdKarrasScheduleTests).Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.sd15-denoise.reference.json")!;
        using var reference = JsonDocument.Parse(file);
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            using var result = SdKarrasSchedule.Create(item.GetProperty("steps").GetInt32(), item.GetProperty("denoise").GetDouble(), SdDiscreteSampling.Default);
            var expected = item.GetProperty("sigmas").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            var actual = result.data<float>().ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++) Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-6 + 1e-6 * Math.Abs(expected[i]));
            if (actual.Length > 0) Assert.Equal(item.GetProperty("maximumNoise").GetBoolean(), SdKarrasSchedule.UsesMaximumNoise(actual[0], SdDiscreteSampling.Default.SigmaMax));
        }
    }

    [Theory]
    [InlineData(-.1)] [InlineData(1.1)] [InlineData(double.NaN)] [InlineData(double.PositiveInfinity)]
    public void Invalid_strength_is_rejected(double strength) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => SdKarrasSchedule.Create(20, strength, SdDiscreteSampling.Default));

    [Fact]
    public void Expanded_schedule_budget_and_cancellation_are_explicit()
    {
        Assert.Throws<NotSupportedException>(() => SdKarrasSchedule.Create(100, .00001, SdDiscreteSampling.Default));
        Assert.ThrowsAny<OperationCanceledException>(() => SdKarrasSchedule.Create(20, .5, SdDiscreteSampling.Default, new CancellationToken(true)));
        Assert.True(SdKarrasSchedule.UsesMaximumNoise(9.99995, 10));
        Assert.False(SdKarrasSchedule.UsesMaximumNoise(9.999, 10));
    }

    [Theory]
    [InlineData("euler")] [InlineData("heun")]
    public async Task Zero_strength_keeps_raw_latent_bits_and_metadata_without_model_or_conditioning_access(string sampler)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var input = new RuntimeNodeContext())
        using (var output = new RuntimeNodeContext())
        using (var scope = NewDisposeScope())
        {
            var registry = new NodeRegistry(); Sd15Nodes.Register(registry, new CheckpointFiles(null));
            Assert.True(registry.TryGet("KSampler", out var node));
            var raw = input.Own(tensor(new[] { .125f, -7.125f, float.Epsilon }).reshape(1, 1, 1, 3));
            var values = new Dictionary<string, RuntimeValue>
            {
                ["sampler_name"] = input.Json(JsonValue.Create(sampler)), ["scheduler"] = input.Json(JsonValue.Create("karras")),
                ["steps"] = input.Json(JsonValue.Create(20)), ["cfg"] = input.Json(JsonValue.Create(7)), ["denoise"] = input.Json(JsonValue.Create(0)),
                ["latent_image"] = input.Map(new Dictionary<string, RuntimeValue> { ["samples"] = raw,
                    ["custom"] = input.Json(JsonValue.Create("retained")), ["downscale_ratio_spacial"] = input.Json(JsonValue.Create(8)) })
            };
            var result = await node.ExecuteAsync(output, values, default);
            Assert.Equal(raw.GetNative<Tensor>().data<float>().ToArray(), result.Result[0].Properties["samples"].GetNative<Tensor>().data<float>().ToArray());
            Assert.Equal("retained", result.Result[0].Properties["custom"].ToJson()!.GetValue<string>());
            Assert.False(result.Result[0].Properties.ContainsKey("downscale_ratio_spacial"));
        }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
