using Xunit;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdInpaintMaskTests
{
    [Fact]
    public void Invalid_inputs_and_disposed_masks_fail_without_leaks_or_model_calls()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            var latent = zeros(new long[] {1,4,2,3}); var mask = ones(new long[] {1,2,3});
            Assert.ThrowsAny<OperationCanceledException>(() => new SdInpaintMask(latent,latent,mask,new(true)));
            Assert.Throws<ArgumentException>(() => new SdInpaintMask(latent,zeros(new long[] {1,4,3,3}),mask));
            Assert.Throws<ArgumentException>(() => SdInpaintMask.PrepareMask(full_like(mask,float.NaN),latent.shape));
            Assert.Throws<ArgumentException>(() => SdInpaintMask.PrepareMask(mask,[0,4,2,3]));
            using var paint = new SdInpaintMask(latent,latent,mask);
            Tensor Never(Tensor x, Tensor sigma) => throw new Exception("unexpected model call");
            Assert.Throws<ArgumentException>(() => paint.Denoise(latent,tensor(new[]{1f,2f}),Never));
            paint.Dispose();
            Assert.Throws<ObjectDisposedException>(() => paint.Denoise(latent,tensor(new[]{1f}),Never));
            var pixels = zeros(new long[] {1,8,8,3});
            Assert.Throws<ArgumentOutOfRangeException>(() => SdInpaintImage.Prepare(pixels,mask,65));
            Assert.Throws<ArgumentException>(() => SdInpaintImage.Prepare(pixels,ones(new long[] {2,8,8})));
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Mask_preparation_blends_and_inpaint_pixels_match_frozen_source()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using var stream = GetType().Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.inpaint.reference.json")!;
        using var document = JsonDocument.Parse(stream);
        foreach (var item in document.RootElement.GetProperty("prepared").EnumerateArray())
        {
            using var scope = NewDisposeScope();
            using var actual = SdInpaintMask.PrepareMask(Read(item.GetProperty("mask")), item.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray());
            Compare(actual, item.GetProperty("output"));
        }
        foreach (var item in document.RootElement.GetProperty("calls").EnumerateArray())
        {
            using var scope = NewDisposeScope();
            var mask = Read(item.GetProperty("mask")); var latent = Read(item.GetProperty("latent"));
            var noise = Read(item.GetProperty("noise")); var current = Read(item.GetProperty("current"));
            using var paint = new SdInpaintMask(latent, noise, mask);
            var expectedInputs = item.GetProperty("modelInputs").EnumerateArray().ToArray(); int call = 0;
            foreach (var output in item.GetProperty("outputs").EnumerateArray())
            {
                using var actual = paint.Denoise(current, Read(output.GetProperty("sigma")), (x, sigma) =>
                {
                    Compare(x, expectedInputs[call++]); using var local = NewDisposeScope();
                    return (x * .25 + sigma.reshape(-1, 1, 1, 1) * .125).DetachFromDisposeScope();
                });
                Compare(actual, output.GetProperty("output"));
            }
            Compare(current, item.GetProperty("current")); Compare(latent, item.GetProperty("latent"));
        }
        foreach (var item in document.RootElement.GetProperty("images").EnumerateArray())
        {
            using var scope = NewDisposeScope();
            var pixels = Read(item.GetProperty("pixels")); var mask = Read(item.GetProperty("mask"));
            var prepared = SdInpaintImage.Prepare(pixels, mask, item.GetProperty("grow").GetInt32());
            using var p = prepared.Pixels; using var m = prepared.NoiseMask;
            Compare(p, item.GetProperty("preparedPixels")); Compare(m, item.GetProperty("noiseMask"));
            Compare(pixels, item.GetProperty("pixels")); Compare(mask, item.GetProperty("mask"));
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory]
    [InlineData("euler")] [InlineData("heun")] [InlineData("dpmpp_2m")]
    public void Real_model_masked_trajectory_preserves_black_regions_and_white_matches_unmasked(string method)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope();
            var config = new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
            using var euler = new SdEulerSampler(denoiser); using var heun = new SdHeunSampler(denoiser); using var dpm = new SdDpmpp2MSampler(denoiser);
            var latent = full(new long[] {1,4,4,5}, .125f); var noise = full_like(latent, .25f);
            var initial = full_like(latent, .625f); var sigma = tensor(new[] {2f, .5f, 0f}); var context = zeros(new long[] {1,3,16});
            Tensor Sample(SdInpaintMask? mask) => method switch {
                "euler" => euler.Sample(initial,sigma,context,null,inpaint:mask),
                "heun" => heun.Sample(initial,sigma,context,null,inpaint:mask),
                _ => dpm.Sample(initial,sigma,context,null,inpaint:mask) };
            using var plain = Sample(null);
            using var white = new SdInpaintMask(latent,noise,ones(new long[] {1,4,5}));
            using var whiteResult = Sample(white);
            Assert.Equal(plain.data<float>().ToArray(), whiteResult.data<float>().ToArray());
            var mask = zeros(new long[] {1,4,5}); mask.narrow(2, 2, 3).fill_(1);
            using var partial = new SdInpaintMask(latent,noise,mask);
            using var retained = partial.Retain(); partial.Dispose();
            using var result = Sample(retained);
            Assert.All(result.narrow(3,0,2).contiguous().data<float>().ToArray(), v => Assert.InRange(Math.Abs(v-.125),0,1e-6));
            Assert.Contains(result.narrow(3,2,3).contiguous().data<float>().ToArray(), v => Math.Abs(v-.125) > 1e-4);
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public async Task Set_mask_preserves_latent_metadata_and_retains_borrowed_resources()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var input = new RuntimeNodeContext())
        using (var output = new RuntimeNodeContext())
        using (var scope = NewDisposeScope())
        {
            var registry = new NodeRegistry(); Sd15Nodes.Register(registry, new CheckpointFiles(null));
            Assert.True(registry.TryGet("SetLatentNoiseMask", out var node));
            var samples = input.Own(ones(new long[] {1,4,2,3}).DetachFromDisposeScope());
            var mask = input.Own(arange(6, dtype: ScalarType.Float32).reshape(1,2,3).DetachFromDisposeScope());
            var latent = input.Map(new Dictionary<string,RuntimeValue> { ["samples"] = samples, ["custom"] = input.Json(JsonValue.Create("kept")) });
            var values = new Dictionary<string,RuntimeValue> { ["samples"] = latent, ["mask"] = mask };
            var result = await node.ExecuteAsync(output, values, default);
            Assert.False(latent.Properties.ContainsKey("noise_mask"));
            input.Dispose();
            Assert.Equal("kept", result.Result[0].Properties["custom"].ToJson()!.GetValue<string>());
            Assert.Equal(new long[] {1,1,2,3}, result.Result[0].Properties["noise_mask"].GetNative<Tensor>().shape);
            Assert.All(result.Result[0].Properties["samples"].GetNative<Tensor>().data<float>().ToArray(), v => Assert.Equal(1,v));
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    private static Tensor Read(JsonElement item) => tensor(item.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray())
        .reshape(item.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray());
    private static void Compare(Tensor actual, JsonElement expected)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()), actual.shape);
        var values = expected.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var data = actual.contiguous().data<float>().ToArray(); Assert.Equal(values.Length, data.Length);
        for (int i=0;i<data.Length;i++) Assert.InRange(Math.Abs((double)data[i]-values[i]),0,1e-6+1e-6*Math.Abs(values[i]));
    }

    [Fact]
    public void Soft_mask_blends_both_model_input_and_prediction_without_mutating_inputs()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            var latent = full(new long[] { 1, 4, 1, 3 }, 2f);
            var noise = full_like(latent, 3f); var current = full_like(latent, 10f);
            var mask = tensor(new[] { 0f, .25f, 1f }).reshape(1, 1, 3);
            using var paint = new SdInpaintMask(latent, noise, mask);
            // Construction snapshots the input storages, including noncontiguous views.
            latent.fill_(42); noise.fill_(42); mask.fill_(42);
            using var actual = paint.Denoise(current, tensor(new[] { 2f }), (x, sigma) =>
            {
                Assert.Equal(Enumerable.Repeat(new[] { 8f, 8.5f, 10f }, 4).SelectMany(x => x), x.data<float>().ToArray());
                return x * 2;
            });
            Assert.Equal(Enumerable.Repeat(new[] { 2f, 5.75f, 20f }, 4).SelectMany(x => x), actual.data<float>().ToArray());
            Assert.All(current.data<float>().ToArray(), v => Assert.Equal(10, v));
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Mask_resize_repeats_batches_cyclically_and_preserves_fractional_values()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        var mask = tensor(new[] { 0f, 1f, 1f, 0f }).reshape(2, 1, 2);
        using var actual = SdInpaintMask.PrepareMask(mask, [3, 4, 1, 3]);
        Assert.Equal(new long[] { 3, 4, 1, 3 }, actual.shape);
        var expected = Enumerable.Repeat(new[] { 0f, .5f, 1f }, 4).SelectMany(x => x)
            .Concat(Enumerable.Repeat(new[] { 1f, .5f, 0f }, 4).SelectMany(x => x))
            .Concat(Enumerable.Repeat(new[] { 0f, .5f, 1f }, 4).SelectMany(x => x));
        Assert.Equal(expected, actual.data<float>().ToArray());
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void Failed_or_cancelled_prediction_releases_blend_and_restores_grad_mode(bool cancel)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        using (var grad = set_grad_enabled(true))
        using (var cancellation = new CancellationTokenSource())
        {
            var latent = ones(new long[] { 1, 4, 2, 2 });
            using var paint = new SdInpaintMask(latent, latent, zeros(new long[] { 1, 2, 2 }));
            Tensor? borrowed = null;
            Tensor Model(Tensor x, Tensor sigma)
            {
                borrowed = x; Assert.False(is_grad_enabled());
                if (!cancel) throw new InvalidOperationException("prediction failed");
                cancellation.Cancel(); return x.clone();
            }
            if (cancel) Assert.ThrowsAny<OperationCanceledException>(() => paint.Denoise(latent, tensor(new[] { 1f }), Model, cancellation.Token));
            else Assert.Throws<InvalidOperationException>(() => paint.Denoise(latent, tensor(new[] { 1f }), Model));
            Assert.NotNull(borrowed); Assert.True(borrowed.IsInvalid); Assert.True(is_grad_enabled());
        }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
