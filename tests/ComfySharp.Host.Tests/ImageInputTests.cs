using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Host;
using ComfySharp.Media;
using ComfySharp.Nodes.Tensor;
using ComfySharp.Storage;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using static TorchSharp.torch;

namespace ComfySharp.Host.Tests;

public sealed class ImageInputTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "comfysharp-input-" + Guid.NewGuid().ToString("N"));
    private static byte[] Fixture(string name)
    {
        using var source = typeof(ImageInputTests).Assembly.GetManifestResourceStream("ComfySharp.Host.Tests.Fixtures.input-" + name)!;
        using var buffer = new MemoryStream(); source.CopyTo(buffer); return buffer.ToArray();
    }

    [Fact]
    public void Native_rgb_and_alpha_decode_preserves_unassociated_colors_and_frame_shape()
    {
        using var rgb = new MemoryStream(Fixture("rgb.png")); var opaque = NativeImageDecoder.Decode(rgb);
        Assert.Equal((3, 2, 1), (opaque.Width, opaque.Height, opaque.Frames)); Assert.Null(opaque.Alpha);
        var expected = new byte[] { 255,0,0,0,255,0,0,0,255,51,102,153,128,64,32,255,255,255 };
        for (int i = 0; i < expected.Length; i++) Assert.InRange(Math.Abs(opaque.Rgb[i] - expected[i] / 255f), 0, 1e-7);
        using var rgba = new MemoryStream(Fixture("alpha.png")); var transparent = NativeImageDecoder.Decode(rgba);
        Assert.Equal((3, 1, 1), (transparent.Width, transparent.Height, transparent.Frames));
        Assert.InRange(Math.Abs(transparent.Rgb[0] - 240 / 255f), 0, 1e-7);
        Assert.NotNull(transparent.Alpha); Assert.Equal(0, transparent.Alpha[0]); Assert.Equal(1, transparent.Alpha[2]);
        Assert.InRange(Math.Abs(transparent.Alpha[1] - 128 / 255f), 0, 1e-7);
    }

    [Fact]
    public void Native_animation_orientation_jpeg_and_high_depth_are_not_silently_flattened()
    {
        using var animated = new MemoryStream(Fixture("animated.gif")); var frames = NativeImageDecoder.Decode(animated);
        Assert.Equal(2, frames.Frames); Assert.Equal(1, frames.Rgb[0]); Assert.Equal(1, frames.Rgb[3 * 2 * 3 + 1]);
        Assert.NotNull(frames.Alpha); Assert.All(frames.Alpha, value => Assert.Equal(1, value));
        using var normal = new MemoryStream(Fixture("rgb.png")); var original = NativeImageDecoder.Decode(normal);
        using var palette = new MemoryStream(Fixture("palette.png")); var indexed = NativeImageDecoder.Decode(palette);
        Assert.NotNull(indexed.Alpha); Assert.Equal(original.Rgb, indexed.Rgb); Assert.All(indexed.Alpha, value => Assert.Equal(1, value));
        using var rotated = new MemoryStream(Fixture("oriented.png")); var turned = NativeImageDecoder.Decode(rotated);
        Assert.Equal((2, 3), (turned.Width, turned.Height));
        for (int channel = 0; channel < 3; channel++) Assert.Equal(original.Rgb[9 + channel], turned.Rgb[channel]);
        using var jpeg = new MemoryStream(Fixture("rgb.jpg")); var lossy = NativeImageDecoder.Decode(jpeg);
        Assert.Equal((3, 2, 1), (lossy.Width, lossy.Height, lossy.Frames)); Assert.Null(lossy.Alpha);
        using var sixteen = new MemoryStream(Fixture("gray16.png")); var high = NativeImageDecoder.Decode(sixteen);
        Assert.Equal(1 / 65535f, high.Rgb[0]); Assert.Equal(257 / 65535f, high.Rgb[3]);
        Assert.Equal(32769 / 65535f, high.Rgb[6]); Assert.Equal(65534 / 65535f, high.Rgb[9]);
        Assert.Throws<InvalidDataException>(() => NativeImageDecoder.Decode(new MemoryStream(new byte[] { 1, 2, 3 })));
        Assert.ThrowsAny<OperationCanceledException>(() => NativeImageDecoder.Decode(new MemoryStream(), new CancellationToken(true)));
    }

    [Fact]
    public async Task LoadImage_outputs_rgb_inverse_alpha_or_64_square_empty_mask_and_reloads_changes()
    {
        var store = new ImageFileStore(root); var service = new ImageInputService(store);
        await service.UploadAsync("image.png", "", new MemoryStream(Fixture("alpha.png")), false, default);
        var registry = new NodeRegistry(); ImageInputNodes.Register(registry, service);
        using var engine = new EngineService(registry);
        var prompt = JsonNode.Parse("""{"image":{"class_type":"LoadImage","inputs":{"image":"image.png"}}}""")!.AsObject();
        using (var result = await engine.ExecuteValuesAsync(prompt, ["image"]))
        {
            Assert.Equal("success", result.Status);
            var mask = result.Outputs["image"][1][0].GetNative<Tensor>();
            Assert.Equal(new long[] { 1,1,3 }, mask.shape);
            Assert.Equal(new[] { 1f, 1f - 128 / 255f, 0f }, mask.data<float>().ToArray());
        }
        await service.UploadAsync("image.png", "", new MemoryStream(Fixture("rgb.png")), true, default);
        using var next = await engine.ExecuteValuesAsync(prompt, ["image"]);
        Assert.Equal("success", next.Status);
        Assert.Equal(new long[] { 1,2,3,3 }, next.Outputs["image"][0][0].GetNative<Tensor>().shape);
        var empty = next.Outputs["image"][1][0].GetNative<Tensor>(); Assert.Equal(new long[] { 1,64,64 }, empty.shape);
        using var nonzero = empty.count_nonzero(); Assert.Equal(0, nonzero.item<long>());
    }

    [Theory]
    [InlineData("rgba16-filters.png")] [InlineData("rgba16-adam7.png")]
    public void Every_16_bit_color_and_alpha_sample_survives_filters_and_interlacing(string name)
    {
        using var source = new MemoryStream(Fixture(name)); var decoded = NativeImageDecoder.Decode(source);
        Assert.Equal((9,10,1),(decoded.Width,decoded.Height,decoded.Frames)); Assert.NotNull(decoded.Alpha);
        for(int y=0;y<10;y++) for(int x=0;x<9;x++) for(int c=0;c<4;c++)
        {
            float expected=(c==3 && x==0 && y==0 ? 0 : (x*4099+y*8191+c*12345)%65536)/65535f;
            Assert.Equal(expected,c==3?decoded.Alpha[y*9+x]:decoded.Rgb[(y*9+x)*3+c]);
        }
        byte[] corrupt=Fixture(name); corrupt[45]^=1;
        Assert.ThrowsAny<Exception>(()=>NativeImageDecoder.Decode(new MemoryStream(corrupt)));
    }

    [Fact]
    public async Task Cancelled_input_replacement_preserves_original_and_cleans_temporary_upload()
    {
        var store=new ImageFileStore(root); store.PrepareDirectory("input", "");
        var file=new ImageFileDescriptor("photo.png","","input"); await store.WriteAtomicAsync(file,Fixture("rgb.png"),default);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(()=>store.WriteAtomicAsync(file,Fixture("alpha.png"),new CancellationToken(true)));
        Assert.Equal(Fixture("rgb.png"),await File.ReadAllBytesAsync(Path.Combine(root,"input","photo.png")));
        Assert.Equal(new[]{"photo.png"},store.PrepareDirectory("input",""));
    }

    [Theory]
    [InlineData("../escape.png")] [InlineData("C:/escape.png")] [InlineData("/escape.png")] [InlineData("a/../escape.png")]
    public void Input_paths_cannot_escape(string path) => Assert.Throws<ArgumentException>(() => ImageInputService.Descriptor(path));

    [Fact]
    public async Task Upload_aliases_deduplicate_and_rename_collisions_without_corrupting_originals()
    {
        var store = new ImageFileStore(root);
        await using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder => builder.ConfigureServices(s => s.AddSingleton(store)));
        using var client = factory.CreateClient();
        async Task<JsonObject> Upload(string prefix, byte[] bytes)
        {
            using var body = new MultipartFormDataContent(); body.Add(new ByteArrayContent(bytes), "image", "photo.png");
            using var response = await client.PostAsync(prefix + "/upload/image", body); response.EnsureSuccessStatusCode();
            return (await response.Content.ReadFromJsonAsync<JsonObject>())!;
        }
        var first = await Upload("", Fixture("rgb.png")); var duplicate = await Upload("/api", Fixture("rgb.png"));
        Assert.Equal("photo.png", first["name"]!.GetValue<string>()); Assert.True(JsonNode.DeepEquals(first, duplicate));
        var collision = await Upload("", Fixture("alpha.png")); Assert.Equal("photo (1).png", collision["name"]!.GetValue<string>());
        Assert.Equal(Fixture("rgb.png"), await File.ReadAllBytesAsync(Path.Combine(root,"input","photo.png")));
        var service = factory.Services.GetRequiredService<ImageInputService>();
        var read = await service.ReadAsync("photo.png [input]", default);
        Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(Fixture("rgb.png"))), read.Sha256);
        Assert.Equal(new[] { "photo (1).png", "photo.png" }, service.Names());
        await Assert.ThrowsAsync<InvalidDataException>(() => service.UploadAsync("bad.png", "", new MemoryStream(new byte[] { 1,2 }), false, default));
        Assert.False(File.Exists(Path.Combine(root,"input","bad.png")));
    }

    public void Dispose()
    {
        string path = Path.GetFullPath(root);
        Assert.StartsWith(Path.GetFullPath(Path.GetTempPath()), path, StringComparison.OrdinalIgnoreCase);
        if (Directory.Exists(path)) Directory.Delete(path, true);
    }
}
