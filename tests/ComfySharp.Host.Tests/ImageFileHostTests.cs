using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Host;
using ComfySharp.Storage;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using static ComfySharp.Testing.PngFixtureReader;

namespace ComfySharp.Host.Tests;

public sealed class ImageFileHostTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "comfysharp-image-http-" + Guid.NewGuid().ToString("N"));

    [Theory]
    [InlineData("", "SaveImage")]
    [InlineData("/api", "SaveImage")]
    [InlineData("", "PreviewImage")]
    [InlineData("/api", "PreviewImage")]
    public async Task Submitted_images_reach_events_history_disk_and_original_png_routes(string prefix, string nodeType)
    {
        await using var factory = Factory(); using var client = factory.CreateClient();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var hub = factory.Services.GetRequiredService<EventHub>(); var subscriber = hub.Subscribe("image-client");
        try
        {
            var prompt = JsonNode.Parse("""
                {"image":{"class_type":"EmptyImage","inputs":{"width":2,"height":3,"batch_size":2,"color":3368601}},
                 "file":{"class_type":"SaveImage","inputs":{"images":["image",0],"filename_prefix":"albums/render_%width%x%height%"}}}
                """)!.AsObject();
            prompt["file"]!["class_type"] = nodeType;
            if (nodeType == "PreviewImage") prompt["file"]!["inputs"]!.AsObject().Remove("filename_prefix");
            var workflow = JsonNode.Parse("""{"version":1,"extension":{"retain":["猫",null,4]}}""");
            var body = new JsonObject { ["prompt"] = prompt.DeepClone(), ["client_id"] = "image-client",
                ["extra_data"] = new JsonObject { ["extra_pnginfo"] = new JsonObject { ["workflow"] = workflow!.DeepClone() } } };
            using var submitted = await client.PostAsJsonAsync(prefix + "/prompt", body, timeout.Token);
            submitted.EnsureSuccessStatusCode();
            string id = (await submitted.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();
            JsonNode? output = null;
            while (true)
            {
                var message = JsonNode.Parse(await subscriber.Reader.ReadAsync(timeout.Token))!;
                string type = message["type"]!.GetValue<string>();
                Assert.NotEqual("execution_error", type); Assert.NotEqual("execution_interrupted", type);
                if (type == "executed") output = message["data"]!["output"]!.DeepClone();
                if (type == "execution_success") break;
            }
            Assert.NotNull(output);
            var history = (await client.GetFromJsonAsync<JsonObject>(prefix + "/history/" + id, timeout.Token))![id]!;
            Assert.True(history["status"]!["completed"]!.GetValue<bool>());
            Assert.True(JsonNode.DeepEquals(output, history["outputs"]!["file"]));
            var images = output["images"]!.AsArray(); Assert.Equal(2, images.Count);
            foreach (var descriptor in images)
            {
                var file = descriptor!.Deserialize<ImageFileDescriptor>()!;
                Assert.Equal(nodeType == "SaveImage" ? "output" : "temp", file.Type);
                Assert.Equal(nodeType == "SaveImage" ? "albums" : "", file.Subfolder);
                string url = prefix + "/view?filename=" + Uri.EscapeDataString(file.Filename) +
                    "&subfolder=" + Uri.EscapeDataString(file.Subfolder) + "&type=" + file.Type;
                using var response = await client.GetAsync(url, timeout.Token); response.EnsureSuccessStatusCode();
                Assert.Equal("image/png", response.Content.Headers.ContentType!.MediaType);
                Assert.True(response.Headers.CacheControl!.NoStore);
                byte[] bytes = await response.Content.ReadAsByteArrayAsync(timeout.Token);
                Assert.Equal(await File.ReadAllBytesAsync(Path.Combine(root, file.Type, file.Subfolder, file.Filename), timeout.Token), bytes);
                var decoded = Read(bytes); Assert.Equal(2, decoded.Width); Assert.Equal(3, decoded.Height);
                Assert.Equal(Enumerable.Repeat(new byte[] { 0x33, 0x66, 0x99 }, 6).SelectMany(p => p), decoded.Pixels);
                Assert.Equal(new[] { "prompt", "workflow" }, decoded.Text.Select(t => t.Keyword));
                Assert.True(JsonNode.DeepEquals(prompt, JsonNode.Parse(decoded.Text[0].Text)));
                Assert.True(JsonNode.DeepEquals(workflow, JsonNode.Parse(decoded.Text[1].Text)));
                using var head = new HttpRequestMessage(HttpMethod.Head, url);
                using var headResult = await client.SendAsync(head, timeout.Token); headResult.EnsureSuccessStatusCode();
                Assert.Equal(bytes.Length, headResult.Content.Headers.ContentLength);
                Assert.Empty(await headResult.Content.ReadAsByteArrayAsync(timeout.Token));
                using var range = new HttpRequestMessage(HttpMethod.Get, url); range.Headers.Range = new RangeHeaderValue(0, 7);
                using var rangeResult = await client.SendAsync(range, timeout.Token);
                Assert.Equal(HttpStatusCode.PartialContent, rangeResult.StatusCode);
                Assert.Equal(bytes[..8], await rangeResult.Content.ReadAsByteArrayAsync(timeout.Token));
            }
            Assert.Equal(2, Directory.GetFiles(root, "*.png", SearchOption.AllDirectories).Length);
            if (nodeType == "SaveImage") Assert.Equal("render_2x3_00001_.png", images[0]!["filename"]!.GetValue<string>());
        }
        finally { hub.Remove(subscriber.Id); }
    }

    [Theory]
    [InlineData("", 400)]
    [InlineData("?filename=absent.png", 404)]
    [InlineData("?filename=absent.png&subfolder=unknown", 404)]
    [InlineData("?filename=..%2Fescape.png", 400)]
    [InlineData("?filename=escape.png&subfolder=..", 400)]
    [InlineData("?filename=escape.png&type=input", 404)]
    [InlineData("?filename=image.png&channel=alpha", 400)]
    [InlineData("?filename=image.jpg", 400)]
    public async Task View_rejects_unsupported_requests_and_confines_paths(string query, int status)
    {
        await using var factory = Factory(); using var client = factory.CreateClient();
        foreach (string prefix in new[] { "", "/api" })
        {
            using var response = await client.GetAsync(prefix + "/view" + query);
            Assert.Equal((HttpStatusCode)status, response.StatusCode);
            if (status == 400) Assert.NotNull((await response.Content.ReadFromJsonAsync<JsonObject>())!["error"]);
        }
        Assert.Empty(Directory.GetFiles(root, "*", SearchOption.AllDirectories));
    }

    private WebApplicationFactory<Program> Factory()
    {
        var store = new ImageFileStore(root);
        return new WebApplicationFactory<Program>().WithWebHostBuilder(builder => builder.ConfigureServices(services =>
        {
            services.AddSingleton(store); services.AddSingleton<IImageFileStore>(store);
        }));
    }

    public void Dispose()
    {
        string resolved = Path.GetFullPath(root);
        Assert.StartsWith(Path.GetFullPath(Path.GetTempPath()), resolved, StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith("comfysharp-image-http-", Path.GetFileName(resolved), StringComparison.Ordinal);
        if (Directory.Exists(resolved)) Directory.Delete(resolved, true);
    }
}
