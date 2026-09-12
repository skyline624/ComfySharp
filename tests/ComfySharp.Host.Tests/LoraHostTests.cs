using System.Net.Http.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;

namespace ComfySharp.Host.Tests;

public sealed class LoraHostTests
{
    [Fact]
    public async Task Host_publishes_both_nodes_using_only_the_shared_loras_category()
    {
        string root = Path.Combine(Path.GetTempPath(), "comfysharp-host-lora-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(Path.Combine(root, "loras", "nested"));
            Directory.CreateDirectory(Path.Combine(root, "checkpoints"));
            string path = Path.Combine(root, "loras", "nested", "adapter.safetensors");
            File.WriteAllBytes(path, [1, 2, 3]);
            File.WriteAllBytes(Path.Combine(root, "checkpoints", "base.safetensors"), [4, 5, 6]);
            await using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder => builder.ConfigureAppConfiguration((_, config) =>
                config.AddInMemoryCollection(new Dictionary<string, string?> { ["models-dir"] = root })));
            using var client = factory.CreateClient();
            foreach (string type in new[] { "LoraLoader", "LoraLoaderModelOnly" })
            {
                var info = (await client.GetFromJsonAsync<JsonObject>("/object_info/" + type))![type]!;
                Assert.Equal("nodes", info["python_module"]!.GetValue<string>());
                Assert.Equal(new[] { "nested/adapter.safetensors" }, info["input"]!["required"]!["lora_name"]![0]!.AsArray().Select(v => v!.GetValue<string>()));
                Assert.Equal(type == "LoraLoader" ? 2 : 1, info["output"]!.AsArray().Count);
            }
            Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(path));
            Assert.Equal(2, Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories).Count());
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}
