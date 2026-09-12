using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Host;
using ComfySharp.Inference;
using ComfySharp.Nodes.Tensor;
using ComfySharp.Storage;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using static TorchSharp.torch;

namespace ComfySharp.Host.Tests;

public sealed class SaveLoraHostTests
{
    private sealed class StateNode(string? source) : IRuntimeNode
    {
        public NodeSchema Schema { get; } = new("TestAdapterState","Test Adapter State","test",[],[new("LORA_MODEL")]);
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,IReadOnlyDictionary<string,RuntimeValue> inputs,CancellationToken cancellationToken)
        {
            var values=new Dictionary<string,RuntimeValue>(StringComparer.Ordinal);
            if(source is not null)
            {
                using var file=new SafeTensorFile(source);
                if(file.Tensors.Values.Sum(v=>v.End-v.Start)>512L*1024*1024)throw new InvalidDataException("Adapter test input exceeds its allowance.");
                foreach(string key in file.Tensors.Keys)values.Add(key,context.Own(file.ReadTensor(key,cancellationToken)));
            }
            else
            {
                NativeRuntimeBootstrap.Initialize();
                values.Add("diffusion_model.layer.diff",context.Own(ones(new long[]{2,3},dtype:ScalarType.Float16)));
                values.Add("diffusion_model.layer.diff_b",context.Own(zeros(new long[]{2},dtype:ScalarType.BFloat16)));
            }
            return ValueTask.FromResult(new NodeExecutionOutput([context.Map(values)]));
        }
    }
    [Fact]
    public async Task Host_exports_native_adapter_state_and_preserves_each_tensor()
    {
        string? source=Environment.GetEnvironmentVariable("COMFYSHARP_SAVE_LORA_INPUT");
        string? evidence=Environment.GetEnvironmentVariable("COMFYSHARP_SAVE_LORA_EVIDENCE");
        string root=evidence is null?Path.Combine(Path.GetTempPath(),"comfysharp-save-lora-host-"+Guid.NewGuid().ToString("N")):Path.GetFullPath(evidence);
        if(Directory.Exists(root)||File.Exists(root))throw new IOException("SaveLoRA qualification requires a new output directory.");
        var store=new ImageFileStore(root);var registry=new NodeRegistry();registry.Register(new StateNode(source));registry.Register(new SaveLoraNode(store));
        try
        {
            await using(var factory=new WebApplicationFactory<Program>().WithWebHostBuilder(builder=>
                builder.ConfigureServices(services=>services.AddSingleton(new EngineService(registry)))))
            {
                using var client=factory.CreateClient();
                using var response=await client.PostAsJsonAsync("/prompt",JsonNode.Parse("""
                    {"prompt":{"state":{"class_type":"TestAdapterState","inputs":{}},"save":{"class_type":"SaveLoRA","inputs":{"lora":["state",0],"prefix":"loras/adapter","steps":2}}}}
                    """));
                response.EnsureSuccessStatusCode();var accepted=await response.Content.ReadFromJsonAsync<JsonObject>();
                string id=accepted!["prompt_id"]!.GetValue<string>();using var timeout=new CancellationTokenSource(TimeSpan.FromSeconds(30));
                JsonObject? entry=null;
                while(entry is null)
                {
                    var history=await client.GetFromJsonAsync<JsonObject>("/history",timeout.Token);entry=history![id] as JsonObject;
                    if(entry is null)await Task.Delay(10,timeout.Token);
                }
                Assert.True(entry["status"]!["completed"]!.GetValue<bool>(),entry.ToJsonString());
                Assert.Empty(entry["outputs"]!.AsObject());
            }
            string output=Path.Combine(root,"output","loras","adapter_2_steps_00001_.safetensors");
            using var written=new SafeTensorFile(output);var hashes=new Dictionary<string,string>();
            if(source is not null)
            {
                using var original=new SafeTensorFile(source);Assert.Equal(original.Tensors.Keys.Order(),written.Tensors.Keys.Order());
                foreach(string name in original.Tensors.Keys)
                {
                    var expected=original.Tensors[name];var actual=written.Tensors[name];
                    Assert.Equal(expected.DType,actual.DType);Assert.Equal(expected.Shape,actual.Shape);
                    using var left=original.ReadTensor(name);using var right=written.ReadTensor(name);
                    string sha=Convert.ToHexStringLower(SHA256.HashData(left.bytes));Assert.Equal(sha,Convert.ToHexStringLower(SHA256.HashData(right.bytes)));hashes.Add(name,sha);
                }
            }
            else
            {
                Assert.Equal("F16",written.Tensors["diffusion_model.layer.diff"].DType);
                Assert.Equal("BF16",written.Tensors["diffusion_model.layer.diff_b"].DType);
                using var weight=written.ReadTensor("diffusion_model.layer.diff");using var difference=written.ReadTensor("diffusion_model.layer.diff_b");
                using var sum=weight.sum();using var zero=difference.sum();using var converted=zero.to_type(ScalarType.Float32);
                Assert.Equal((Half)6,sum.item<Half>());Assert.Equal(0,converted.item<float>());
            }
            if(evidence is not null)
                File.WriteAllText(Path.Combine(root,"result.json"),JsonSerializer.Serialize(new{success=true,sourceProvided=source is not null,tensorCount=written.Tensors.Count,
                    outputSha256=written.ComputeSha256(),parameterHashes=hashes,scope="SaveLoRA Host export only; training and model-family qualification remain separate."}));
        }
        finally
        {
            if(evidence is null)
            {
                string resolved=Path.GetFullPath(root);
                Assert.StartsWith(Path.GetFullPath(Path.GetTempPath()),resolved,StringComparison.OrdinalIgnoreCase);
                Assert.StartsWith("comfysharp-save-lora-host-",Path.GetFileName(resolved));
                Directory.Delete(resolved,true);
            }
        }
    }
}
