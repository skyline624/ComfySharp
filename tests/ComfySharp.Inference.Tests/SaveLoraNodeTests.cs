using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SaveLoraNodeTests
{
    private sealed class Store(params string[] entries) : IStreamingFileStore
    {
        public readonly Dictionary<string,byte[]> Files=entries.ToDictionary(n=>n,_=>Array.Empty<byte>());
        public IReadOnlyList<string> PrepareDirectory(string type,string subfolder,CancellationToken cancellationToken=default)
        { Assert.Equal("output",type);return Files.Keys.ToArray(); }
        public ValueTask WriteAsync(ImageFileDescriptor file,ReadOnlyMemory<byte> bytes,CancellationToken cancellationToken=default)=>throw new NotSupportedException();
        public ValueTask WriteAtomicAsync(ImageFileDescriptor file,Action<Stream,CancellationToken> write,CancellationToken cancellationToken=default)
        {using var output=new MemoryStream();write(output,cancellationToken);Files[file.Filename]=output.ToArray();return ValueTask.CompletedTask;}
    }
    [Theory]
    [InlineData(null,"adapter_00001_.safetensors")]
    [InlineData(0,"adapter_0_steps_00001_.safetensors")]
    [InlineData(7,"adapter_7_steps_00001_.safetensors")]
    [InlineData(-2,"adapter_-2_steps_00001_.safetensors")]
    public async Task Source_filename_steps_and_empty_output_contract_are_preserved(int? steps,string expected)
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var input=new RuntimeNodeContext())using(var context=new RuntimeNodeContext())
        {
            var store=new Store();var node=new SaveLoraNode(store);
            var map=input.Map(new Dictionary<string,RuntimeValue>{{"diffusion_model.layer.diff",input.Own(ones(2))}});
            var values=new Dictionary<string,RuntimeValue>{{"lora",map},{"prefix",input.Json(JsonValue.Create("adapter"))}};
            if(steps.HasValue)values["steps"]=input.Json(JsonValue.Create(steps.Value));
            var result=await node.ExecuteAsync(context,values,default);
            Assert.Empty(result.Result);Assert.Null(result.Ui);Assert.Equal(expected,Assert.Single(store.Files).Key);
            using var sum=map.Properties.Single().Value.GetNative<Tensor>().sum();Assert.Equal(2,sum.item<float>());
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
    [Fact]
    public async Task Counter_scan_includes_steps_segment_exactly_as_the_source_helper()
    {
        using var input=new RuntimeNodeContext();using var context=new RuntimeNodeContext();
        var store=new Store("adapter_20_steps_00001_.safetensors","other_10000_.png");
        var result=await new SaveLoraNode(store).ExecuteAsync(context,new Dictionary<string,RuntimeValue>
        {{"lora",input.Map(new Dictionary<string,RuntimeValue>())},{"prefix",input.Json(JsonValue.Create("adapter"))},{"steps",input.Json(JsonValue.Create(30))}},default);
        Assert.Contains("adapter_30_steps_00021_.safetensors",store.Files.Keys);Assert.Empty(result.Result);
    }
    [Fact]
    public async Task Non_tensor_maps_and_cancelled_jobs_cannot_produce_an_asset()
    {
        using var input=new RuntimeNodeContext();using var context=new RuntimeNodeContext();var store=new Store();var node=new SaveLoraNode(store);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async()=>await node.ExecuteAsync(context,new Dictionary<string,RuntimeValue>(),new(true)));
        await Assert.ThrowsAsync<ArgumentException>(async()=>await node.ExecuteAsync(context,new Dictionary<string,RuntimeValue>
        {{"lora",input.Json(new JsonObject())},{"prefix",input.Json(JsonValue.Create("adapter"))}},default));
        await Assert.ThrowsAsync<InvalidOperationException>(async()=>await node.ExecuteAsync(context,new Dictionary<string,RuntimeValue>
        {{"lora",input.Map(new Dictionary<string,RuntimeValue>{{"bad",input.Json(JsonValue.Create(1))}})},{"prefix",input.Json(JsonValue.Create("adapter"))}},default));
        Assert.Empty(store.Files);
    }
    [Fact]
    public void Schema_preserves_native_map_optional_steps_and_output_node_flags()
    {
        var registry=new NodeRegistry();registry.Register(new SaveLoraNode(new Store()));
        var row=registry.ToObjectInfo()["SaveLoRA"]!;
        Assert.Equal(new[]{"lora","prefix"},row["input_order"]!["required"]!.AsArray().Select(v=>v!.GetValue<string>()));
        Assert.Equal(new[]{"steps"},row["input_order"]!["optional"]!.AsArray().Select(v=>v!.GetValue<string>()));
        Assert.Equal("LORA_MODEL",row["input"]!["required"]!["lora"]![0]!.GetValue<string>());
        Assert.True(row["experimental"]!.GetValue<bool>());Assert.True(row["output_node"]!.GetValue<bool>());
        Assert.Empty(row["output"]!.AsArray());
    }
}
