using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class ImageNodeTests
{
    [Fact]
    public async Task RealImageGraphRetainsIndependentNativeResultsBeyondInvocationScopes()
    {
        NativeRuntimeBootstrap.Initialize();
        var prompt = JsonNode.Parse("""
            {"empty":{"class_type":"EmptyImage","inputs":{"width":1,"height":1,"batch_size":2,"color":16711680}},
             "repeat":{"class_type":"RepeatImageBatch","inputs":{"image":["empty",0],"amount":3}},
             "extract":{"class_type":"ImageFromBatch","inputs":{"image":["repeat",0],"batch_index":-1,"length":10}},
             "invert":{"class_type":"ImageInvert","inputs":{"image":["extract",0]}},
             "preview":{"class_type":"PreviewAny","inputs":{"source":["invert",0]}}}
            """)!.AsObject();
        var engine = new EngineService(TensorNodes.CreateRegistry());
        using var result = await engine.ExecuteValuesAsync(prompt, ["empty", "repeat", "extract", "invert", "preview"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        var empty = result.Outputs["empty"][0][0].GetNative<Tensor>();
        var repeat = result.Outputs["repeat"][0][0].GetNative<Tensor>();
        var extract = result.Outputs["extract"][0][0].GetNative<Tensor>();
        var invert = result.Outputs["invert"][0][0].GetNative<Tensor>();
        Assert.Equal(new long[] { 2, 1, 1, 3 }, empty.shape);
        Assert.Equal(new long[] { 6, 1, 1, 3 }, repeat.shape);
        Assert.Equal(new long[] { 1, 1, 1, 3 }, extract.shape);
        Assert.Equal(new float[] { 0, 1, 1 }, invert.data<float>().ToArray());
        empty.fill_(.25);
        Assert.Equal(new float[] { 1, 0, 0 }, extract.data<float>().ToArray());
        Assert.Equal(new float[] { 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0 }, repeat.data<float>().ToArray());
        Assert.Equal("tensor([[[[0., 1., 1.]]]])", result.UiOutputs["preview"]["text"]![0]!.GetValue<string>());
    }

    [Theory]
    [InlineData("EmptyImage")]
    [InlineData("ImageInvert")]
    [InlineData("RepeatImageBatch")]
    [InlineData("ImageFromBatch")]
    public async Task CancellationPrecedesInputBorrow(string id)
    {
        Assert.True(TensorNodes.CreateRegistry().TryGet(id, out var node));
        using var context = new RuntimeNodeContext();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => node.ExecuteAsync(context,
            new Dictionary<string, RuntimeValue>(), new CancellationToken(true)).AsTask());
    }

    [Theory]
    [InlineData("ImageInvert")]
    [InlineData("RepeatImageBatch")]
    [InlineData("ImageFromBatch")]
    public async Task JsonValuesDoNotMasqueradeAsNativeImageBuffers(string id)
    {
        Assert.True(TensorNodes.CreateRegistry().TryGet(id, out var node));
        using var context = new RuntimeNodeContext();
        var inputs = new Dictionary<string, RuntimeValue> { ["image"] = context.Json(new JsonArray(1, 2, 3)),
            ["amount"] = context.Json(JsonValue.Create(1)), ["batch_index"] = context.Json(JsonValue.Create(0)), ["length"] = context.Json(JsonValue.Create(1)) };
        await Assert.ThrowsAsync<ArgumentException>(() => node.ExecuteAsync(context, inputs, default).AsTask());
    }

    [Fact]
    public async Task FailedOwnershipTransferReleasesTheNewImage()
    {
        NativeRuntimeBootstrap.Initialize();
        Assert.True(TensorNodes.CreateRegistry().TryGet("EmptyImage", out var node));
        using var inputsContext = new RuntimeNodeContext();
        var inputs = new Dictionary<string, RuntimeValue>
        {
            ["width"] = inputsContext.Json(JsonValue.Create(1)), ["height"] = inputsContext.Json(JsonValue.Create(1)),
            ["batch_size"] = inputsContext.Json(JsonValue.Create(1)), ["color"] = inputsContext.Json(JsonValue.Create(0))
        };
        var disposed = new RuntimeNodeContext(); disposed.Dispose();
        long before = Tensor.TotalCount;
        await Assert.ThrowsAsync<ObjectDisposedException>(() => node.ExecuteAsync(disposed, inputs, default).AsTask());
        Assert.Equal(before, Tensor.TotalCount);
    }
}
