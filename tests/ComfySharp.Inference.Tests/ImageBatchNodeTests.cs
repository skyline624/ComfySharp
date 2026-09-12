using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class ImageBatchNodeTests
{
    [Fact]
    public async Task RealGraphResizesSecondBatchAndRetainsIndependentResult()
    {
        NativeRuntimeBootstrap.Initialize();
        var prompt = JsonNode.Parse("""
            {"first":{"class_type":"EmptyImage","inputs":{"width":1,"height":1,"batch_size":2,"color":16711680}},
             "second":{"class_type":"EmptyImage","inputs":{"width":2,"height":2,"batch_size":1,"color":65280}},
             "batch":{"class_type":"ImageBatch","inputs":{"image1":["first",0],"image2":["second",0]}},
             "last":{"class_type":"ImageFromBatch","inputs":{"image":["batch",0],"batch_index":-1,"length":1}},
             "preview":{"class_type":"PreviewAny","inputs":{"source":["last",0]}}}
            """)!.AsObject();
        var engine = new EngineService(TensorNodes.CreateRegistry());
        using var result = await engine.ExecuteValuesAsync(prompt, ["first", "second", "batch", "preview"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        var batch = result.Outputs["batch"][0][0].GetNative<Tensor>();
        Assert.Equal(new long[] { 3, 1, 1, 3 }, batch.shape);
        float[] expected = [1, 0, 0, 1, 0, 0, 0, 1, 0];
        Assert.Equal(expected, batch.data<float>().ToArray());
        result.Outputs["first"][0][0].GetNative<Tensor>().fill_(99);
        result.Outputs["second"][0][0].GetNative<Tensor>().fill_(99);
        Assert.Equal(expected, batch.data<float>().ToArray());
        Assert.Equal("tensor([[[[0., 1., 0.]]]])", result.UiOutputs["preview"]["text"]![0]!.GetValue<string>());
    }

    [Theory]
    [InlineData("image1")]
    [InlineData("image2")]
    public async Task EachInputRequiresNativeImageStorage(string invalid)
    {
        NativeRuntimeBootstrap.Initialize();
        Assert.True(TensorNodes.CreateRegistry().TryGet("ImageBatch", out var node));
        using var owner = new RuntimeNodeContext();
        using var invocation = new RuntimeNodeContext();
        var inputs = new Dictionary<string, RuntimeValue>
        {
            ["image1"] = owner.Own(ImageOperations.EmptyImage(1, 1)),
            ["image2"] = owner.Own(ImageOperations.EmptyImage(1, 1))
        };
        inputs[invalid] = owner.Json(new JsonArray(1, 2, 3));
        await Assert.ThrowsAsync<ArgumentException>(() => node.ExecuteAsync(invocation, inputs, default).AsTask());
    }

    [Fact]
    public async Task CancellationPrecedesBothInputBorrows()
    {
        Assert.True(TensorNodes.CreateRegistry().TryGet("ImageBatch", out var node));
        using var invocation = new RuntimeNodeContext();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => node.ExecuteAsync(invocation,
            new Dictionary<string, RuntimeValue>(), new CancellationToken(true)).AsTask());
    }

    [Fact]
    public async Task RejectedOwnershipTransferDisposesResizedBatchWithoutChangingInputs()
    {
        NativeRuntimeBootstrap.Initialize();
        Assert.True(TensorNodes.CreateRegistry().TryGet("ImageBatch", out var node));
        using var owner = new RuntimeNodeContext();
        var inputs = new Dictionary<string, RuntimeValue>
        {
            ["image1"] = owner.Own(ImageOperations.EmptyImage(1, 1, color: 0xff0000)),
            ["image2"] = owner.Own(ImageOperations.EmptyImage(2, 2, color: 0x00ff00))
        };
        var disposed = new RuntimeNodeContext(); disposed.Dispose();
        long before = Tensor.TotalCount;
        await Assert.ThrowsAsync<ObjectDisposedException>(() => node.ExecuteAsync(disposed, inputs, default).AsTask());
        Assert.Equal(before, Tensor.TotalCount);
        Assert.Equal(new float[] { 1, 0, 0 }, inputs["image1"].GetNative<Tensor>().data<float>().ToArray());
        Assert.Equal(Enumerable.Repeat(new float[] { 0, 1, 0 }, 4).SelectMany(x => x),
            inputs["image2"].GetNative<Tensor>().data<float>().ToArray());
    }
}
