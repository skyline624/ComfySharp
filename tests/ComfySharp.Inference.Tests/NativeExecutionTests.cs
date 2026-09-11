using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

// Real native operations exercise the engine ownership boundary; these are not model-family fixtures.
public sealed class NativeExecutionTests
{
    [Fact]
    public async Task ViewInCompositeResultOutlivesParentWrapperAndDiesWithResult()
    {
        NativeRuntimeBootstrap.Initialize();
        var registry = BuiltInNodes.CreateRegistry();
        Tensor? source = null, view = null;
        registry.Register(new RuntimeNode("TestTensorSource", [], [new("TENSOR")], (context, _) =>
        {
            source = tensor(new float[] { 1, 2, 3, 4 }).DetachFromDisposeScope();
            return [context.Own(source)];
        }));
        registry.Register(new RuntimeNode("TestTensorView", [new("value", "TENSOR")], [new("LATENT")], (context, inputs) =>
        {
            view = inputs["value"].GetNative<Tensor>().reshape(2, 2).DetachFromDisposeScope();
            return [context.Map(new Dictionary<string, RuntimeValue>
            {
                ["samples"] = context.Own(view), ["label"] = context.Json(JsonValue.Create("retained view"))
            })];
        }));

        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"source":{"class_type":"TestTensorSource","inputs":{}},
             "view":{"class_type":"TestTensorView","inputs":{"value":["source",0]}}}
            """), ["view"]);

        Assert.Equal("success", result.Status);
        Assert.NotNull(source); Assert.NotNull(view);
        Assert.True(source.IsInvalid);
        Assert.False(view.IsInvalid);
        var samples = result.Outputs["view"][0][0].Properties["samples"].GetNative<Tensor>();
        Assert.Equal(new long[] { 2, 2 }, samples.shape);
        using (var sum = samples.sum()) Assert.Equal(10f, sum.item<float>());
        result.Dispose();
        Assert.True(view.IsInvalid);
    }

    [Fact]
    public async Task LazySwitchAndFanoutCarryNativeTensorToJsonOutputs()
    {
        NativeRuntimeBootstrap.Initialize();
        var registry = BuiltInNodes.CreateRegistry();
        Tensor? source = null;
        int creations = 0, unselectedCalls = 0;
        registry.Register(new RuntimeNode("TestTensorSource", [], [new("TENSOR")], (context, _) =>
        {
            creations++;
            source = tensor(new float[] { 1, 2, 3, 4 }).DetachFromDisposeScope();
            return [context.Own(source)];
        }));
        registry.Register(new RuntimeNode("TestUnselectedTensor", [], [new("TENSOR")], (_, _) =>
        {
            unselectedCalls++;
            throw new InvalidOperationException("Unselected lazy input executed.");
        }));
        registry.Register(new RuntimeNode("TestTensorSum", [new("value", "TENSOR")], [new("FLOAT")], (context, inputs) =>
        {
            using var sum = inputs["value"].GetNative<Tensor>().sum();
            return [context.Json(JsonValue.Create(sum.item<float>()))];
        }));

        var result = await new EngineService(registry).ExecuteAsync(Prompt("""
            {"source":{"class_type":"TestTensorSource","inputs":{}},
             "unused":{"class_type":"TestUnselectedTensor","inputs":{}},
             "switch":{"class_type":"ComfySwitchNode","inputs":{"switch":true,"on_true":["source",0],"on_false":["unused",0]}},
             "sum1":{"class_type":"TestTensorSum","inputs":{"value":["switch",0]}},
             "sum2":{"class_type":"TestTensorSum","inputs":{"value":["switch",0]}}}
            """), ["sum1", "sum2"]);

        Assert.Equal("success", result.Status);
        Assert.Equal(10f, result.Outputs["sum1"][0][0]!.GetValue<float>());
        Assert.Equal(10f, result.Outputs["sum2"][0][0]!.GetValue<float>());
        Assert.Equal(1, creations); Assert.Equal(0, unselectedCalls);
        Assert.NotNull(source); Assert.True(source.IsInvalid);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ExceptionOrCancellationAfterNativeAllocationReclaimsTensor(bool cancel)
    {
        NativeRuntimeBootstrap.Initialize();
        using var cancellation = new CancellationTokenSource();
        var registry = new NodeRegistry();
        Tensor? allocated = null;
        registry.Register(new RuntimeNode("TestAbortTensor", [], [new("TENSOR")], (context, _) =>
        {
            allocated = ones(new long[] { 8 }).DetachFromDisposeScope();
            var value = context.Own(allocated);
            if (cancel) { cancellation.Cancel(); return [value]; }
            throw new InvalidOperationException("Failure after allocation.");
        }));

        using var result = await new EngineService(registry).ExecuteValuesAsync(
            Prompt("""{"node":{"class_type":"TestAbortTensor","inputs":{}}}"""), ["node"], cancellationToken: cancellation.Token);

        Assert.Equal(cancel ? "cancelled" : "error", result.Status);
        Assert.NotNull(allocated); Assert.True(allocated.IsInvalid);
    }

    [Fact]
    public async Task JsonBoundaryRejectsNativeOutputAndReleasesIt()
    {
        NativeRuntimeBootstrap.Initialize();
        var registry = new NodeRegistry();
        Tensor? allocated = null;
        registry.Register(new RuntimeNode("TestTensorSource", [], [new("TENSOR")], (context, _) =>
        {
            allocated = ones(new long[] { 2 }).DetachFromDisposeScope();
            return [context.Own(allocated)];
        }));

        var result = await new EngineService(registry).ExecuteAsync(
            Prompt("""{"node":{"class_type":"TestTensorSource","inputs":{}}}"""), ["node"]);

        Assert.Equal("error", result.Status);
        Assert.NotEmpty(result.Diagnostics);
        Assert.Empty(result.Outputs);
        Assert.NotNull(allocated); Assert.True(allocated.IsInvalid);
    }

    private static JsonObject Prompt(string json) => JsonNode.Parse(json)!.AsObject();

    private sealed class RuntimeNode(string type, IReadOnlyList<InputSchema> inputs, IReadOnlyList<OutputSchema> outputs,
        Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, IReadOnlyList<RuntimeValue>> execute) : IRuntimeNode
    {
        public NodeSchema Schema { get; } = new(type, type, "test", inputs, outputs);
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> values, CancellationToken cancellationToken)
            => ValueTask.FromResult(new NodeExecutionOutput(execute(context, values)));
    }
}
