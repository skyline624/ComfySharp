using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Nodes.Tensor;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class TensorNodeTests
{
    [Theory]
    [InlineData("KarrasScheduler", 21)]
    [InlineData("ExponentialScheduler", 21)]
    [InlineData("PolyexponentialScheduler", 21)]
    [InlineData("LaplaceScheduler", 20)]
    [InlineData("VPScheduler", 21)]
    public async Task SchedulerDefaultsProduceOneNativeSlotWithoutUi(string id, int count)
    {
        var registry = TensorNodes.CreateRegistry();
        var prompt = new JsonObject { ["schedule"] = Node(registry, id) };
        using var result = await new EngineService(registry).ExecuteValuesAsync(prompt, ["schedule"]);
        Assert.Equal("success", result.Status);
        var value = Assert.Single(Assert.Single(result.Outputs["schedule"]));
        var tensor = value.GetNative<Tensor>();
        Assert.Equal(new long[] { count }, tensor.shape); Assert.Equal(ScalarType.Float32, tensor.dtype);
        Assert.Equal(TorchSharp.DeviceType.CPU, tensor.device.type); Assert.Empty(result.UiOutputs);
        result.Dispose(); Assert.True(tensor.IsInvalid);
    }

    [Fact]
    public async Task RealSigmaFanoutAndPreviewTraverseEngineWithoutSerializingTensors()
    {
        var registry = TensorNodes.CreateRegistry();
        var prompt = Prompt("""
            {"manual":{"class_type":"ManualSigmas","inputs":{"sigmas":"9, 6, 3, 0"}},
             "split":{"class_type":"SplitSigmas","inputs":{"sigmas":["manual",0],"step":2}},
             "set":{"class_type":"SetFirstSigma","inputs":{"sigmas":["split",0],"sigma":100}},
             "high":{"class_type":"PreviewAny","inputs":{"source":["split",0]}},
             "low":{"class_type":"PreviewAny","inputs":{"source":["split",1]}},
             "changed":{"class_type":"PreviewAny","inputs":{"source":["set",0]}}}
            """);
        var events = new List<EngineEvent>();
        var result = await new EngineService(registry).ExecuteUiAsync(prompt, ["high", "low", "changed"], onEvent: e => { events.Add(e); return ValueTask.CompletedTask; });
        Assert.Equal("success", result.Status);
        Assert.Equal("tensor([9., 6., 3.])", result.Outputs["high"]["text"]![0]!.GetValue<string>());
        Assert.Equal("tensor([3., 0.])", result.Outputs["low"]["text"]![0]!.GetValue<string>());
        Assert.Equal("tensor([100.,   6.,   3.])", result.Outputs["changed"]["text"]![0]!.GetValue<string>());
        Assert.Equal(3, result.Outputs.Count);
        Assert.Equal(new[] { "high", "low", "changed" }, events.Where(e => e.Type == "executed").Select(e => e.NodeId));
    }

    [Theory]
    [InlineData("SplitSigmasDenoise")]
    [InlineData("FlipSigmas")]
    [InlineData("ExtendIntermediateSigmas")]
    public async Task RemainingOperationsAreExecutableNativeNodes(string id)
    {
        var registry = TensorNodes.CreateRegistry();
        var operation = Node(registry, id); operation["inputs"]!["sigmas"] = new JsonArray("manual", 0);
        var prompt = new JsonObject { ["manual"] = new JsonObject { ["class_type"] = "ManualSigmas", ["inputs"] = new JsonObject { ["sigmas"] = "9, 3, 0" } }, ["operation"] = operation };
        using var result = await new EngineService(registry).ExecuteValuesAsync(prompt, ["operation"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.UiOutputs);
        Assert.All(result.Outputs["operation"], slot => Assert.Equal(RuntimeValueKind.Native, Assert.Single(slot).Kind));
    }

    [Fact]
    public async Task EmptyFlipRetainsSameNativeLeaseAndInvalidSigmasFails()
    {
        var registry = TensorNodes.CreateRegistry();
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"manual":{"class_type":"ManualSigmas","inputs":{"sigmas":""}},
             "flip":{"class_type":"FlipSigmas","inputs":{"sigmas":["manual",0]}}}
            """), ["manual", "flip"]);
        Assert.Equal("success", result.Status);
        Assert.Same(result.Outputs["manual"][0][0].GetNative<Tensor>(), result.Outputs["flip"][0][0].GetNative<Tensor>());
        var failed = await new EngineService(registry).ExecuteUiAsync(Prompt("""
            {"flip":{"class_type":"FlipSigmas","inputs":{"sigmas":3}},
             "preview":{"class_type":"PreviewAny","inputs":{"source":["flip",0]}}}
            """), ["preview"]);
        Assert.Equal("error", failed.Status); Assert.NotEmpty(failed.Diagnostics);
    }

    [Theory]
    [InlineData("null", "None")]
    [InlineData("true", "True")]
    [InlineData("false", "False")]
    [InlineData("1.0", "1.0")]
    [InlineData("0.00001", "1e-05")]
    [InlineData("1e16", "1e+16")]
    [InlineData("\"été 雪\"", "été 雪")]
    [InlineData("{\"é\":[1,true,null]}", "{\n    \"é\": [\n        1,\n        true,\n        null\n    ]\n}")]
    [InlineData("{\"emoji\":\"🌍\",\"separator\":\"\\u2028\"}", "{\n    \"emoji\": \"🌍\",\n    \"separator\": \"\u2028\"\n}")]
    public async Task PreviewMatchesPythonScalarsAndUnicodePrettyJson(string source, string expected)
    {
        var prompt = new JsonObject { ["preview"] = new JsonObject { ["class_type"] = "PreviewAny", ["inputs"] = new JsonObject { ["source"] = JsonNode.Parse(source) } } };
        using var result = await new EngineService(TensorNodes.CreateRegistry()).ExecuteValuesAsync(prompt, ["preview"]);
        Assert.Equal("success", result.Status);
        Assert.Equal(expected, result.Outputs["preview"][0][0].ToJson()!.GetValue<string>());
        Assert.Equal(expected, result.UiOutputs["preview"]["text"]![0]!.GetValue<string>());
    }

    [Fact]
    public async Task NativeRuntimeContainersUsePythonFallbackRepresentation()
    {
        NativeRuntimeBootstrap.Initialize();
        var registry = TensorNodes.CreateRegistry(); registry.Register(new CompositeNode());
        using var result = await new EngineService(registry).ExecuteValuesAsync(Prompt("""
            {"composite":{"class_type":"TestComposite","inputs":{}},"preview":{"class_type":"PreviewAny","inputs":{"source":["composite",0]}}}
            """), ["preview"]);
        Assert.Equal("success", result.Status);
        Assert.Equal("{'samples': tensor([1., 0.]), 'label': 'é'}", result.UiOutputs["preview"]["text"]![0]!.GetValue<string>());
    }

    [Fact]
    public void RegisteredSchemasPreserveOrderedSlotsOptionsAndOutputClassification()
    {
        var registry = TensorNodes.CreateRegistry(); var info = registry.ToObjectInfo();
        Assert.Equal(25, registry.Nodes.Count());
        Assert.Equal(new[] { "steps", "sigma_max", "sigma_min", "rho" }, info["KarrasScheduler"]!["input"]!["required"]!.AsObject().Select(p => p.Key));
        var rho = info["KarrasScheduler"]!["input"]!["required"]!["rho"]![1]!;
        Assert.Equal(7d, rho["default"]!.GetValue<double>()); Assert.Equal(0d, rho["min"]!.GetValue<double>());
        Assert.False(rho["round"]!.GetValue<bool>()); Assert.True(rho["advanced"]!.GetValue<bool>());
        Assert.Equal(new[] { "high_sigmas", "low_sigmas" }, info["SplitSigmas"]!["output_name"]!.AsArray().Select(v => v!.GetValue<string>()));
        Assert.True(info["ManualSigmas"]!["experimental"]!.GetValue<bool>());
        Assert.True(info["PreviewAny"]!["output_node"]!.GetValue<bool>());
        Assert.All(registry.Nodes.Where(n => n.Schema.ClassType.EndsWith("Scheduler", StringComparison.Ordinal) || n.Schema.Outputs.Any(o => o.Type == "SIGMAS")), n => Assert.False(n.Schema.OutputNode));
    }

    [Fact]
    public void DenseFormatterPreservesLinebreaksDtypeSignedZeroAndLimits()
    {
        NativeRuntimeBootstrap.Initialize();
        using var matrix = tensor(new float[] { 1, 2, 3, 4 }).reshape(2, 2);
        Assert.Equal("tensor([[1., 2.],\n        [3., 4.]])", TensorPreviewFormatter.Format(matrix));
        using var zero = tensor(new double[] { -0.0, double.NaN, double.PositiveInfinity });
        Assert.Equal("tensor([-0., nan, inf], dtype=torch.float64)", TensorPreviewFormatter.Format(zero));
        using var integer = tensor(new long[] { long.MaxValue, long.MinValue });
        Assert.Equal("tensor([ 9223372036854775807, -9223372036854775808])", TensorPreviewFormatter.Format(integer));
        using var values = arange(1001, dtype: ScalarType.Int64);
        Assert.Equal("tensor([   0,    1,    2,    3,    4,    5,  ...,  995,  996,  997,  998,  999,\n        1000])", TensorPreviewFormatter.Format(values));
    }

    [Fact]
    public void FormatterPreservesLeafGradientAndReportsUnportedGraphFormatting()
    {
        NativeRuntimeBootstrap.Initialize();
        using var leaf = tensor(new float[] { 1 }, requires_grad: true);
        Assert.Equal("tensor([1.], requires_grad=True)", TensorPreviewFormatter.Format(leaf));
        using var nonLeaf = leaf * 2;
        Assert.Throws<NotSupportedException>(() => TensorPreviewFormatter.Format(nonLeaf));
        Assert.Throws<OperationCanceledException>(() => TensorPreviewFormatter.Format(leaf, new CancellationToken(true)));
        Assert.False(leaf.IsInvalid); Assert.True(leaf.requires_grad);
    }

    [Fact]
    public void NonContiguousTensorMatchesIndependentCorpusTextWithoutChangingInputStrides()
    {
        NativeRuntimeBootstrap.Initialize();
        var assembly = typeof(TensorNodeTests).Assembly;
        using var stream = assembly.GetManifestResourceStream(assembly.GetManifestResourceNames()
            .Single(name => name.EndsWith("preview-any.cpu.json", StringComparison.Ordinal)))!;
        using var reference = System.Text.Json.JsonDocument.Parse(stream);
        var sample = reference.RootElement.GetProperty("cases").EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == "float32/non-contiguous-source");
        var shape = sample.GetProperty("shape").EnumerateArray().Select(item => item.GetInt64()).ToArray();
        byte[] bytes = Convert.FromBase64String(sample.GetProperty("dataBase64").GetString()!);
        Assert.Equal(sample.GetProperty("dataSha256").GetString(),
            Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(bytes)));
        using var scope = NewDisposeScope();
        var logical = empty(shape, dtype: ScalarType.Float32, device: CPU);
        bytes.CopyTo(logical.bytes);
        // The corpus stores logical values in contiguous order. Rebuild a genuinely strided view with
        // those same values, instead of treating the export's source label as proof of this code path.
        var strided = logical.transpose(0, 1).contiguous().transpose(0, 1);
        Assert.False(strided.is_contiguous());
        var strides = strided.stride();
        Assert.Equal(sample.GetProperty("expectedText").GetString(), TensorPreviewFormatter.Format(strided));
        Assert.False(strided.is_contiguous());
        Assert.Equal(strides, strided.stride());
        Assert.Equal(logical.data<float>().ToArray(), strided.contiguous().data<float>().ToArray());
    }

    private static JsonObject Node(NodeRegistry registry, string id)
    {
        Assert.True(registry.TryGet(id, out var node));
        var inputs = new JsonObject();
        foreach (var input in node.Schema.Inputs)
        {
            if (input.Options?["default"] is { } value) inputs[input.Name] = value.DeepClone();
            else if (input.Options?["options"]?[0] is { } option) inputs[input.Name] = option.DeepClone();
        }
        return new JsonObject { ["class_type"] = id, ["inputs"] = inputs };
    }
    private static JsonObject Prompt(string json) => JsonNode.Parse(json)!.AsObject();
    private sealed class CompositeNode : IRuntimeNode
    {
        public NodeSchema Schema { get; } = new("TestComposite", "Test", "test", [], [new("*")]);
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
            => ValueTask.FromResult(new NodeExecutionOutput([context.Map(new Dictionary<string, RuntimeValue>
            { ["samples"] = context.Own(tensor(new float[] { 1, 0 }).DetachFromDisposeScope()), ["label"] = context.Json(JsonValue.Create("é")) })]));
    }
}
