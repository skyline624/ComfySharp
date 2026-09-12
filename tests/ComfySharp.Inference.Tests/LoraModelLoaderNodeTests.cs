using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Nodes.Tensor;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraModelLoaderNodeTests
{
    [Fact]
    public async Task Bypass_node_keeps_factor_application_in_forward_with_zero_patched_weight_budget()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope(); using var producer = new RuntimeNodeContext(); using var output = new RuntimeNodeContext();
            var config = new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            using var patch = new TrainableLoraPatch(full(new long[]{4,1}, .1f), full(new long[]{1,288}, .01f), 1);
            var state = TrainingNodeValues.CaptureAdapters(producer, new Dictionary<string,TrainableLoraPatch>{{"out.2.weight",patch}}, ScalarType.Float32);
            var inputs = new Dictionary<string,RuntimeValue>{["model"]=producer.Own(model.Retain()),["lora"]=state,
                ["strength_model"]=producer.Json(JsonValue.Create(1)),["bypass"]=producer.Json(JsonValue.Create(true))};
            var result = await new LoraModelLoaderNode(0).ExecuteAsync(output, inputs, default);
            Assert.True(result.Ui!["comfysharp_lora"]![0]!["bypass"]!.GetValue<bool>());
            inputs["bypass"] = producer.Json(JsonValue.Create(false));
            await Assert.ThrowsAsync<NotSupportedException>(async()=>await new LoraModelLoaderNode(0).ExecuteAsync(output,inputs,default));
            producer.Dispose(); patch.Dispose();
            using var input = NativeMath.CpuNoise([1,4,8,8],51); using var text = NativeMath.CpuNoise([1,3,16],52); using var time = tensor(new[]{17.25f});
            using var changed = result.Result[0].GetNative<SdUnet>().Forward(input,time,text); using var baseline = model.Forward(input,time,text);
            Assert.NotEqual(baseline.bytes.ToArray(),changed.bytes.ToArray());
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Theory]
    [InlineData(ScalarType.Float32, .5)] [InlineData(ScalarType.BFloat16, .5)]
    [InlineData(ScalarType.Float32, -.5)] [InlineData(ScalarType.BFloat16, -.5)]
    public async Task Trained_native_state_applies_like_a_file_and_leaves_the_original_model_unchanged(ScalarType dtype, double strength)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        string path = Path.Combine(Path.GetTempPath(), "comfysharp-state-model-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            using var scope = NewDisposeScope(); using var producer = new RuntimeNodeContext(); using var output = new RuntimeNodeContext();
            var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            using var latent = NativeMath.CpuNoise([1, 4, 8, 8], 51); using var text = NativeMath.CpuNoise([1, 3, 16], 52); using var time = tensor(new[] { 17.25f });
            using var baseline = model.Forward(latent, time, text);
            RuntimeValue state;
            using (var patch = new TrainableDifferencePatch(ones(4)))
            using (var optimizer = new LoraTrainingOptimizer(new[] { patch }, "SGD", .1))
            {
                using var loss = patch.Difference.square().sum(); optimizer.Accumulate(loss); optimizer.Step();
                state = TrainingNodeValues.CaptureAdapters(producer, new Dictionary<string, TrainableDifferencePatch> { ["out.2.bias"] = patch }, dtype);
            }
            var values = state.Properties.ToDictionary(p => p.Key, p => p.Value.GetNative<Tensor>());
            using (var stream = File.Create(path)) SafeTensorWriter.Write(stream, values);
            using var file = new SafeTensorFile(path); var plan = LoraFileLoader.Inspect(file, LoraModelAliases.ForUnet(config));
            using var adapter = LoraFileLoader.Load(file, plan, new Dictionary<string, double> { ["model"] = strength });
            using var expectedModel = adapter.ApplyTo(model); using var expected = expectedModel.Forward(latent, time, text);
            var result = await new LoraModelLoaderNode().ExecuteAsync(output, new Dictionary<string, RuntimeValue>
            {
                ["model"] = producer.Own(model.Retain()), ["lora"] = state,
                ["strength_model"] = producer.Json(JsonValue.Create(strength)), ["bypass"] = producer.Json(JsonValue.Create(false))
            }, default);
            producer.Dispose(); file.Dispose(); adapter.Dispose();
            using var actual = result.Result[0].GetNative<SdUnet>().Forward(latent, time, text);
            using var unchanged = model.Forward(latent, time, text);
            Assert.Equal(expected.bytes.ToArray(), actual.bytes.ToArray());
            Assert.NotEqual(baseline.bytes.ToArray(), actual.bytes.ToArray());
            Assert.Equal(baseline.bytes.ToArray(), unchanged.bytes.ToArray());
            Assert.Empty(result.Ui!["comfysharp_lora"]![0]!["unclaimed_tensors"]!.AsArray());
        }
        finally { File.Delete(path); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public async Task Zero_strength_retains_original_without_reading_adapter_and_precancellation_wins()
    {
        using var input = new RuntimeNodeContext(); using var output = new RuntimeNodeContext(); var node = new LoraModelLoaderNode();
        var values = new Dictionary<string, RuntimeValue> { ["model"] = input.Json(JsonValue.Create("unread")),
            ["strength_model"] = input.Json(JsonValue.Create(0)), ["bypass"] = input.Json(JsonValue.Create(true)) };
        var result = await node.ExecuteAsync(output, values, default);
        Assert.Equal("unread", result.Result[0].ToJson()!.GetValue<string>());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await node.ExecuteAsync(output, values, new(true)));
    }
}
