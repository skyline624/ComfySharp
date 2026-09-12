using System.Security.Cryptography;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraTrainingBypassTests
{
    private static Tensor Read(JsonElement value) => tensor(value.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray(),
        value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray()).clone();
    private static void Compare(Tensor actual, JsonElement expected)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()), actual.shape);
        var wanted = expected.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var values = actual.data<float>().ToArray();
        for (int i = 0; i < wanted.Length; i++) Assert.InRange(Math.Abs((double)values[i] - wanted[i]), 0, 3e-5 + 3e-5 * Math.Abs(wanted[i]));
    }
    private static Dictionary<string, Tensor> Parameters(TrainableWeightPatch patch) => patch switch
    {
        TrainableLoraPatch lora => new() { ["up"] = lora.Up, ["down"] = lora.Down, ["alpha"] = lora.AlphaParameter! },
        TrainableDifferencePatch diff => new() { ["difference"] = diff.Difference },
        _ => throw new InvalidOperationException()
    };

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Source_bypass_graph_matches_outputs_all_gradients_and_two_updates(bool linear)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope(); using var grad = set_grad_enabled(true);
            using var resource = GetType().Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora-training-bypass.reference.json")!;
            Assert.Equal("31e3879b3139c29a9c19d3e24c4e95df65e5006eda6dc758cf3dfc3797fa48dd", Convert.ToHexStringLower(SHA256.HashData(resource)));
            resource.Position = 0; using var json = JsonDocument.Parse(resource);
            var row = json.RootElement.GetProperty("cases").EnumerateArray().Single(v => v.GetProperty("linearProjection").GetBoolean() == linear);
            var config = new SdUnetConfig(32,16,linear ? SdAttentionHeadMode.FixedSize : SdAttentionHeadMode.FixedCount,linear ? 8 : 4,linear);
            using var bank = SdSyntheticInputs.CreateUnet(config); using var model = new SdUnet(bank);
            var patches = new Dictionary<string, TrainableWeightPatch>();
            try
            {
                foreach (var pair in row.GetProperty("initial").EnumerateObject())
                    patches.Add(pair.Name, pair.Value.TryGetProperty("difference", out var difference)
                        ? new TrainableDifferencePatch(Read(difference))
                        : new TrainableLoraPatch(Read(pair.Value.GetProperty("up")), Read(pair.Value.GetProperty("down")),
                            pair.Value.GetProperty("alpha").GetProperty("values")[0].GetDouble(), trainAlpha: true));
                var input = Read(row.GetProperty("latent")); var context = Read(row.GetProperty("context"));
                var times = Read(row.GetProperty("timesteps")); var target = Read(row.GetProperty("target"));
                using var baseline = model.Forward(input,times,context);
                Assert.Throws<NotSupportedException>(() => model.ForwardForTraining(input,times,context,patches,143,bypassMode:true));
                foreach (var step in row.GetProperty("steps").EnumerateArray())
                {
                    using var iteration = NewDisposeScope();
                    foreach (var patch in patches.Values) foreach (var value in patch.Parameters) { using var g = value.grad; g?.zero_(); }
                    // Only norm/bias differences count toward the patched-weight allowance (32 + 4 floats).
                    using var prediction = model.ForwardForTraining(input,times,context,patches,144,bypassMode:true);
                    Compare(prediction,step.GetProperty("output"));
                    using var loss = (prediction-target).square().mean(); loss.backward();
                    Assert.InRange(Math.Abs(loss.item<float>()-step.GetProperty("loss").GetDouble()),0,3e-5);
                    foreach (var (name, patch) in patches)
                        foreach (var (key, value) in Parameters(patch))
                        {
                            using var gradient = value.grad; Assert.NotNull(gradient);
                            Compare(gradient!,step.GetProperty("gradients").GetProperty(name).GetProperty(key));
                            using (var noGrad = no_grad()) value.add_(gradient!,alpha:-.01);
                            Compare(value,step.GetProperty("updated").GetProperty(name).GetProperty(key));
                        }
                    foreach (string name in UnetWeightSchema.Describe(config).Keys) Assert.Null(bank.GetTensor(name).grad);
                }
                using var unchanged = model.Forward(input,times,context); Assert.Equal(baseline.bytes.ToArray(),unchanged.bytes.ToArray());
                using var trained = model.ForwardForTraining(input,times,context,patches,144,bypassMode:true);
                using var state = LoraTrainingState.Capture(patches,ScalarType.Float32);
                // The existing native loader consumes the trained snapshot without intermediate files.
                using var source = new NativeLoraTensorSource(state.Tensors);
                var plan = LoraFileLoader.Inspect(source,LoraModelAliases.ForUnet(config));
                using var adapter = LoraFileLoader.Load(source,plan); using var reloaded = adapter.ApplyBypassTo(model, maxPatchedWeightBytes:144);
                foreach (var patch in patches.Values) patch.Dispose();
                source.Dispose(); state.Dispose(); adapter.Dispose();
                using var predictionAfter = reloaded.Forward(input,times,context);
                using var error = (trained-predictionAfter).abs().max(); Assert.InRange(error.item<float>(),0,3e-5);
            }
            finally { foreach (var patch in patches.Values) patch.Dispose(); }
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Bypass_errors_and_cancellation_preserve_parameter_owners()
    {
        NativeRuntimeBootstrap.Initialize(); long before=Tensor.TotalCount;
        using (var scope=NewDisposeScope())
        {
            var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var bank=SdSyntheticInputs.CreateUnet(config); using var model=new SdUnet(bank);
            using var patch=new TrainableLoraPatch(ones([4,1])*.01,ones([1,288])*.01,1,trainAlpha:true);
            using var input=zeros([1,4,8,8]); using var context=zeros([1,3,16]); using var times=zeros([1]);
            var patches=new Dictionary<string,TrainableLoraPatch>{{"out.2.weight",patch}};
            Assert.Throws<OperationCanceledException>(()=>model.ForwardForTraining(input,times,context,patches,0,new(true),true));
            Assert.Throws<ArgumentException>(()=>model.ForwardForTraining(input,times,context,new Dictionary<string,TrainableLoraPatch>{{"time_embed.0.weight",patch}},0,bypassMode:true));
            Assert.Throws<ArgumentException>(()=>model.ForwardForTraining(input,times,context,new Dictionary<string,TrainableLoraPatch>(),0,bypassMode:true));
            Assert.Throws<NotSupportedException>(()=>model.ForwardForTraining(input,times,context,patches,0));
            using var kept=patch.Retain(); patch.Dispose();
            using var result=model.ForwardForTraining(input,times,context,new Dictionary<string,TrainableLoraPatch>{{"out.2.weight",kept}},0,bypassMode:true);
            using var loss=result.square().mean(); loss.backward(); using var alphaGradient=kept.AlphaParameter!.grad;
            Assert.NotNull(alphaGradient);
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
