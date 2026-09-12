using System.Text.Json;
using ComfySharp.Inference;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdDpmpp2MSamplerTests
{
    [Fact]
    public void Terminal_step_uses_real_guided_prediction_and_result_outlives_disposed_parents()
    {
        NativeRuntimeBootstrap.Initialize(); int previousThreads=get_num_threads(); set_num_threads(1);
        Tensor result; float[] expected;
        try
        {
            using(var scope=NewDisposeScope())
            {
                var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
                using var bank=SdSyntheticInputs.CreateUnet(config);
                using var model=new SdUnet(bank);
                using var denoiser=new SdDenoiser(model,SdPredictionKind.Epsilon);
                using var sampler=new SdDpmpp2MSampler(denoiser);
                var initial=zeros(new long[]{1,4,4,5});var positive=zeros(new long[]{1,3,16});var negative=ones(new long[]{1,3,16});
                var guidance=new SdGuidanceOptions{Scale=3.5,BatchMode=SdGuidanceBatchMode.Separate};
                using var prediction=denoiser.DenoiseGuided(initial,tensor(new[]{1f}),positive,negative,guidance);
                expected=prediction.data<float>().ToArray();
                bank.Dispose();model.Dispose();denoiser.Dispose();
                var schedule=tensor(new[]{1f,0f});
                result=sampler.Sample(initial,schedule,positive,negative,guidance);
                sampler.Dispose();
                Assert.Throws<ObjectDisposedException>(()=>sampler.Sample(initial,schedule,positive,negative,guidance));
            }
            using(result) Assert.Equal(expected,result.data<float>().ToArray());
        }
        finally{set_num_threads(previousThreads);}
    }

    [Fact]
    public void Multistep_trajectory_and_every_model_input_match_frozen_source()
    {
        NativeRuntimeBootstrap.Initialize();
        using var file = GetType().Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.dpmpp-2m.reference.json")!;
        using var reference = JsonDocument.Parse(file);
        foreach (var row in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            using var scope = NewDisposeScope();
            var initial = tensor(Values(row.GetProperty("initial"))).reshape(2,4,1,1);
            var before = initial.data<float>().ToArray();
            var sigmas = tensor(Values(row.GetProperty("sigmas")));
            var expectedCalls = row.GetProperty("calls").EnumerateArray().ToArray(); int calls = 0;
            Tensor Model(Tensor x, Tensor sigma)
            {
                Assert.True(calls < expectedCalls.Length);
                Compare(x, expectedCalls[calls].GetProperty("x"));
                Compare(sigma, expectedCalls[calls++].GetProperty("sigma"));
                using var local = NewDisposeScope();
                return (x * .25 + sigma.reshape(2,1,1,1) * .125).DetachFromDisposeScope();
            }
            if (!row.GetProperty("finite").GetBoolean())
                Assert.Throws<ArithmeticException>(() => SdDpmpp2MSampler.Integrate(initial, sigmas, Model));
            else
            {
                using var actual = SdDpmpp2MSampler.Integrate(initial, sigmas, Model);
                Assert.Equal(expectedCalls.Length, calls); Compare(actual, row.GetProperty("output"));
                actual.fill_(42);
            }
            Assert.Equal(before, initial.data<float>().ToArray());
        }
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void Failure_or_cancellation_after_history_exists_releases_native_state(bool fail)
    {
        NativeRuntimeBootstrap.Initialize(); long baseline = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        using (var cancellation = new CancellationTokenSource())
        using (var grad = set_grad_enabled(true))
        {
            var initial = ones(new long[] {1,4,2,2}, requires_grad:true);
            var sigmas = tensor(new[] {2f,1f,.5f,0f}); int calls=0; Tensor? state=null;
            Tensor Model(Tensor x, Tensor sigma)
            {
                Assert.False(is_grad_enabled());
                if (++calls==2)
                {
                    state=x;
                    if (fail) throw new InvalidOperationException("second evaluation failed");
                    cancellation.Cancel();
                }
                return zeros_like(x);
            }
            if(fail) Assert.Throws<InvalidOperationException>(()=>SdDpmpp2MSampler.Integrate(initial,sigmas,Model,cancellation.Token));
            else Assert.Throws<OperationCanceledException>(()=>SdDpmpp2MSampler.Integrate(initial,sigmas,Model,cancellation.Token));
            Assert.Equal(2,calls); Assert.NotNull(state); Assert.True(state.IsInvalid); Assert.True(is_grad_enabled());
            Assert.All(initial.data<float>().ToArray(),v=>Assert.Equal(1,v));
        }
        Assert.Equal(baseline,Tensor.TotalCount);
    }

    [Fact]
    public void Invalid_schedules_and_pre_cancellation_never_call_model()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        var initial=zeros(new long[]{1,4,1,1});
        Tensor Never(Tensor x,Tensor sigma)=>throw new Exception("unexpected model call");
        Assert.Throws<OperationCanceledException>(()=>SdDpmpp2MSampler.Integrate(initial,tensor(new[]{1f,0f}),Never,new(true)));
        foreach(var values in new[]{new[]{1f},new[]{1f,2f,0f},new[]{0f,0f},new[]{float.NaN,0f},new[]{1f,.5f}})
            Assert.Throws<ArgumentException>(()=>SdDpmpp2MSampler.Integrate(initial,tensor(values),Never));
    }

    private static float[] Values(JsonElement values)=>values.EnumerateArray().Select(x=>x.GetSingle()).ToArray();
    private static void Compare(Tensor actual,JsonElement expected)
    {
        var a=actual.data<float>().ToArray();var e=Values(expected);Assert.Equal(e.Length,a.Length);
        for(int i=0;i<a.Length;i++) Assert.InRange(Math.Abs((double)a[i]-e[i]),0,1e-6+1e-6*Math.Abs(e[i]));
    }
}
