using System.Text.Json;
using Xunit;
using ComfySharp.Inference;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdHeunSamplerTests
{
    [Fact]
    public void Real_denoiser_is_retained_and_terminal_step_matches_euler_after_parents_dispose()
    {
        NativeRuntimeBootstrap.Initialize(); int threads = get_num_threads(); set_num_threads(1);
        Tensor result; float[] expected;
        try
        {
            using (var scope = NewDisposeScope())
            {
                var config = new SdUnetConfig(32, 16, SdAttentionHeadMode.FixedCount, 4, false);
                using var bank = SdSyntheticInputs.CreateUnet(config);
                using var model = new SdUnet(bank);
                using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon);
                using var heun = new SdHeunSampler(denoiser);
                using var euler = new SdEulerSampler(denoiser);
                bank.Dispose(); model.Dispose(); denoiser.Dispose();
                var initial = zeros(new long[] { 1, 4, 4, 5 });
                var context = zeros(new long[] { 1, 3, 16 });
                var schedule = tensor(new[] { 1f, 0f });
                using var lastEuler = euler.Sample(initial, schedule, context, null);
                expected = lastEuler.data<float>().ToArray();
                result = heun.Sample(initial, schedule, context, null);
                heun.Dispose();
                Assert.Throws<ObjectDisposedException>(() => heun.Sample(initial, schedule, context, null));
            }
            using (result) Assert.Equal(expected, result.data<float>().ToArray());
        }
        finally { set_num_threads(threads); }
    }

    [Fact]
    public void Trajectory_and_every_predictor_corrector_input_match_frozen_source()
    {
        NativeRuntimeBootstrap.Initialize();
        using var file = GetType().Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.heun.reference.json")!;
        using var reference = JsonDocument.Parse(file);
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            using var scope = NewDisposeScope();
            var initial = tensor(Values(item.GetProperty("initial"))).reshape(2, 4, 1, 1);
            var before = initial.data<float>().ToArray();
            var sigmas = tensor(Values(item.GetProperty("sigmas")));
            var expectedCalls = item.GetProperty("calls").EnumerateArray().ToArray(); int calls = 0;
            using var result = SdHeunSampler.Integrate(initial, sigmas, (x, sigma) =>
            {
                Assert.True(calls < expectedCalls.Length);
                Compare(x, expectedCalls[calls].GetProperty("x"));
                Compare(sigma, expectedCalls[calls++].GetProperty("sigma"));
                using var local = NewDisposeScope();
                return (x * .25 + sigma.reshape(2, 1, 1, 1) * .125).DetachFromDisposeScope();
            });
            Assert.Equal(expectedCalls.Length, calls);
            Compare(result, item.GetProperty("output"));
            Assert.Equal(before, initial.data<float>().ToArray());
            result.fill_(42); Assert.Equal(before, initial.data<float>().ToArray());
        }
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void Cancellation_or_failure_during_corrector_releases_states_and_preserves_inputs(bool fail)
    {
        NativeRuntimeBootstrap.Initialize(); long baseline = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        using (var cancellation = new CancellationTokenSource())
        using (var grad = set_grad_enabled(true))
        {
            var initial = ones(new long[] { 1, 4, 2, 2 }, requires_grad: true);
            var sigmas = tensor(new[] { 1f, .5f, 0f }); int calls = 0;
            Tensor? predictor = null;
            Tensor Model(Tensor x, Tensor sigma)
            {
                Assert.False(is_grad_enabled());
                if (++calls == 2)
                {
                    predictor = x;
                    if (fail) throw new InvalidOperationException("corrector failed");
                    cancellation.Cancel();
                }
                return zeros_like(x);
            }
            if (fail) Assert.Throws<InvalidOperationException>(() => SdHeunSampler.Integrate(initial, sigmas, Model, cancellation.Token));
            else Assert.Throws<OperationCanceledException>(() => SdHeunSampler.Integrate(initial, sigmas, Model, cancellation.Token));
            Assert.Equal(2, calls); Assert.NotNull(predictor); Assert.True(predictor.IsInvalid);
            Assert.True(is_grad_enabled()); Assert.All(initial.data<float>().ToArray(), x => Assert.Equal(1, x));
        }
        Assert.Equal(baseline, Tensor.TotalCount);
    }

    [Fact]
    public void Precancelled_or_invalid_schedule_never_evaluates_denoiser()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        var initial = zeros(new long[] { 1, 4, 2, 2 });
        Tensor Never(Tensor x, Tensor sigma) => throw new Exception("unexpected evaluation");
        Assert.Throws<OperationCanceledException>(() => SdHeunSampler.Integrate(initial, tensor(new[] { 1f, 0f }), Never, new(true)));
        foreach (var values in new[] { new[] { 1f }, new[] { 0f, 0f }, new[] { 1f, 2f, 0f }, new[] { 1f, .1f }, new[] { float.NaN, 0f } })
            Assert.Throws<ArgumentException>(() => SdHeunSampler.Integrate(initial, tensor(values), Never));
    }

    private static float[] Values(JsonElement item) => item.EnumerateArray().Select(x => x.GetSingle()).ToArray();
    private static void Compare(Tensor actual, JsonElement expected)
    {
        var a = actual.data<float>().ToArray(); var e = Values(expected); Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < a.Length; i++) Assert.InRange(Math.Abs((double)a[i] - e[i]), 0, 1e-6 + 1e-6 * Math.Abs(e[i]));
    }
}
