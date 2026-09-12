using System.Text.Json;
using ComfySharp.Inference;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SdSchedulerTests
{
    [Fact]
    public void Every_scheduler_and_partial_slice_match_frozen_source_including_variable_lengths()
    {
        NativeRuntimeBootstrap.Initialize();
        using var stream = GetType().Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.sd-schedulers.reference.json")!;
        using var reference = JsonDocument.Parse(stream);
        foreach (var row in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            string name = row.GetProperty("scheduler").GetString()!;
            int steps = row.GetProperty("steps").GetInt32(); double denoise = row.GetProperty("denoise").GetDouble();
            if (!row.GetProperty("finite").GetBoolean())
            {
                Assert.Throws<ArithmeticException>(() => SdScheduler.Create(name, steps, denoise, SdDiscreteSampling.Default));
                continue;
            }
            using var actual = SdScheduler.Create(name, steps, denoise, SdDiscreteSampling.Default);
            var values = actual.data<float>().ToArray();
            var expected = row.GetProperty("sigmas").EnumerateArray().Select(x => x.GetSingle()).ToArray();
            Assert.True(values.Length == expected.Length, $"{name}/{steps}/{denoise}: length {values.Length} != {expected.Length}");
            for (int i = 0; i < values.Length; i++)
                Assert.True(Math.Abs((double)values[i] - expected[i]) <= 1e-6 + 1e-6 * Math.Abs(expected[i]),
                    $"{name}/{steps}/{denoise}[{i}]: actual={values[i]:R} expected={expected[i]:R}");
            if (values.Length > 0) Assert.Equal(0, values[^1]);
        }
    }

    [Fact]
    public void Schedule_returns_owned_values_and_invalid_requests_do_not_leak_tensors()
    {
        NativeRuntimeBootstrap.Initialize(); _ = SdDiscreteSampling.Default.SigmaMin;
        long before = Tensor.TotalCount; Tensor result;
        using (var scope = NewDisposeScope())
        {
            result = SdScheduler.Create("beta", 20, .5, SdDiscreteSampling.Default).DetachFromDisposeScope();
            Assert.Throws<NotSupportedException>(() => SdScheduler.Create("unknown", 20, 0, SdDiscreteSampling.Default));
            Assert.Throws<NotSupportedException>(() => SdScheduler.Create("normal", 100, .00001, SdDiscreteSampling.Default));
            Assert.Throws<ArgumentOutOfRangeException>(() => SdScheduler.Create("normal", 0, 1, SdDiscreteSampling.Default));
            Assert.Throws<ArgumentOutOfRangeException>(() => SdScheduler.Create("normal", 1, double.NaN, SdDiscreteSampling.Default));
            Assert.Throws<OperationCanceledException>(() => SdScheduler.Create("beta", 10000, 1, SdDiscreteSampling.Default, new(true)));
        }
        using (result) { Assert.False(result.IsInvalid); Assert.Equal(0, result.data<float>().ToArray()[^1]); }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
