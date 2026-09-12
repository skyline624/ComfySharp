using System.Text.Json;
using System.Security.Cryptography;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraBypassTests
{
    private static Tensor Read(JsonElement value) => tensor(value.GetProperty("values").EnumerateArray().Select(n => n.GetSingle()).ToArray(),
        value.GetProperty("shape").EnumerateArray().Select(n => n.GetInt64()).ToArray()).clone().requires_grad_();
    private static void Near(Tensor value, JsonElement reference, double absolute, double relative)
    {
        Assert.Equal(reference.GetProperty("shape").EnumerateArray().Select(n => n.GetInt64()), value.shape);
        var expected = reference.GetProperty("values").EnumerateArray().Select(n => n.GetSingle()).ToArray();
        var actual = value.data<float>().ToArray();
        for (int i = 0; i < actual.Length; i++) Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= absolute + relative * Math.Abs(expected[i]), $"Element {i}: {actual[i]} vs {expected[i]}");
    }
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)]
    public void Forward_and_gradients_match_frozen_bypass_functions(int index)
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount; int threads = get_num_threads(); set_num_threads(1);
        try
        {
            using var scope = NewDisposeScope(); using var gradMode = set_grad_enabled(true);
            using var stream = GetType().Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora-bypass.reference.json")!;
            using var buffer = new MemoryStream(); stream.CopyTo(buffer); byte[] fixture = buffer.ToArray();
            Assert.Equal("7cd58396213b27d4e6ed6963141ab19d4677906ca08e45b2e4c0ce4901aa8519", Convert.ToHexStringLower(SHA256.HashData(fixture)));
            using var document = JsonDocument.Parse(fixture); var root = document.RootElement; var row = root.GetProperty("cases")[index];
            var values = row.GetProperty("tensors").EnumerateObject().ToDictionary(p => p.Name, p => Read(p.Value));
            double strength = row.GetProperty("strength").GetDouble(); double? alpha = row.GetProperty("alpha").ValueKind == JsonValueKind.Null ? null : row.GetProperty("alpha").GetDouble();
            long[]? kernel = row.GetProperty("kernel").ValueKind == JsonValueKind.Null ? null : row.GetProperty("kernel").EnumerateArray().Select(n => n.GetInt64()).ToArray();
            using var result = LoraBypassMath.Apply(values["input"], values["baseOutput"], values["up"], values["down"], strength, alpha,
                values.GetValueOrDefault("mid"), kernel, row.GetProperty("stride").GetInt64(), row.GetProperty("padding").GetInt64());
            double absolute = root.GetProperty("absTolerance").GetDouble(), relative = root.GetProperty("relTolerance").GetDouble();
            Near(result, row.GetProperty("output"), absolute, relative);
            using var loss = result.square().mean(); loss.backward();
            foreach (var (name, value) in values)
            {
                using var gradient = value.grad; Assert.NotNull(gradient);
                Near(gradient!, row.GetProperty("gradients").GetProperty(name), absolute, relative);
            }
            if (row.GetProperty("doraField").GetBoolean())
            {
                using var dora = full(new long[] { 4, 1 }, 3f); using var patch = new LoraWeightPatch(values["up"], values["down"], strength, alpha, doraScale: dora);
                using var actual = patch.ApplyBypass(values["input"], values["baseOutput"]);
                Near(actual, row.GetProperty("output"), absolute, relative);
            }
        }
        finally { set_num_threads(threads); }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Fact]
    public void Invalid_geometry_and_cancellation_leave_borrowed_inputs_alive()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        {
            var input = ones(2, 3); var output = ones(2, 4); var up = ones(4, 2); var down = ones(2, 3);
            long borrowed = Tensor.TotalCount;
            Assert.Throws<OperationCanceledException>(() => LoraBypassMath.Apply(input, output, up, down, cancellationToken: new(true)));
            Assert.Throws<ArgumentException>(() => LoraBypassMath.Apply(input, output, up, down, kernelSize: new long[] { 3, 3 }));
            Assert.Equal(borrowed, Tensor.TotalCount); Assert.Equal(new long[] { 2, 3 }, input.shape);
        }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
