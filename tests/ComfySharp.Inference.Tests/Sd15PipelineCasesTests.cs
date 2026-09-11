using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.RuntimeProbe;
using ComfySharp.Tokenization;
using Xunit;

namespace ComfySharp.Inference.Tests;

// Managed input/resource tests only: no Torch constructor, bootstrap or graph forward.
public sealed class Sd15PipelineCasesTests
{
    private const string ProtocolPin = "f0e4f537414c6f8692c852832d040fe3fd5dc23fdc270bb351a16bbba6cd75e7";

    [Fact]
    public void EmbeddedProtocolHasTheFrozenBytesAndExactlyFourOrdinalCaseIds()
    {
        using var stream = typeof(Sd15PipelineCases).Assembly.GetManifestResourceStream(
            "ComfySharp.RuntimeProbe.Fixtures.sd15-pipeline.protocol.json");
        Assert.NotNull(stream);
        Assert.Equal(78526, stream.Length);
        Assert.Equal(ProtocolPin, Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant());
        Assert.Equal(ProtocolPin, Sd15PipelineCases.ProtocolSha256);
        Assert.Equal(new[] { "empty-one-step", "weighted-three-step", "two-chunks-separate", "maximum-start" },
            Sd15PipelineCases.All.Select(c => c.Id));
        foreach (var item in Sd15PipelineCases.All) Assert.Same(item, Sd15PipelineCases.Get(item.Id));
        Assert.Throws<ArgumentNullException>(() => Sd15PipelineCases.Get(null!));
        Assert.Throws<ArgumentException>(() => Sd15PipelineCases.Get("Empty-one-step"));
        Assert.Throws<ArgumentException>(() => Sd15PipelineCases.Get("empty-one-step "));
        Assert.Throws<ArgumentException>(() => Sd15PipelineCases.Get("unknown"));
        var list = Assert.IsAssignableFrom<IList<Sd15PipelineCase>>(Sd15PipelineCases.All);
        Assert.True(list.IsReadOnly);
        Assert.Throws<NotSupportedException>(() => list[0] = list[1]);
    }

    [Theory]
    [InlineData("empty-one-step", 1, "52727422e6b362b1568e39c239ef6fc81a336ed85ea48dc45ff3b4127bfa292b", "3c076e693c51f8000439e20c7107a4ee0a0f883f418fd78a2cf66a946a3a8e9c")]
    [InlineData("weighted-three-step", 1, "69924ea37d7ed7a2fb3f81733f5b9eefef2be39371ff764d30fb8061113a7cdf", "3bb2a6233a39c9f1f4bf50e3b0f1849d7e28cb46845ac838a84693ee03bd412a")]
    [InlineData("two-chunks-separate", 2, "279b573be888c5403a724cb58e24f178dff46055e2f39d36c27bdc49064992d5", "c31a22b7f25d8cbe512651e536c7e4d1b1854a84d1429f66aa92c64211d7d057")]
    [InlineData("maximum-start", 1, "67f137d4b7aa75a70c4f9fc6fc46d43144b97672b6b3541b9cd53cd3da53f062", "f5e113d0e100491cb7b74b5dcf9efa8d811db876880c1f36c1c5a50285264470")]
    public void RealTokenizationMatchesSourceTriplesAndInputCopiesKeepFrozenBits(
        string id, int positiveChunks, string noiseHash, string sigmaHash)
    {
        var item = Sd15PipelineCases.Get(id);
        var conditioning = item.TokenizeAndVerify();
        Assert.Equal(ClipProfile.Sd1L, conditioning.Positive.Profile);
        Assert.Equal(ClipProfile.Sd1L, conditioning.Negative.Profile);
        Assert.Equal(positiveChunks, conditioning.Positive.Chunks.Count);
        Assert.Single(conditioning.Negative.Chunks);
        using var stream = typeof(Sd15PipelineCasesTests).Assembly.GetManifestResourceStream(
            "ComfySharp.RuntimeProbe.Fixtures.sd15-pipeline.protocol.json");
        Assert.NotNull(stream);
        using var protocol = JsonDocument.Parse(stream);
        var root = protocol.RootElement;
        var definition = root.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("id").GetString() == id);
        AssertSourceTokens(conditioning.Positive, definition.GetProperty("positiveText").GetString()!, root);
        AssertSourceTokens(conditioning.Negative, definition.GetProperty("negativeText").GetString()!, root);
        Assert.Equal(definition.GetProperty("scale").GetDouble(), conditioning.GuidanceScale);
        Assert.Equal(definition.GetProperty("maximumDenoise").GetBoolean(), conditioning.MaximumDenoise);
        Assert.Equal(3, item.Repetitions);
        Assert.Equal(ProtocolPin, item.ProtocolSha256);

        var noise = item.NoiseValues();
        var sigmas = item.SigmaValues();
        Assert.Equal(80, noise.Length);
        Assert.Equal(definition.GetProperty("steps").GetInt32() + 1, sigmas.Length);
        // Hash the supplied values; do not recalculate the generator or invent expected outputs.
        Assert.Equal(noiseHash, HashValues(noise));
        Assert.Equal(sigmaHash, HashValues(sigmas));
        Assert.Equal(noiseHash, item.NoiseSha256);
        Assert.Equal(sigmaHash, item.SigmaSha256);
        Assert.Equal(id == "maximum-start" ? 0x4169d592 : 0x3fc00000, BitConverter.SingleToInt32Bits(sigmas[0]));
        Assert.Equal(0, BitConverter.SingleToInt32Bits(sigmas[^1]));
        noise[0] = float.NaN;
        sigmas[0] = float.NaN;
        var secondNoise = Sd15PipelineCases.Get(id).NoiseValues();
        var secondSigmas = Sd15PipelineCases.Get(id).SigmaValues();
        Assert.NotSame(noise, secondNoise);
        Assert.NotSame(sigmas, secondSigmas);
        Assert.Equal(noiseHash, HashValues(secondNoise));
        Assert.Equal(sigmaHash, HashValues(secondSigmas));
    }

    [Fact]
    public void PreCancelledTokenizationPreservesTheCancellationTokenAndCaseInputs()
    {
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        foreach (var item in Sd15PipelineCases.All)
        {
            var error = Assert.Throws<OperationCanceledException>(() => item.TokenizeAndVerify(cancelled.Token));
            Assert.Equal(cancelled.Token, error.CancellationToken);
            Assert.Equal(item.NoiseSha256, HashValues(item.NoiseValues()));
            Assert.Equal(item.SigmaSha256, HashValues(item.SigmaValues()));
        }
    }

    private static void AssertSourceTokens(ClipTokenization actual, string textId, JsonElement protocol)
    {
        var source = protocol.GetProperty("texts").EnumerateArray().Single(t => t.GetProperty("id").GetString() == textId);
        var chunks = source.GetProperty("sourceChunks");
        Assert.Equal(chunks.GetArrayLength(), actual.Chunks.Count);
        for (int row = 0; row < chunks.GetArrayLength(); row++)
        {
            Assert.Equal(77, actual.Chunks[row].Count);
            for (int column = 0; column < 77; column++)
            {
                var expected = chunks[row][column];
                var token = actual.Chunks[row][column];
                Assert.Equal(expected[0].GetInt32(), token.Id);
                Assert.Equal(expected[2].GetInt32(), token.WordId);
                Assert.Equal(ulong.Parse(expected[1].GetString()!, NumberStyles.HexNumber, CultureInfo.InvariantCulture),
                    unchecked((ulong)BitConverter.DoubleToInt64Bits(token.Weight)));
            }
        }
    }

    private static string HashValues(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float)), values[i]);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }
}
