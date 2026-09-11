using System.Security.Cryptography;
using System.Text;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[CollectionDefinition("Classical VAE", DisableParallelization = true)]
public sealed class ClassicalVaeCollection;

[Collection("Classical VAE")]
public sealed class ClassicalVaeGraphTests : IDisposable
{
    private readonly int originalThreads;

    public ClassicalVaeGraphTests()
    {
        NativeRuntimeBootstrap.Initialize();
        originalThreads = get_num_threads();
        set_num_threads(1);
    }

    public void Dispose() => set_num_threads(originalThreads);

    [Fact]
    public void EncodeReturnsUnscaledQuantizedMeanWithoutSampling()
    {
        using var scope = NewDisposeScope();
        using var bank = SyntheticWeights(zero: true, modify: weights =>
        {
            using var bias = tensor(new float[] { -3, 2, 5, .25f, 7, 8, 9, 10 });
            weights["quant_conv.bias"].copy_(bias);
        });
        using var vae = new ClassicalVae(bank);
        using var input = zeros(new long[] { 2, 3, 17, 25 });
        manual_seed(1701);
        using var expectedNextNoise = randn(new long[] { 9 });
        manual_seed(1701);
        using var mean = vae.Encode(input);
        using var moments = vae.EncodeMoments(input);
        using var nextNoise = randn(new long[] { 9 });
        Assert.Equal(new long[] { 2, 4, 2, 3 }, mean.shape);
        Assert.Equal(new long[] { 2, 8, 2, 3 }, moments.shape);
        var expected = Enumerable.Range(0, 2).SelectMany(_ => new[] { -3f, 2f, 5f, .25f }
            .SelectMany(value => Enumerable.Repeat(value, 6))).ToArray();
        Assert.Equal(expected, Values(mean));
        Assert.Equal(expectedNextNoise.data<float>().ToArray(), nextNoise.data<float>().ToArray());
        using var meanView = moments.narrow(1, 0, 4).contiguous();
        Assert.Equal(meanView.data<float>().ToArray(), Values(mean));
    }

    [Fact]
    public void CompleteGraphsDependOnInputsAndPreserveCallerInputs()
    {
        using var scope = NewDisposeScope();
        using var bank = SyntheticWeights();
        using var vae = new ClassicalVae(bank);
        using var image = Pattern(new long[] { 1, 3, 16, 24 });
        using var changedImage = image.clone();
        using var changedPixel = changedImage.narrow(0, 0, 1).narrow(1, 1, 1).narrow(2, 7, 1).narrow(3, 13, 1);
        changedPixel.fill_(3);
        var originalPixels = image.data<float>().ToArray();
        using var first = vae.Encode(image);
        using var second = vae.Encode(changedImage);
        Assert.False(first.data<float>().ToArray().SequenceEqual(second.data<float>().ToArray()));
        using var decoded = vae.Decode(first);
        using var otherDecoded = vae.Decode(second);
        Assert.Equal(new long[] { 1, 3, 16, 24 }, decoded.shape);
        Assert.False(decoded.data<float>().ToArray().SequenceEqual(otherDecoded.data<float>().ToArray()));
        Assert.Equal(originalPixels, image.data<float>().ToArray());
        Assert.All(decoded.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void MiddleAttentionAndPostQuantWeightsParticipateInTheGraph()
    {
        using var scope = NewDisposeScope();
        using var originalBank = SyntheticWeights();
        using var changedBank = SyntheticWeights(modify: weights =>
        {
            weights["encoder.mid.attn_1.proj_out.weight"].zero_();
            weights["encoder.mid.attn_1.proj_out.bias"].zero_();
            weights["post_quant_conv.weight"].zero_();
            weights["post_quant_conv.bias"].zero_();
        });
        using var original = new ClassicalVae(originalBank);
        using var changed = new ClassicalVae(changedBank);
        using var image = Pattern(new long[] { 1, 3, 16, 24 });
        using var latent = Pattern(new long[] { 1, 4, 2, 3 });
        using var encodedA = original.Encode(image);
        using var encodedB = changed.Encode(image);
        using var decodedA = original.Decode(latent);
        using var decodedB = changed.Decode(latent);
        Assert.False(encodedA.data<float>().ToArray().SequenceEqual(encodedB.data<float>().ToArray()));
        Assert.False(decodedA.data<float>().ToArray().SequenceEqual(decodedB.data<float>().ToArray()));
    }

    [Fact]
    public void ImageEncodeCenterCropsDropsAlphaAndNormalizesWithoutPreclamping()
    {
        using var scope = NewDisposeScope();
        using var bank = SyntheticWeights();
        using var raw = new ClassicalVae(bank);
        long[]? observedStride = null;
        float[]? observedPixels = null;
        raw.DiagnosticObserver = (name, value) =>
        {
            if (name != "encoder.conv_in.input") return;
            observedStride = value.stride();
            observedPixels = Values(value);
        };
        using var wrapper = new ComfyImageVae(raw);
        var pixels = Enumerable.Range(0, 19 * 26 * 4).Select(i => (i % 257 - 64) / 128f).ToArray();
        using var image = tensor(pixels).reshape(1, 19, 26, 4);
        var normalized = new float[3 * 16 * 24];
        for (int channel = 0; channel < 3; channel++)
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 24; x++)
                    normalized[(channel * 16 + y) * 24 + x] = pixels[((y + 1) * 26 + x + 1) * 4 + channel] * 2 - 1;
        // Match source compute layout as well as values: the crop/movedim and
        // pointwise normalization retain a channels-last NCHW allocation.
        using var expectedInput = tensor(normalized).reshape(1, 3, 16, 24)
            .permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2);
        using var expected = raw.Encode(expectedInput);
        using var actual = wrapper.Encode(image);
        Assert.Equal(new long[] { 1, 4, 2, 3 }, actual.shape);
        Assert.Equal(new long[] { 1152, 1, 72, 3 }, observedStride);
        Assert.Equal(normalized, observedPixels);
        Assert.True(actual.is_contiguous()); // Source wrapper owns a fresh NCHW output buffer.
        Assert.Equal(Values(expected), Values(actual));
        Assert.Equal(pixels, image.data<float>().ToArray());
    }

    [Fact]
    public void RawGraphsPreserveBorrowedStridesAndMeanViewSurvivesMomentStorageOwners()
    {
        using var scope = NewDisposeScope();
        using var bank = SyntheticWeights();
        using var raw = new ClassicalVae(bank);
        var observations = new Dictionary<string, (long[] Stride, bool IsCallerInput)>();
        using var imageStorage = Pattern(new long[] { 2, 16, 24, 3 });
        using var image = imageStorage.permute(0, 3, 1, 2);
        Tensor expectedInput = image;
        raw.DiagnosticObserver = (name, value) => observations[name] = (value.stride(), ReferenceEquals(value, expectedInput));
        Assert.False(image.is_contiguous());
        var imageBefore = Values(image);
        using var mean = raw.Encode(image);
        Assert.Equal(image.stride(), observations["encoder.conv_in.input"].Stride);
        Assert.True(observations["encoder.conv_in.input"].IsCallerInput);
        using var moments = raw.EncodeMoments(image);
        Assert.Equal(moments.stride(), mean.stride());
        Assert.Equal(new long[] { 48, 1, 24, 8 }, mean.stride());
        using var expectedMean = moments.narrow(1, 0, 4);
        Assert.Equal(Values(expectedMean), Values(mean));
        moments.Dispose();
        var meanBefore = Values(mean);
        expectedInput = mean;
        using var decoded = raw.Decode(mean);
        Assert.Equal(mean.stride(), observations["post_quant_conv.input"].Stride);
        Assert.True(observations["post_quant_conv.input"].IsCallerInput);
        Assert.Equal(meanBefore, Values(mean));
        Assert.Equal(imageBefore, Values(image));
        Assert.Equal(new long[] { 2, 3, 16, 24 }, decoded.shape);
        Assert.All(Values(decoded), value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void ImageDecodeNormalizesClampsAndReturnsChannelsLast()
    {
        using var scope = NewDisposeScope();
        using var bank = SyntheticWeights(zero: true, modify: weights =>
        {
            using var bias = tensor(new float[] { -2, 0, 4 });
            weights["decoder.conv_out.bias"].copy_(bias);
        });
        using var raw = new ClassicalVae(bank);
        using var wrapper = new ComfyImageVae(raw);
        using var latent = Pattern(new long[] { 2, 4, 1, 2 });
        var original = latent.data<float>().ToArray();
        using var image = wrapper.Decode(latent);
        Assert.Equal(new long[] { 2, 8, 16, 3 }, image.shape);
        Assert.Equal(new long[] { 384, 16, 1, 128 }, image.stride());
        Assert.False(image.is_contiguous());
        Assert.Equal(Enumerable.Range(0, 2 * 8 * 16).SelectMany(_ => new float[] { 0, .5f, 1 }), Values(image));
        Assert.Equal(original, latent.data<float>().ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AttentionProjectionPreservesTheKeyBufferLayout(bool channelsLast)
    {
        using var scope = NewDisposeScope();
        using var bank = SyntheticWeights();
        using var raw = new ClassicalVae(bank);
        var observed = new Dictionary<string, long[]>();
        raw.DiagnosticObserver = (name, value) => observed[name] = value.stride();
        var image = channelsLast
            ? Pattern(new long[] { 2, 16, 24, 3 }).permute(0, 3, 1, 2)
            : Pattern(new long[] { 2, 3, 16, 24 });
        var latent = channelsLast
            ? Pattern(new long[] { 2, 2, 3, 4 }).permute(0, 3, 1, 2)
            : Pattern(new long[] { 2, 4, 2, 3 });
        using var encoded = raw.Encode(image);
        using var decoded = raw.Decode(latent);
        // Source zeros_like(k) retains k's dense permutation before copy and
        // reshape; projecting the fresh contiguous BMM result loses this layout.
        long[] expected = channelsLast ? new long[] { 768, 1, 384, 128 } : new long[] { 768, 6, 3, 1 };
        Assert.Equal(expected, observed["encoder.mid.attn_1.proj_out.input"]);
        Assert.Equal(expected, observed["decoder.mid.attn_1.proj_out.input"]);
    }

    [Fact]
    public void ResultsSurviveModelsAndAmbientScopesAndRestoreGradMode()
    {
        using var gradMode = set_grad_enabled(true);
        Tensor latent, decoded;
        float[] values;
        using (var scope = NewDisposeScope())
        using (var bank = SyntheticWeights())
        using (var raw = new ClassicalVae(bank))
        using (var wrapper = new ComfyImageVae(raw))
        using (var image = Pattern(new long[] { 1, 16, 16, 3 }).requires_grad_(true))
        {
            latent = wrapper.Encode(image);
            decoded = wrapper.Decode(latent);
            Assert.True(is_grad_enabled());
            Assert.False(latent.requires_grad);
            Assert.False(decoded.requires_grad);
            Assert.Equal(ScalarType.Float32, decoded.dtype);
            values = Values(decoded);
        }
        using (latent)
        using (decoded)
        {
            Assert.Equal(values, Values(decoded));
            Assert.All(latent.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
        }
        Assert.True(latent.IsInvalid);
        Assert.True(decoded.IsInvalid);
    }

    [Fact]
    public void RetainedModelsSurviveOwnerDisposalAndCancellationDoesNotPoisonLaterCalls()
    {
        using var scope = NewDisposeScope();
        using var bank = SyntheticWeights();
        using var original = new ClassicalVae(bank);
        using var retained = original.Retain();
        using var wrapper = new ComfyImageVae(original);
        using var retainedWrapper = wrapper.Retain();
        original.Dispose();
        wrapper.Dispose();
        bank.Dispose();
        using var image = Pattern(new long[] { 1, 3, 8, 8 });
        using var nhwc = image.permute(0, 2, 3, 1);
        Assert.Throws<OperationCanceledException>(() => retained.Encode(image, new(true)));
        Assert.Throws<OperationCanceledException>(() => retainedWrapper.Encode(nhwc, new(true)));
        using var first = retained.Encode(image);
        using var second = retained.Encode(image);
        using var wrapped = retainedWrapper.Encode(nhwc);
        Assert.Equal(first.data<float>().ToArray(), second.data<float>().ToArray());
        first.Dispose();
        Assert.All(second.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
        Assert.Equal(new long[] { 1, 4, 1, 1 }, wrapped.shape);
        Assert.Throws<ObjectDisposedException>(() => original.Retain());
        Assert.Throws<ObjectDisposedException>(() => original.Encode(image));
        Assert.Throws<ObjectDisposedException>(() => wrapper.Encode(nhwc));
    }

    [Fact]
    public void InvalidShapesDtypesAndDisposedInputsFailExplicitly()
    {
        using var scope = NewDisposeScope();
        using var bank = SyntheticWeights();
        using var raw = new ClassicalVae(bank);
        using var wrapper = new ComfyImageVae(raw);
        Assert.Throws<ArgumentException>(() => raw.Encode(zeros(new long[] { 1, 4, 8, 8 })));
        Assert.Throws<ArgumentException>(() => raw.Encode(zeros(new long[] { 1, 3, 7, 8 })));
        Assert.Throws<ArgumentException>(() => raw.Encode(zeros(new long[] { 0, 3, 8, 8 })));
        Assert.Throws<ArgumentException>(() => raw.Encode(zeros(new long[] { 3, 8, 8 })));
        Assert.Throws<ArgumentException>(() => raw.Encode(zeros(new long[] { 1, 3, 8, 8 }, dtype: ScalarType.Float64)));
        Assert.Throws<ArgumentException>(() => raw.Decode(zeros(new long[] { 1, 8, 1, 1 })));
        Assert.Throws<ArgumentException>(() => raw.Decode(zeros(new long[] { 1, 4, 0, 1 })));
        Assert.Throws<ArgumentException>(() => wrapper.Encode(zeros(new long[] { 1, 8, 8, 2 })));
        Assert.Throws<ArgumentException>(() => wrapper.Encode(zeros(new long[] { 1, 7, 8, 3 })));
        using var disposed = zeros(new long[] { 1, 3, 8, 8 });
        disposed.Dispose();
        Assert.Throws<ArgumentException>(() => raw.Encode(disposed));
    }

    internal static ClassicalVaeWeightSet SyntheticWeights(bool zero = false, Action<Dictionary<string, Tensor>>? modify = null)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var config = new ClassicalVaeConfig(32);
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var (name, shape) in ClassicalVaeWeightSchema.Describe(config))
        {
            int count = checked((int)shape.Aggregate(1L, (a, b) => a * b));
            var hash = SHA256.HashData(Encoding.ASCII.GetBytes(name));
            int seed = hash[0] * 256 + hash[1];
            bool norm = name.Contains("norm", StringComparison.Ordinal) && name.EndsWith(".weight", StringComparison.Ordinal);
            var values = Enumerable.Range(0, count).Select(i => zero ? 0 :
                (float)(((i * 31L + seed) % 257 - 128) / 8192.0) + (norm ? 1f : 0f)).ToArray();
            tensors.Add(name, tensor(values).reshape(shape.ToArray()));
        }
        modify?.Invoke(tensors);
        return ClassicalVaeWeightSet.FromOwnedTensors(config, tensors);
    }

    private static Tensor Pattern(long[] shape)
    {
        using var scope = NewDisposeScope();
        int count = checked((int)shape.Aggregate(1L, (a, b) => a * b));
        return tensor(Enumerable.Range(0, count).Select(i => (i % 127 - 48) / 64f).ToArray())
            .reshape(shape).MoveToOuterDisposeScope();
    }

    private static float[] Values(Tensor value)
    {
        using var contiguous = value.contiguous();
        return contiguous.data<float>().ToArray();
    }
}
