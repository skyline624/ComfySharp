using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>The complete classical image VAE graph, Float32 on CPU/CUDA and deterministic mean encoding.
/// Latents are raw VAE values; diffusion scaling belongs to the conditioning/sampling boundary.</summary>
public sealed class ClassicalVae : IDisposable
{
    private readonly object gate = new();
    private ClassicalVaeWeightSet? weights;

    public ClassicalVae(ClassicalVaeWeightSet weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        this.weights = weights.Retain();
        Config = weights.Config;
    }

    public ClassicalVaeConfig Config { get; }
    public Device Device { get { using var bank = RetainWeights(); return bank.Device; } }
    public ClassicalVae To(Device device, CancellationToken cancellationToken = default)
    {
        using var bank = RetainWeights(); using var moved = bank.To(device, cancellationToken); return new ClassicalVae(moved);
    }

    // Optional test diagnostics borrow the tensor only for the callback duration.
    internal Action<string, Tensor>? DiagnosticObserver { get; set; }

    public ClassicalVae Retain()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(weights is null, this);
            return new ClassicalVae(weights) { DiagnosticObserver = DiagnosticObserver };
        }
    }

    /// <summary>Encode RGB NCHW, conventionally [-1,1], into an owned unscaled four-channel mean.
    /// Values are not clamped, and no posterior noise is sampled.</summary>
    public Tensor Encode(Tensor image, CancellationToken cancellationToken = default) =>
        EncodeCore(image, returnMoments: false, cancellationToken);

    /// <summary>Diagnostic access to the eight raw post-quant-convolution moments, before posterior processing.</summary>
    public Tensor EncodeMoments(Tensor image, CancellationToken cancellationToken = default) =>
        EncodeCore(image, returnMoments: true, cancellationToken);

    private Tensor EncodeCore(Tensor image, bool returnMoments, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var operation = RetainWeights();
        NativeRuntimeBootstrap.Initialize();
        ValidateNchw(image, ClassicalVaeConfig.ImageChannels, ClassicalVaeConfig.Compression, nameof(image));
        InferenceDevice.RequireSame(operation.Device, image, nameof(image));
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        // Preserve source layout: an NHWC image moved to NCHW can select the
        // channels-last convolution route without a contiguous conversion.
        DiagnosticObserver?.Invoke("encoder.conv_in.input", image);
        var x = Conv(image, operation, "encoder.conv_in");
        for (int level = 0; level < ClassicalVaeConfig.Levels; level++)
        {
            for (int block = 0; block < ClassicalVaeConfig.EncoderBlocksPerLevel; block++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                x = Advance(x, Residual(x, operation, $"encoder.down.{level}.block.{block}"));
            }
            if (level < ClassicalVaeConfig.Levels - 1)
            {
                cancellationToken.ThrowIfCancellationRequested();
                x = Advance(x, Downsample(x, operation, $"encoder.down.{level}.downsample.conv"));
            }
        }
        x = Middle(x, operation, "encoder.mid", cancellationToken);
        x = Advance(x, nn.functional.silu(Normalize(x, operation, "encoder.norm_out"), inplace: true));
        x = Advance(x, Conv(x, operation, "encoder.conv_out"));
        x = Advance(x, Conv(x, operation, "quant_conv", padding: 0));
        cancellationToken.ThrowIfCancellationRequested();
        var result = returnMoments ? x : x.narrow(1, 0, ClassicalVaeConfig.LatentChannels);
        return result.DetachFromDisposeScope();
    }

    /// <summary>Decode raw four-channel NCHW latents into owned raw RGB NCHW, without image-range normalization.</summary>
    public Tensor Decode(Tensor latent, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var operation = RetainWeights();
        NativeRuntimeBootstrap.Initialize();
        ValidateNchw(latent, ClassicalVaeConfig.LatentChannels, 1, nameof(latent));
        InferenceDevice.RequireSame(operation.Device, latent, nameof(latent));
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        DiagnosticObserver?.Invoke("post_quant_conv.input", latent);
        var x = Conv(latent, operation, "post_quant_conv", padding: 0);
        x = Advance(x, Conv(x, operation, "decoder.conv_in"));
        x = Middle(x, operation, "decoder.mid", cancellationToken);
        for (int level = ClassicalVaeConfig.Levels - 1; level >= 0; level--)
        {
            for (int block = 0; block < ClassicalVaeConfig.DecoderBlocksPerLevel; block++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                x = Advance(x, Residual(x, operation, $"decoder.up.{level}.block.{block}"));
            }
            if (level > 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
                x = Advance(x, Upsample(x, operation, $"decoder.up.{level}.upsample.conv"));
            }
        }
        x = Advance(x, nn.functional.silu(Normalize(x, operation, "decoder.norm_out"), inplace: true));
        x = Advance(x, Conv(x, operation, "decoder.conv_out"));
        cancellationToken.ThrowIfCancellationRequested();
        return x.DetachFromDisposeScope();
    }

    private Tensor Middle(Tensor ownedInput, ClassicalVaeWeightSet bank, string prefix, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var x = Advance(ownedInput, Residual(ownedInput, bank, prefix + ".block_1"));
        cancellationToken.ThrowIfCancellationRequested();
        x = Advance(x, Attention(x, bank, prefix + ".attn_1"));
        cancellationToken.ThrowIfCancellationRequested();
        return Advance(x, Residual(x, bank, prefix + ".block_2"));
    }

    private static Tensor Residual(Tensor input, ClassicalVaeWeightSet bank, string prefix)
    {
        using var scope = NewDisposeScope();
        var h = nn.functional.silu(Normalize(input, bank, prefix + ".norm1"), inplace: true);
        h = Conv(h, bank, prefix + ".conv1");
        h = nn.functional.silu(Normalize(h, bank, prefix + ".norm2"), inplace: true);
        // Source dropout is zero in this frozen inference topology.
        h = Conv(h, bank, prefix + ".conv2");
        var shortcut = input.shape[1] == h.shape[1] ? input : Conv(input, bank, prefix + ".nin_shortcut", padding: 0);
        return (shortcut + h).MoveToOuterDisposeScope();
    }

    private Tensor Attention(Tensor input, ClassicalVaeWeightSet bank, string prefix)
    {
        using var scope = NewDisposeScope();
        var shape = input.shape;
        long batch = shape[0], channels = shape[1];
        var normalized = Normalize(input, bank, prefix + ".norm");
        var q = Conv(normalized, bank, prefix + ".q", padding: 0).reshape(batch, channels, -1).permute(0, 2, 1);
        var k = Conv(normalized, bank, prefix + ".k", padding: 0).reshape(batch, channels, -1);
        var v = Conv(normalized, bank, prefix + ".v", padding: 0).reshape(batch, channels, -1);
        // Exact normal_attention/slice_attention order with one unsliced query block:
        // allocate like K, QK, scale, softmax over keys, then copy V times
        // transposed probabilities into that buffer, preserving K's strides.
        var attendedBuffer = zeros_like(k);
        var scores = bmm(q, k) * Math.Pow(channels, -0.5);
        var probabilities = scores.softmax(2).permute(0, 2, 1);
        attendedBuffer.copy_(bmm(v, probabilities));
        var attended = attendedBuffer.reshape(shape);
        DiagnosticObserver?.Invoke(prefix + ".proj_out.input", attended);
        var projected = Conv(attended, bank, prefix + ".proj_out", padding: 0);
        return (input + projected).MoveToOuterDisposeScope();
    }

    private static Tensor Downsample(Tensor input, ClassicalVaeWeightSet bank, string prefix)
    {
        using var scope = NewDisposeScope();
        var padded = nn.functional.pad(input, new long[] { 0, 1, 0, 1 }, PaddingModes.Constant, 0);
        return Conv(padded, bank, prefix, stride: 2, padding: 0).MoveToOuterDisposeScope();
    }

    private static Tensor Upsample(Tensor input, ClassicalVaeWeightSet bank, string prefix)
    {
        using var scope = NewDisposeScope();
        var expanded = nn.functional.interpolate(input, scale_factor: new double[] { 2, 2 }, mode: InterpolationMode.Nearest);
        return Conv(expanded, bank, prefix).MoveToOuterDisposeScope();
    }

    private static Tensor Normalize(Tensor input, ClassicalVaeWeightSet bank, string prefix) =>
        nn.functional.group_norm(input, ClassicalVaeConfig.NormalizationGroups,
            bank.GetTensor(prefix + ".weight"), bank.GetTensor(prefix + ".bias"), eps: 1e-6);

    private static Tensor Conv(Tensor input, ClassicalVaeWeightSet bank, string prefix, long stride = 1, long padding = 1) =>
        nn.functional.conv2d(input, bank.GetTensor(prefix + ".weight"), bank.GetTensor(prefix + ".bias"),
            strides: new[] { stride, stride }, padding: new[] { padding, padding });

    // Every graph stage returns a fresh owned wrapper; release the preceding stage
    // promptly instead of retaining an entire encoder/decoder activation history.
    private static Tensor Advance(Tensor previous, Tensor next)
    {
        previous.Dispose();
        return next;
    }

    internal static void ValidateTensor(Tensor tensor, string parameterName)
    {
        ArgumentNullException.ThrowIfNull(tensor, parameterName);
        if (tensor.IsInvalid || tensor.dtype != ScalarType.Float32 || !InferenceDevice.IsSupported(tensor.device_type) || tensor.is_sparse)
            throw new ArgumentException("VAE inputs must be live dense CPU or CUDA Float32 tensors.", parameterName);
    }

    private static void ValidateNchw(Tensor tensor, int channels, int minimumSpatialSize, string parameterName)
    {
        ValidateTensor(tensor, parameterName);
        var shape = tensor.shape;
        if (shape.Length != 4 || shape[0] <= 0 || shape[1] != channels || shape[2] < minimumSpatialSize || shape[3] < minimumSpatialSize)
            throw new ArgumentException($"VAE expects positive-batch NCHW, {channels} channels and spatial dimensions at least {minimumSpatialSize}.", parameterName);
    }

    private ClassicalVaeWeightSet RetainWeights()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(weights is null, this);
            return weights.Retain();
        }
    }

    public void Dispose()
    {
        ClassicalVaeWeightSet? released;
        lock (gate) { released = weights; weights = null; }
        released?.Dispose();
    }
}
