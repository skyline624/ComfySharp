using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>The plain SD1/SD2 four-level U-Net on CPU/CUDA Float32, with explicit basic attention.
/// Inputs are borrowed for the call; the returned raw prediction is independently owned.
/// Sigma scaling, prediction conversion and CFG belong to the denoiser wrapper.</summary>
public sealed class SdUnet : IDisposable
{
    private readonly object gate = new();
    private UnetWeightSet? weights;

    public SdUnet(UnetWeightSet weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        this.weights = weights.Retain();
        Config = weights.Config;
    }

    public SdUnetConfig Config { get; }
    public Device Device { get { using var bank = RetainWeights(); return bank.Device; } }
    public SdUnet To(Device device, CancellationToken cancellationToken = default)
    {
        using var bank = RetainWeights(); using var moved = bank.To(device, cancellationToken); return new SdUnet(moved);
    }

    public SdUnet WithLora(IReadOnlyDictionary<string,LoraWeightPatch> patches,
        long maxPatchedWeightBytes=512L*1024*1024,CancellationToken cancellationToken=default)
    {
        using var bank=RetainWeights();using var patched=bank.WithLora(patches,maxPatchedWeightBytes,cancellationToken);return new(patched);
    }

    // Diagnostic callbacks borrow live tensors synchronously. They must copy any data
    // they keep and must not dispose or mutate tensors. Null adds no tensor allocations.
    internal Action<string, Tensor>? DiagnosticObserver { get; set; }

    // Opt-in primitive capture for the last downsample and its two residual blocks.
    // The same borrowing rules apply; observers must not allocate native tensors.
    internal Action<string, Tensor>? FineDiagnosticObserver { get; set; }

    public SdUnet Retain()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(weights is null, this);
            return new SdUnet(weights);
        }
    }

    /// <summary>Runs a raw model prediction. Timesteps have length one or batch size;
    /// context is [batch, positive sequence length, ContextSize], without masks or broadcast.
    /// Cancellation is observed between operations; it does not interrupt a native kernel.</summary>
    public Tensor Forward(Tensor latentNchw, Tensor timesteps, Tensor context,
        CancellationToken cancellationToken = default)
        => ForwardCore(latentNchw, timesteps, context, null, 0, cancellationToken);

    /// <summary>Raw differentiable prediction with frozen base weights and caller-owned LoRA leaves.
    /// The returned autograd graph owns its saved native tensors until backward/graph disposal.
    /// This allowance limits patched weights, not activations. No gradient checkpointing or offload yet.</summary>
    public Tensor ForwardForTraining<TPatch>(Tensor latentNchw, Tensor timesteps, Tensor context,
        IReadOnlyDictionary<string, TPatch> patches, long maxPatchedWeightBytes = 512L * 1024 * 1024,
        CancellationToken cancellationToken = default) where TPatch : TrainableWeightPatch
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(patches);
        return ForwardCore(latentNchw, timesteps, context, TrainableWeightPatch.Widen(patches), maxPatchedWeightBytes, cancellationToken);
    }

    private Tensor ForwardCore(Tensor latentNchw, Tensor timesteps, Tensor context,
        IReadOnlyDictionary<string, TrainableWeightPatch>? patches, long maxPatchedWeightBytes, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var source = RetainWeights();
        // Bootstrap precedes even DisposeScope/no_grad creation on macOS ARM64.
        NativeRuntimeBootstrap.Initialize();
        ValidateInputs(latentNchw, timesteps, context);
        InferenceDevice.RequireSame(source.Device, latentNchw, nameof(latentNchw));
        InferenceDevice.RequireSame(source.Device, timesteps, nameof(timesteps));
        InferenceDevice.RequireSame(source.Device, context, nameof(context));
        using var scope = NewDisposeScope();
        using var gradMode = set_grad_enabled(patches is not null);
        using var operation = patches is null ? source.Retain() : source.WithTrainingLora(patches, maxPatchedWeightBytes, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        var observer = DiagnosticObserver;
        var fineObserver = FineDiagnosticObserver;

        // Borrowed managed buffers and offset views can select a different CPU linear
        // reduction path. Canonicalize only unaligned context storage, without changing
        // caller data. A scalar view lets the address check also accept strided context.
        using (var firstContextValue = context[0, 0, 0])
        {
            if (context.device_type == DeviceType.CPU && !CpuModelWeightBank.IsAligned(firstContextValue))
                context = context.is_contiguous() ? context.clone() : context.contiguous();
        }

        var embedding = TimeEmbedding(timesteps, operation);
        observer?.Invoke("timeEmbedding", embedding);
        var skips = new Stack<Tensor>(12);
        var current = Conv(latentNchw, operation, "input_blocks.0.0");
        skips.Push(current);
        int inputBlock = 1;
        for (int level = 0; level < SdUnetConfig.Levels; level++)
        {
            for (int block = 0; block < SdUnetConfig.ResidualBlocksPerLevel; block++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var blockScope = NewDisposeScope();
                string prefix = $"input_blocks.{inputBlock++}";
                current = Residual(current, embedding, operation, prefix + ".0", level == 3 ? fineObserver : null);
                if (level < 3)
                    current = SpatialTransformer(current, context, operation, prefix + ".1", cancellationToken);
                current.MoveToOuterDisposeScope();
                skips.Push(current);
                if (block == 1) observer?.Invoke($"down{level}", current);
            }
            if (level < 3)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (fineObserver is not null && level == 2)
                    fineObserver("input_blocks.9.0.op.input", current);
                current = Conv(current, operation, $"input_blocks.{inputBlock++}.0.op", stride: 2);
                if (fineObserver is not null && level == 2)
                    fineObserver("input_blocks.9.0.op.output", current);
                skips.Push(current);
            }
        }

        cancellationToken.ThrowIfCancellationRequested();
        using (var middleScope = NewDisposeScope())
        {
            current = Residual(current, embedding, operation, "middle_block.0");
            current = SpatialTransformer(current, context, operation, "middle_block.1", cancellationToken);
            current = Residual(current, embedding, operation, "middle_block.2").MoveToOuterDisposeScope();
            observer?.Invoke("middle", current);
        }

        for (int outputBlock = 0; outputBlock < 12; outputBlock++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var blockScope = NewDisposeScope();
            var skip = skips.Pop();
            var joined = cat(new[] { current, skip }, 1);
            // Both owners have been consumed. Remaining skips and the time embedding are the
            // only history kept across output blocks; a failing block is covered by the root scope.
            current.Dispose();
            skip.Dispose();
            string prefix = $"output_blocks.{outputBlock}";
            current = Residual(joined, embedding, operation, prefix + ".0");
            if (outputBlock >= 3)
                current = SpatialTransformer(current, context, operation, prefix + ".1", cancellationToken);
            if (outputBlock is 2 or 5 or 8)
            {
                var nextShape = skips.Peek().shape;
                // Source passes the next remaining skip shape, including odd/rectangular sizes.
                var enlarged = nn.functional.interpolate(current, size: new[] { nextShape[2], nextShape[3] },
                    mode: InterpolationMode.Nearest);
                string upsample = outputBlock == 2 ? ".1.conv" : ".2.conv";
                current = Conv(enlarged, operation, prefix + upsample);
                observer?.Invoke($"up{outputBlock / 3}", current);
            }
            current.MoveToOuterDisposeScope();
        }

        cancellationToken.ThrowIfCancellationRequested();
        var result = Conv(nn.functional.silu(GroupNorm(current, operation, "out.0", 1e-5)), operation, "out.2");
        cancellationToken.ThrowIfCancellationRequested();
        return result.DetachFromDisposeScope();
    }

    private Tensor TimeEmbedding(Tensor timesteps, UnetWeightSet bank)
    {
        using var scope = NewDisposeScope();
        int half = Config.BaseChannels / 2;
        // Preserve the source's Float32 native expression order: multiply, divide, exp;
        // timestep phases concatenate cosine before sine, with the default period 10000.
        var frequencies = (-Math.Log(10000) * arange(half, dtype: ScalarType.Float32, device: timesteps.device) / half).exp();
        var phases = timesteps.unsqueeze(1) * frequencies.unsqueeze(0);
        var sinusoidal = cat(new[] { phases.cos(), phases.sin() }, -1);
        var embedded = Linear(nn.functional.silu(Linear(sinusoidal, bank, "time_embed.0")), bank, "time_embed.2");
        return embedded.MoveToOuterDisposeScope();
    }

    private static Tensor Residual(Tensor input, Tensor embedding, UnetWeightSet bank,
        string prefix, Action<string, Tensor>? observe = null)
    {
        // All captures observe this same implementation, including the sums. Null
        // callbacks add no native tensors or stage strings; operation order is unchanged.
        using var scope = NewDisposeScope();
        observe?.Invoke(prefix + ".input", input);
        observe?.Invoke(prefix + ".embedding", embedding);
        observe?.Invoke(prefix + ".in_layers.0.input", input);
        var normalized = GroupNorm(input, bank, prefix + ".in_layers.0", 1e-5);
        observe?.Invoke(prefix + ".in_layers.0.output", normalized);
        observe?.Invoke(prefix + ".in_layers.1.input", normalized);
        var activated = nn.functional.silu(normalized);
        observe?.Invoke(prefix + ".in_layers.1.output", activated);
        observe?.Invoke(prefix + ".in_layers.2.input", activated);
        var hidden = Conv(activated, bank, prefix + ".in_layers.2");
        observe?.Invoke(prefix + ".in_layers.2.output", hidden);
        observe?.Invoke(prefix + ".emb_layers.0.input", embedding);
        var activatedTime = nn.functional.silu(embedding);
        observe?.Invoke(prefix + ".emb_layers.0.output", activatedTime);
        observe?.Invoke(prefix + ".emb_layers.1.input", activatedTime);
        var time = Linear(activatedTime, bank, prefix + ".emb_layers.1");
        observe?.Invoke(prefix + ".emb_layers.1.output", time);
        var projectedTime = time.unsqueeze(-1).unsqueeze(-1);
        hidden = hidden + projectedTime;
        observe?.Invoke(prefix + ".out_layers.0.input", hidden);
        normalized = GroupNorm(hidden, bank, prefix + ".out_layers.0", 1e-5);
        observe?.Invoke(prefix + ".out_layers.0.output", normalized);
        observe?.Invoke(prefix + ".out_layers.1.input", normalized);
        activated = nn.functional.silu(normalized);
        observe?.Invoke(prefix + ".out_layers.1.output", activated);
        observe?.Invoke(prefix + ".out_layers.3.input", activated);
        hidden = Conv(activated, bank, prefix + ".out_layers.3");
        observe?.Invoke(prefix + ".out_layers.3.output", hidden);
        var residual = input.shape[1] == hidden.shape[1] ? input : Conv(input, bank, prefix + ".skip_connection");
        var output = residual + hidden;
        observe?.Invoke(prefix + ".output", output);
        return output.MoveToOuterDisposeScope();
    }

    private Tensor SpatialTransformer(Tensor input, Tensor context, UnetWeightSet bank, string prefix,
        CancellationToken cancellationToken)
    {
        using var scope = NewDisposeScope();
        var shape = input.shape;
        long batch = shape[0], width = shape[1], height = shape[2], spatialWidth = shape[3];
        int heads = Config.HeadsAtWidth(checked((int)width));
        var hidden = GroupNorm(input, bank, prefix + ".norm", 1e-6);
        if (!Config.UseLinearProjection) hidden = Conv(hidden, bank, prefix + ".proj_in");
        hidden = hidden.permute(0, 2, 3, 1).flatten(1, 2).contiguous();
        if (Config.UseLinearProjection) hidden = Linear(hidden, bank, prefix + ".proj_in");
        string block = prefix + ".transformer_blocks.0";

        cancellationToken.ThrowIfCancellationRequested();
        using (var attentionScope = NewDisposeScope())
        {
            var normalized = LayerNorm(hidden, bank, block + ".norm1");
            var updated = Attention(normalized, normalized, heads, bank, block + ".attn1") + hidden;
            hidden.Dispose();
            hidden = updated.MoveToOuterDisposeScope();
        }
        cancellationToken.ThrowIfCancellationRequested();
        using (var attentionScope = NewDisposeScope())
        {
            var updated = Attention(LayerNorm(hidden, bank, block + ".norm2"), context, heads, bank, block + ".attn2") + hidden;
            hidden.Dispose();
            hidden = updated.MoveToOuterDisposeScope();
        }
        cancellationToken.ThrowIfCancellationRequested();
        using (var feedForwardScope = NewDisposeScope())
        {
            var projected = Linear(LayerNorm(hidden, bank, block + ".norm3"), bank, block + ".ff.net.0.proj");
            long gatedWidth = projected.shape[^1] / 2;
            var value = projected.narrow(-1, 0, gatedWidth);
            var gate = projected.narrow(-1, gatedWidth, gatedWidth);
            var feedForward = Linear(value * nn.functional.gelu(gate), bank, block + ".ff.net.2");
            var updated = hidden + feedForward;
            hidden.Dispose();
            hidden = updated.MoveToOuterDisposeScope();
        }

        if (Config.UseLinearProjection) hidden = Linear(hidden, bank, prefix + ".proj_out");
        hidden = hidden.reshape(batch, height, spatialWidth, width).permute(0, 3, 1, 2).contiguous();
        if (!Config.UseLinearProjection) hidden = Conv(hidden, bank, prefix + ".proj_out");
        return (hidden + input).MoveToOuterDisposeScope();
    }

    private static Tensor Attention(Tensor input, Tensor context, int heads, UnetWeightSet bank, string prefix)
    {
        using var scope = NewDisposeScope();
        long batch = input.shape[0], queryLength = input.shape[1];
        long width = input.shape[2], headWidth = width / heads;
        var query = AttentionProjection(input, bank.GetTensor(prefix + ".to_q.weight"), heads, headWidth);
        var key = AttentionProjection(context, bank.GetTensor(prefix + ".to_k.weight"), heads, headWidth);
        var value = AttentionProjection(context, bank.GetTensor(prefix + ".to_v.weight"), heads, headWidth);
        Tensor scores;
        using (var scoreScope = NewDisposeScope())
            scores = (einsum("b i d, b j d -> b i j", query, key) * Math.Pow(headWidth, -0.5)).MoveToOuterDisposeScope();
        query.Dispose();
        key.Dispose();
        var probabilities = scores.softmax(-1);
        scores.Dispose();
        // Noncausal self/cross attention: query and key lengths are independent and no
        // CLIP padding/causal mask, positional embedding or context batch broadcast is added.
        var mixed = einsum("b i j, b j d -> b i d", probabilities, value)
            .unsqueeze(0).reshape(batch, heads, queryLength, headWidth).permute(0, 2, 1, 3)
            .reshape(batch, queryLength, width);
        probabilities.Dispose();
        value.Dispose();
        return Linear(mixed, bank, prefix + ".to_out.0").MoveToOuterDisposeScope();
    }

    private static Tensor AttentionProjection(Tensor input, Tensor weight, int heads, long headWidth)
    {
        using var scope = NewDisposeScope();
        var shape = input.shape;
        // Source Q/K/V linears have no bias. Preserve that route; do not look up or invent biases.
        return nn.functional.linear(input, weight).reshape(shape[0], shape[1], heads, headWidth)
            .permute(0, 2, 1, 3).reshape(shape[0] * heads, shape[1], headWidth).contiguous().MoveToOuterDisposeScope();
    }

    private static Tensor GroupNorm(Tensor input, UnetWeightSet bank, string prefix, double epsilon)
    {
        // Python functional.group_norm validates this even in inference, while the C++
        // frontend calls torch::group_norm directly. Preserve the source's degenerate-case refusal.
        var shape = input.shape;
        if (shape[0] == 1 && shape[1] == SdUnetConfig.NormalizationGroups && shape[2] == 1 && shape[3] == 1)
            throw new ArgumentException("Source GroupNorm requires more than one value per group across batch/spatial dimensions.", nameof(input));
        return nn.functional.group_norm(input, SdUnetConfig.NormalizationGroups,
            bank.GetTensor(prefix + ".weight"), bank.GetTensor(prefix + ".bias"), eps: epsilon);
    }

    private static Tensor LayerNorm(Tensor input, UnetWeightSet bank, string prefix) =>
        nn.functional.layer_norm(input, new[] { input.shape[^1] },
            bank.GetTensor(prefix + ".weight"), bank.GetTensor(prefix + ".bias"), eps: 1e-5);

    private static Tensor Conv(Tensor input, UnetWeightSet bank, string prefix, long stride = 1)
    {
        var weight = bank.GetTensor(prefix + ".weight");
        long padding = weight.shape[2] == 3 ? 1 : 0;
        return nn.functional.conv2d(input, weight, bank.GetTensor(prefix + ".bias"),
            strides: new[] { stride, stride }, padding: new[] { padding, padding });
    }

    private static Tensor Linear(Tensor input, UnetWeightSet bank, string prefix)
    {
        var weight = bank.GetTensor(prefix + ".weight");
        var bias = bank.GetTensor(prefix + ".bias");
        var shape = input.shape;
        // Match Python aten::linear's fused biased path for contiguous 3D activations;
        // the LibTorch 2.10 C++ frontend otherwise separates matmul and bias addition.
        if (shape.Length == 3 && input.is_contiguous())
            return nn.functional.linear(input.reshape(shape[0] * shape[1], shape[2]), weight, bias)
                .reshape(shape[0], shape[1], weight.shape[0]);
        return nn.functional.linear(input, weight, bias);
    }

    private void ValidateInputs(Tensor latent, Tensor timesteps, Tensor context)
    {
        CheckTensor(latent, nameof(latent), 4);
        CheckTensor(timesteps, nameof(timesteps), 1);
        CheckTensor(context, nameof(context), 3);
        var shape = latent.shape;
        if (shape[0] <= 0 || shape[1] != SdUnetConfig.InputChannels || shape[2] <= 0 || shape[3] <= 0)
            throw new ArgumentException("U-Net latent must have shape [positive batch, 4, positive height, positive width].", nameof(latent));
        if (timesteps.shape[0] != 1 && timesteps.shape[0] != shape[0])
            throw new ArgumentException("U-Net timesteps must contain one value or one per latent batch row.", nameof(timesteps));
        var conditioning = context.shape;
        if (conditioning[0] != shape[0] || conditioning[1] <= 0 || conditioning[2] != Config.ContextSize)
            throw new ArgumentException("U-Net context must have matching batch, positive sequence length and the configured context width.", nameof(context));
    }

    private static void CheckTensor(Tensor tensor, string name, int rank)
    {
        ArgumentNullException.ThrowIfNull(tensor, name);
        ObjectDisposedException.ThrowIf(tensor.IsInvalid, tensor);
        if (tensor.dtype != ScalarType.Float32 || !InferenceDevice.IsSupported(tensor.device_type) || tensor.is_sparse)
            throw new ArgumentException("U-Net inputs must be dense CPU or CUDA Float32 tensors.", name);
        if (tensor.shape.Length != rank)
            throw new ArgumentException($"U-Net input must have rank {rank}.", name);
    }

    private UnetWeightSet RetainWeights()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(weights is null, this);
            return weights.Retain();
        }
    }

    public void Dispose()
    {
        UnetWeightSet? released;
        lock (gate) { released = weights; weights = null; }
        released?.Dispose();
    }
}
