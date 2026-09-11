using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Frozen CLIP text inference on CPU/F32 using the source basic attention path.</summary>
public sealed class ClipTextEncoder : IDisposable
{
    private readonly object gate = new();
    private ClipWeightSet? weights;

    public ClipTextEncoder(ClipWeightSet weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        this.weights = weights.Retain();
        Config = weights.Config;
        HasProjection = weights.HasProjection;
    }

    public ClipTextConfig Config { get; }
    public bool HasProjection { get; }

    public ClipTextEncoder Retain()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(weights is null, this);
            return new ClipTextEncoder(weights);
        }
    }

    public ClipForwardResult Forward(IReadOnlyList<IReadOnlyList<int>> tokens,
        ClipForwardOptions? options = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var operation = RetainWeights();
        options ??= new();
        var input = ValidateAndCopy(tokens, options);
        if (options.ProjectPooled && !HasProjection)
            throw new InvalidOperationException("Projected CLIP pooling requires text_projection.weight in the selected weight bank.");

        // Bootstrap must precede even no_grad / dispose scope creation on macOS ARM64.
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        cancellationToken.ThrowIfCancellationRequested();
        int batch = input.PoolIndices.Length;
        const int length = ClipTextConfig.MaxPositions;
        var ids = tensor(input.Ids, dtype: ScalarType.Int64, device: CPU);
        var x = operation.GetTensor("text_model.embeddings.token_embedding.weight").index_select(0, ids)
            .reshape(batch, length, Config.HiddenSize)
            + operation.GetTensor("text_model.embeddings.position_embedding.weight");
        var mask = full(new long[] { length, length }, -float.MaxValue, dtype: ScalarType.Float32, device: CPU).triu_(1);
        if (input.Mask is not null)
        {
            var padding = tensor(input.Mask, dtype: ScalarType.Float32, device: CPU).reshape(batch, 1, 1, length)
                .expand(batch, 1, length, length);
            mask = padding + mask;
        }
        var attentionMask = mask.reshape(input.Mask is null ? 1 : batch, -1, length, length)
            .expand(batch, Config.HeadCount, length, length).reshape(-1, length, length);
        Tensor? intermediate = null;
        var all = options.AllIntermediateLayers ? new List<Tensor>(Config.LayerCount) : null;
        int? selected = options.IntermediateLayer is < 0 ? Config.LayerCount + options.IntermediateLayer : options.IntermediateLayer;
        for (int layer = 0; layer < Config.LayerCount; layer++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var layerScope = NewDisposeScope();
            string prefix = $"text_model.encoder.layers.{layer}";
            var normalized = Normalize(x, operation, prefix + ".layer_norm1");
            x.add_(Attention(normalized, attentionMask, operation, prefix + ".self_attn", batch));
            normalized = Normalize(x, operation, prefix + ".layer_norm2");
            var hidden = Linear(normalized, operation, prefix + ".mlp.fc1");
            hidden = Config.Activation switch
            {
                ClipActivation.QuickGelu => hidden * sigmoid(1.702 * hidden),
                ClipActivation.Gelu => nn.functional.gelu(hidden),
                ClipActivation.GeluTanh => nn.functional.gelu(hidden, approximate: TorchSharp.Modules.GELU.Approximate.tanh),
                _ => throw new InvalidOperationException("Unsupported CLIP activation.")
            };
            x.add_(Linear(hidden, operation, prefix + ".mlp.fc2"));
            if (selected == layer) intermediate = x.clone().MoveToOuterDisposeScope();
            if (all is not null) all.Add(x.unsqueeze(1).clone().MoveToOuterDisposeScope());
        }

        if (all is not null) intermediate = cat(all.ToArray(), 1);
        var final = Normalize(x, operation, "text_model.final_layer_norm");
        if (intermediate is not null && options.NormalizeIntermediate)
            intermediate = Normalize(intermediate, operation, "text_model.final_layer_norm");
        var poolIndices = tensor(input.PoolIndices, dtype: ScalarType.Int64, device: CPU);
        var pooled = final.reshape(batch * length, Config.HiddenSize).index_select(0, poolIndices);
        var projected = options.ProjectPooled
            ? nn.functional.linear(pooled, operation.GetTensor("text_projection.weight")) : null;
        cancellationToken.ThrowIfCancellationRequested();

        // Results have independent ownership, including when the caller has an ambient scope.
        return new ClipForwardResult(final.DetachFromDisposeScope(), intermediate?.DetachFromDisposeScope(),
            projected?.DetachFromDisposeScope(), pooled.DetachFromDisposeScope());
    }

    private Tensor Normalize(Tensor input, ClipWeightSet bank, string prefix) =>
        nn.functional.layer_norm(input, new long[] { Config.HiddenSize },
            bank.GetTensor(prefix + ".weight"), bank.GetTensor(prefix + ".bias"), eps: 1e-5);

    private static Tensor Linear(Tensor input, ClipWeightSet bank, string prefix)
    {
        var weight = bank.GetTensor(prefix + ".weight");
        var bias = bank.GetTensor(prefix + ".bias");
        var shape = input.shape;
        // Python aten::linear fuses bias into addmm for contiguous 3D inputs. TorchSharp's
        // LibTorch 2.10 C++ functional wrapper only does that for 2D, so flatten explicitly
        // to preserve the source's rounding order instead of using matmul then adding bias.
        if (shape.Length == 3 && input.is_contiguous())
            return nn.functional.linear(input.reshape(shape[0] * shape[1], shape[2]), weight, bias)
                .reshape(shape[0], shape[1], weight.shape[0]);
        return nn.functional.linear(input, weight, bias);
    }

    private Tensor Attention(Tensor input, Tensor mask, ClipWeightSet bank, string prefix, int batch)
    {
        int heads = Config.HeadCount;
        int headSize = Config.HiddenSize / heads;
        const int length = ClipTextConfig.MaxPositions;
        Tensor Heads(Tensor value) => value.reshape(batch, length, heads, headSize)
            .permute(0, 2, 1, 3).reshape(batch * heads, length, headSize).contiguous();
        var q = Heads(Linear(input, bank, prefix + ".q_proj"));
        var k = Heads(Linear(input, bank, prefix + ".k_proj"));
        var v = Heads(Linear(input, bank, prefix + ".v_proj"));
        // Preserve attention_basic: flatten B/H, dot first, scale second, then additive mask.
        var scores = einsum("b i d, b j d -> b i j", q, k) * Math.Pow(headSize, -0.5);
        scores.add_(mask);
        var probabilities = scores.softmax(-1);
        var output = einsum("b i j, b j d -> b i d", probabilities, v)
            .unsqueeze(0).reshape(batch, heads, length, headSize).permute(0, 2, 1, 3)
            .reshape(batch, length, Config.HiddenSize);
        return Linear(output, bank, prefix + ".out_proj");
    }

    private (long[] Ids, float[]? Mask, long[] PoolIndices) ValidateAndCopy(
        IReadOnlyList<IReadOnlyList<int>> tokens, ClipForwardOptions options)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Count == 0) throw new ArgumentException("CLIP requires at least one 77-token row.", nameof(tokens));
        if (options.AllIntermediateLayers && options.IntermediateLayer is not null)
            throw new ArgumentException("Select either all intermediate layers or one scalar layer.", nameof(options));
        if (options.IntermediateLayer is int layer && (layer < -Config.LayerCount || layer >= Config.LayerCount))
            throw new ArgumentOutOfRangeException(nameof(options), "Intermediate CLIP layer must be in [-LayerCount, LayerCount-1].");
        int batch = tokens.Count;
        const int length = ClipTextConfig.MaxPositions;
        if (options.AttentionMask is not null && options.AttentionMask.Count != batch)
            throw new ArgumentException("CLIP attention mask must have the same batch size as tokens.", nameof(options));
        if (options.TokenCounts is not null && options.TokenCounts.Count != batch)
            throw new ArgumentException("CLIP token counts must have the same batch size as tokens.", nameof(options));
        var ids = new long[checked(batch * length)];
        var mask = options.AttentionMask is null ? null : new float[ids.Length];
        var pools = new long[batch];
        for (int row = 0; row < batch; row++)
        {
            var tokenRow = tokens[row];
            if (tokenRow is null || tokenRow.Count != length)
                throw new ArgumentException("CLIP requires rectangular rows of exactly 77 token IDs; no truncation or position interpolation is performed.", nameof(tokens));
            var maskRow = options.AttentionMask?[row];
            if (mask is not null && (maskRow is null || maskRow.Count != length))
                throw new ArgumentException("CLIP attention mask rows must have exactly 77 binary entries.", nameof(options));
            int firstEos = -1;
            for (int position = 0; position < length; position++)
            {
                int id = tokenRow[position];
                if (id < 0 || id >= ClipTextConfig.VocabularySize)
                    throw new ArgumentOutOfRangeException(nameof(tokens), $"Token ID at row {row}, position {position} is outside [0, 49407].");
                ids[row * length + position] = id;
                if (id == 49407 && firstEos < 0) firstEos = position;
                if (maskRow is not null)
                {
                    int value = maskRow[position];
                    if (value is not (0 or 1)) throw new ArgumentException("CLIP attention masks must contain only zero or one.", nameof(options));
                    mask![row * length + position] = value == 0 ? -float.MaxValue : 0;
                }
            }
            int pool = Math.Max(firstEos, 0);
            if (options.TokenCounts is not null)
            {
                int count = options.TokenCounts[row];
                if (count < 0 || count > length) throw new ArgumentOutOfRangeException(nameof(options), "CLIP token counts must be in [0, 77].");
                // The source gathers count-1 without compensating for left padding; zero selects -1.
                pool = count == 0 ? length - 1 : count - 1;
            }
            pools[row] = (long)row * length + pool;
        }
        return (ids, mask, pools);
    }

    private ClipWeightSet RetainWeights()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(weights is null, this);
            return weights.Retain();
        }
    }

    public void Dispose()
    {
        ClipWeightSet? released;
        lock (gate) { released = weights; weights = null; }
        released?.Dispose();
    }
}
