using ComfySharp.Tokenization;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>SDXL's independent branch weighting, L-before-G feature concatenation and G projected pooling.</summary>
public sealed class ComfySdxlEncoder : IDisposable
{
    private readonly object gate = new();
    private ComfyClipEncoder? l;
    private ComfyClipEncoder? g;

    public ComfySdxlEncoder(ClipTextEncoder l, ClipTextEncoder g)
    {
        ArgumentNullException.ThrowIfNull(l);
        ArgumentNullException.ThrowIfNull(g);
        if (!g.HasProjection) throw new ArgumentException("SDXL requires G text_projection.weight.", nameof(g));
        this.l = new(l, ClipProfile.SdXlL);
        try { this.g = new(g, ClipProfile.SdXlG); }
        catch { this.l.Dispose(); throw; }
    }

    public ClipConditioningResult Encode(ClipTokenization lTokens, ClipTokenization gTokens,
        ClipSdxlConditioningOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(lTokens);
        ArgumentNullException.ThrowIfNull(gTokens);
        if (lTokens.Profile != ClipProfile.SdXlL || gTokens.Profile != ClipProfile.SdXlG)
            throw new ArgumentException("SDXL composition requires L and G tokenizations in that order.");
        return Encode(lTokens.WithoutWordIds(), gTokens.WithoutWordIds(), options, cancellationToken);
    }

    public ClipConditioningResult Encode(IReadOnlyList<IReadOnlyList<ClipTokenWeight>> lTokens,
        IReadOnlyList<IReadOnlyList<ClipTokenWeight>> gTokens, ClipSdxlConditioningOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        options ??= new();
        if (options.L?.ReturnAttentionMasks == true || options.G?.ReturnAttentionMasks == true)
            throw new ArgumentException("The source SDXL composition cannot return branch attention-mask extras. Use standalone CLIP encoding to request those masks.", nameof(options));
        if (options.G?.ProjectPooled == false)
            throw new ArgumentException("SDXL composition always returns projected G pooling.", nameof(options));
        var lOptions = options.L ?? new();
        var gOptions = options.G ?? new();
        // Standard checkpoint exports omit unused L projection. Explicit requests still require it.
        lOptions = lOptions with { ProjectPooled = lOptions.ProjectPooled ?? false };
        gOptions = gOptions with { ProjectPooled = true };
        ComfyClipEncoder retainedL, retainedG;
        lock (gate)
        {
            retainedL = (l ?? throw new ObjectDisposedException(nameof(ComfySdxlEncoder))).Retain();
            try { retainedG = g!.Retain(); }
            catch { retainedL.Dispose(); throw; }
        }
        using (retainedL)
        using (retainedG)
        using (var gResult = retainedG.Encode(gTokens, gOptions, cancellationToken))
        using (var lResult = retainedL.Encode(lTokens, lOptions, cancellationToken))
        {
            var ls = lResult.Hidden.shape;
            var gs = gResult.Hidden.shape;
            if (ls.Length != gs.Length || Enumerable.Range(2, ls.Length - 3).Any(i => ls[i] != gs[i]))
                throw new ArgumentException("SDXL branch outputs must have matching ranks and nonsliced dimensions; all-layer outputs with different section lengths cannot be composed.");
            NativeRuntimeBootstrap.Initialize();
            using var scope = NewDisposeScope();
            using var inference = no_grad();
            // Frozen source slices axis 1 literally, including the layer axis for all-layer output.
            long cut = Math.Min(ls[1], gs[1]);
            var hidden = cat([lResult.Hidden.slice(1, 0, cut, 1), gResult.Hidden.slice(1, 0, cut, 1)], dim: -1);
            var pooled = gResult.Pooled.alias();
            cancellationToken.ThrowIfCancellationRequested();
            return new(hidden.DetachFromDisposeScope(), pooled.DetachFromDisposeScope());
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            l?.Dispose(); g?.Dispose();
            l = null; g = null;
        }
    }
}
