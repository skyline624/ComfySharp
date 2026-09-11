using ComfySharp.Tokenization;

namespace ComfySharp.Inference;

/// <summary>Retained owners of the three fixed SD1.5 checkpoint components.
/// Provides independent graph factories, without a composed forward, schedule or latent scaling policy.</summary>
public sealed class Sd15Checkpoint : IDisposable
{
    private readonly SdCheckpointComponents components;
    public ClipTextConfig ClipConfig => ClipTextConfig.Large;
    public SdUnetConfig UnetConfig => SdUnetConfig.Sd15;
    public ClassicalVaeConfig VaeConfig => ClassicalVaeConfig.Stock;
    public bool HasClipProjection => components.HasClipProjection;

    // Ownership transfers only on successful construction.
    internal Sd15Checkpoint(SdCheckpointComponents components)
    {
        ArgumentNullException.ThrowIfNull(components);
        if (components.Profile != SdCheckpointAssemblyProfile.Sd15)
            throw new ArgumentException("The public SD1.5 owner requires the fixed stock configurations.", nameof(components));
        this.components = components;
    }

    public Sd15Checkpoint Retain()
    {
        var retained = components.Retain();
        try { return new Sd15Checkpoint(retained); }
        catch { retained.Dispose(); throw; }
    }

    /// <summary>Return an independently disposable SD1-L encoder. Its default pooling is unprojected.</summary>
    public ComfyClipEncoder CreateClipEncoder() => components.CreateClipEncoder();
    /// <summary>Return an independently disposable U-Net graph. Prediction kind remains a caller decision.</summary>
    public SdUnet CreateUnet() => components.CreateUnet();
    /// <summary>Return an independently disposable image VAE wrapper using raw, unscaled latents.</summary>
    public ComfyImageVae CreateImageVae() => components.CreateImageVae();
    public void Dispose() => components.Dispose();
}

// Both the stock facade and reduced tests exercise this owner and these exact factories.
// A returned graph holds its own bank reference and survives disposal of every assembly owner.
internal sealed class SdCheckpointComponents : IDisposable
{
    private sealed class Shared
    {
        internal readonly object Gate = new();
        internal readonly SdCheckpointAssemblyProfile Profile;
        internal readonly ClipWeightSet Clip;
        internal readonly UnetWeightSet Unet;
        internal readonly ClassicalVaeWeightSet Vae;
        internal readonly bool HasClipProjection;
        internal int Owners = 1;

        internal Shared(SdCheckpointAssemblyProfile profile, ClipWeightSet clip,
            UnetWeightSet unet, ClassicalVaeWeightSet vae)
        {
            Profile = profile;
            Clip = clip;
            Unet = unet;
            Vae = vae;
            HasClipProjection = clip.HasProjection;
        }
    }

    private readonly Shared shared;
    private bool disposed;
    internal SdCheckpointAssemblyProfile Profile => shared.Profile;
    internal bool HasClipProjection => shared.HasClipProjection;

    private SdCheckpointComponents(Shared shared) => this.shared = shared;

    // Caller keeps all ownership on failure. Success transfers exactly these three bank owners.
    // No bank is copied or retained here, so transfer adds no full-bank memory peak.
    internal static SdCheckpointComponents FromOwnedBanks(SdCheckpointAssemblyProfile profile,
        ClipWeightSet clip, UnetWeightSet unet, ClassicalVaeWeightSet vae)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(clip);
        ArgumentNullException.ThrowIfNull(unet);
        ArgumentNullException.ThrowIfNull(vae);
        profile.Validate();
        if (clip.Config != profile.Clip || unet.Config != profile.Unet || vae.Config != profile.Vae)
            throw new ArgumentException("Owned component configurations must match the assembly profile.");
        return new(new Shared(profile, clip, unet, vae));
    }

    internal SdCheckpointComponents Retain()
    {
        lock (shared.Gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            // Allocate the new owner before incrementing, preserving the count if allocation fails.
            var retained = new SdCheckpointComponents(shared);
            shared.Owners = checked(shared.Owners + 1);
            return retained;
        }
    }

    internal ComfyClipEncoder CreateClipEncoder()
    {
        lock (shared.Gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            using var graph = new ClipTextEncoder(shared.Clip);
            return new ComfyClipEncoder(graph, ClipProfile.Sd1L);
        }
    }

    internal SdUnet CreateUnet()
    {
        lock (shared.Gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return new SdUnet(shared.Unet);
        }
    }

    internal ComfyImageVae CreateImageVae()
    {
        lock (shared.Gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            using var graph = new ClassicalVae(shared.Vae);
            return new ComfyImageVae(graph);
        }
    }

    public void Dispose()
    {
        bool release;
        lock (shared.Gate)
        {
            if (disposed) return;
            disposed = true;
            shared.Owners--;
            release = shared.Owners == 0;
        }
        if (!release) return;
        try { shared.Vae.Dispose(); }
        finally
        {
            try { shared.Unet.Dispose(); }
            finally { shared.Clip.Dispose(); }
        }
    }
}
