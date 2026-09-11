using TorchSharp;

namespace ComfySharp.Inference;

/// <summary>Strict single-file SD1.5 component assembly on CPU/Float32.
/// Inspect validates all three components without initializing the native runtime.</summary>
public static class Sd15CheckpointLoader
{
    public static Sd15CheckpointPlan Inspect(SafeTensorFile file,
        Sd15CheckpointInspectionOptions? options = null, CancellationToken cancellationToken = default)
        => new(SdCheckpointAssemblyLoading.Inspect(file, SdCheckpointAssemblyProfile.Sd15,
            options?.UnclaimedTensors ?? Sd15UnclaimedTensorHandling.Reject, cancellationToken));

    /// <summary>Load all components from the exact reader used for inspection.
    /// The caller owns the reader and must keep it open and its file unchanged throughout this operation.
    /// The byte limit admits weights only and is not an available-RAM or inference-memory guarantee.</summary>
    public static Sd15Checkpoint Load(SafeTensorFile file, Sd15CheckpointPlan plan,
        long maxEstimatedPeakWeightBytes, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(plan);
        var components = SdCheckpointAssemblyLoading.Load(file, plan.Assembly,
            maxEstimatedPeakWeightBytes, cancellationToken);
        try { return new Sd15Checkpoint(components); }
        catch { components.Dispose(); throw; }
    }

    /// <summary>Open once, inspect all components, load through that handle, then close the reader.
    /// Returned components own copied weight storage and do not retain the file.</summary>
    public static Sd15Checkpoint Load(string path, long maxEstimatedPeakWeightBytes,
        Sd15CheckpointInspectionOptions? options = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (maxEstimatedPeakWeightBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxEstimatedPeakWeightBytes));
        using var file = new SafeTensorFile(path);
        var plan = Inspect(file, options, cancellationToken);
        return Load(file, plan, maxEstimatedPeakWeightBytes, cancellationToken);
    }
}

internal enum SdCheckpointComponent { Clip, Unet, Vae }
internal enum SdCheckpointLoadStage { BeforeComponentLoad, TensorMaterialized, ComponentLoaded, BeforeCommit }

// Synchronous diagnostic/test hook; tensors and dictionaries are borrowed, never transferred.
// Observers must not mutate/dispose them or use them asynchronously. Throwing aborts the transaction.
internal delegate void SdCheckpointLoadObserver(SdCheckpointLoadStage stage,
    SdCheckpointComponent? component, IReadOnlyDictionary<string, torch.Tensor>? tensors);

internal static class SdCheckpointAssemblyLoading
{
    internal static SdCheckpointAssemblyPlan Inspect(SafeTensorFile file,
        SdCheckpointAssemblyProfile profile, Sd15UnclaimedTensorHandling handling,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(profile);
        if (!Enum.IsDefined(handling)) throw new ArgumentOutOfRangeException(nameof(handling));
        cancellationToken.ThrowIfCancellationRequested();
        file.VerifyOpenSnapshotLength();
        profile.Validate();
        var clip = ClipCheckpointLoader.Inspect(file, profile.Clip, ClipCheckpointLayout.Sd1,
            requireProjection: false);
        cancellationToken.ThrowIfCancellationRequested();
        var unet = UnetCheckpointLoader.Inspect(file, profile.Unet, UnetCheckpointLayout.ModelDiffusionModel);
        cancellationToken.ThrowIfCancellationRequested();
        var vae = ClassicalVaeCheckpointLoader.Inspect(file, profile.Vae, ClassicalVaeCheckpointLayout.FirstStageModel);
        cancellationToken.ThrowIfCancellationRequested();

        // Within each component several canonical mappings may share a source tensor.
        // Across components any shared source name is an assembly error.
        var selected = new HashSet<string>(StringComparer.Ordinal);
        Claim(clip.Mappings.Select(m => m.SourceName));
        Claim(unet.Mappings.Select(m => m.SourceName));
        Claim(vae.Mappings.Select(m => m.SourceName));
        long sourceBytes = 0;
        foreach (string name in selected)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var info = file.Tensors[name];
            sourceBytes = checked(sourceBytes + checked(info.End - info.Start));
        }
        var claimed = new HashSet<string>(selected, StringComparer.Ordinal);
        claimed.UnionWith(clip.IgnoredEncoderKeys);
        var unclaimed = new List<string>();
        foreach (string name in file.Tensors.Keys.Order(StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (claimed.Contains(name)) continue;
            // Never let an explicit outside-namespace exception hide an active-namespace error.
            if (name.StartsWith("cond_stage_model.", StringComparison.Ordinal) ||
                name.StartsWith("model.diffusion_model.", StringComparison.Ordinal) ||
                name.StartsWith("first_stage_model.", StringComparison.Ordinal))
                throw new InvalidDataException($"Unclaimed tensor '{name}' inside a selected component namespace.");
            if (handling == Sd15UnclaimedTensorHandling.Reject)
                throw new InvalidDataException($"Tensor '{name}' is outside the selected SD1.5 component namespaces.");
            unclaimed.Add(name);
        }
        file.VerifyOpenSnapshotLength();
        cancellationToken.ThrowIfCancellationRequested();
        return new(file, profile, clip, unet, vae, handling, unclaimed, sourceBytes);

        void Claim(IEnumerable<string> names)
        {
            foreach (string name in names.Distinct(StringComparer.Ordinal))
                if (!selected.Add(name))
                    throw new InvalidDataException($"Tensor '{name}' is claimed by more than one component.");
        }
    }

    internal static SdCheckpointComponents Load(SafeTensorFile file, SdCheckpointAssemblyPlan plan,
        long maxEstimatedPeakWeightBytes, CancellationToken cancellationToken = default,
        SdCheckpointLoadObserver? observer = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(plan);
        if (maxEstimatedPeakWeightBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxEstimatedPeakWeightBytes));
        if (!ReferenceEquals(file, plan.Source) || !ReferenceEquals(file, plan.Clip.Source) ||
            !ReferenceEquals(file, plan.Unet.Plan.Source) || !ReferenceEquals(file, plan.Vae.Plan.Source))
            throw new ArgumentException("All component plans must belong to this exact reader.", nameof(plan));
        cancellationToken.ThrowIfCancellationRequested();
        file.VerifyOpenSnapshotLength();
        if (plan.EstimatedPeakWeightBytes > maxEstimatedPeakWeightBytes)
            throw new InvalidOperationException(
                $"Estimated peak weight bytes {plan.EstimatedPeakWeightBytes} exceed the supplied limit {maxEstimatedPeakWeightBytes}.");

        ClipWeightSet? clip = null;
        UnetWeightSet? unet = null;
        ClassicalVaeWeightSet? vae = null;
        try
        {
            Before(SdCheckpointComponent.Clip);
            clip = ClipCheckpointLoader.Load(file, plan.Clip, cancellationToken,
                Materialized(SdCheckpointComponent.Clip));
            After(SdCheckpointComponent.Clip);
            Before(SdCheckpointComponent.Unet);
            unet = UnetCheckpointLoader.Load(file, plan.Unet, cancellationToken,
                Materialized(SdCheckpointComponent.Unet));
            After(SdCheckpointComponent.Unet);
            Before(SdCheckpointComponent.Vae);
            vae = ClassicalVaeCheckpointLoader.Load(file, plan.Vae, cancellationToken,
                Materialized(SdCheckpointComponent.Vae));
            After(SdCheckpointComponent.Vae);
            observer?.Invoke(SdCheckpointLoadStage.BeforeCommit, null, null);
            cancellationToken.ThrowIfCancellationRequested();
            file.VerifyOpenSnapshotLength();
            var result = SdCheckpointComponents.FromOwnedBanks(plan.Profile, clip, unet, vae);
            clip = null;
            unet = null;
            vae = null;
            return result;
        }
        finally
        {
            try { vae?.Dispose(); }
            finally
            {
                try { unet?.Dispose(); }
                finally { clip?.Dispose(); }
            }
        }

        void Before(SdCheckpointComponent component)
        {
            cancellationToken.ThrowIfCancellationRequested();
            observer?.Invoke(SdCheckpointLoadStage.BeforeComponentLoad, component, null);
            cancellationToken.ThrowIfCancellationRequested();
            file.VerifyOpenSnapshotLength();
        }
        void After(SdCheckpointComponent component)
        {
            observer?.Invoke(SdCheckpointLoadStage.ComponentLoaded, component, null);
            cancellationToken.ThrowIfCancellationRequested();
        }
        Action<IReadOnlyDictionary<string, torch.Tensor>>? Materialized(SdCheckpointComponent component)
            => observer is null ? null : tensors =>
            {
                observer(SdCheckpointLoadStage.TensorMaterialized, component, tensors);
                cancellationToken.ThrowIfCancellationRequested();
            };
    }
}
