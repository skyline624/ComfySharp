using System.Buffers.Binary;
using ComfySharp.Tokenization;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

// All executions use the shared reduced core. They do not claim a stock checkpoint load.
[Collection("Classical VAE")]
public sealed class Sd15CheckpointAssemblyTests
{
    private sealed class Captures
    {
        internal readonly Dictionary<SdCheckpointComponent, Dictionary<string, Tensor>> Tensors =
            Enum.GetValues<SdCheckpointComponent>().ToDictionary(c => c,
                _ => new Dictionary<string, Tensor>(StringComparer.Ordinal));
        internal readonly List<SdCheckpointComponent> Started = [];
        internal readonly List<SdCheckpointComponent> Completed = [];

        internal void Observe(SdCheckpointLoadStage stage, SdCheckpointComponent? component,
            IReadOnlyDictionary<string, Tensor>? borrowed)
        {
            if (stage == SdCheckpointLoadStage.BeforeComponentLoad) Started.Add(component!.Value);
            if (stage == SdCheckpointLoadStage.ComponentLoaded) Completed.Add(component!.Value);
            if (stage != SdCheckpointLoadStage.TensorMaterialized) return;
            Assert.NotNull(borrowed);
            // Snapshot wrapper references now: the loader clears its dictionary at transfer.
            foreach (var (name, tensor) in borrowed)
                Tensors[component!.Value][name] = tensor;
        }

        internal IEnumerable<Tensor> All => Tensors.Values.SelectMany(d => d.Values);
        internal void AssertAlive() => Assert.All(All, t => Assert.False(t.IsInvalid));
        internal void AssertReleased()
        {
            Assert.NotEmpty(All);
            Assert.All(All, t => Assert.True(t.IsInvalid));
        }
    }

    private static SdCheckpointAssemblyPlan Inspect(SafeTensorFile file) =>
        SdCheckpointAssemblyLoading.Inspect(file, SdCheckpointTestFile.ReducedProfile,
            Sd15UnclaimedTensorHandling.Reject);

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public void CompleteLoadPreservesValuesAcrossReaderScopeAndOptionalProjection(bool projection, bool mixed)
    {
        using var fixture = SdCheckpointTestFile.Create(includeProjection: projection, mixedDTypes: mixed);
        var captured = new Captures();
        SdCheckpointComponents? components = null;
        NativeRuntimeBootstrap.Initialize();
        try
        {
            SdCheckpointAssemblyPlan plan;
            using (var scope = NewDisposeScope())
            using (var file = new SafeTensorFile(fixture.Path))
            {
                plan = Inspect(file);
                components = SdCheckpointAssemblyLoading.Load(file, plan, plan.EstimatedPeakWeightBytes,
                    observer: captured.Observe);
                file.VerifyOpenSnapshotLength(); // A borrowed reader is still open after success.
            }
            captured.AssertAlive();
            Assert.Equal(new[] { SdCheckpointComponent.Clip, SdCheckpointComponent.Unet, SdCheckpointComponent.Vae }, captured.Started);
            Assert.Equal(captured.Started, captured.Completed);
            Assert.Equal(plan.Clip.Mappings.Count + plan.Unet.Mappings.Count + plan.Vae.Mappings.Count, captured.All.Count());
            Assert.Equal(projection, components.HasClipProjection);
            var entries = fixture.Entries.ToDictionary(e => e.Name, StringComparer.Ordinal);
            CheckValues(SdCheckpointComponent.Clip, plan.Clip.Mappings.Select(m => (m.SourceName, m.CanonicalName)));
            CheckValues(SdCheckpointComponent.Unet, plan.Unet.Mappings.Select(m => (m.SourceName, m.CanonicalName)));
            CheckValues(SdCheckpointComponent.Vae, plan.Vae.Mappings.Select(m => (m.SourceName, m.CanonicalName)));

            // Exercise only the CLIP factory's profile/projection contract, not a composed SD forward.
            using var clip = components.CreateClipEncoder();
            Assert.Equal(ClipProfile.Sd1L, clip.Profile);
            IReadOnlyList<IReadOnlyList<ClipTokenWeight>> rows = [Enumerable.Range(0, 77)
                .Select(i => new ClipTokenWeight(i == 0 ? 49406 : i == 1 ? 7 : 49407, 1)).ToArray()];
            using var defaultResult = clip.Encode(rows);
            using var explicitlyUnprojected = clip.Encode(rows, new() { ProjectPooled = false });
            Assert.Equal(new long[] { 1, 77, 16 }, defaultResult.Hidden.shape);
            Assert.Equal(new long[] { 1, 16 }, defaultResult.Pooled.shape);
            Assert.Equal(explicitlyUnprojected.Pooled.data<float>().ToArray(), defaultResult.Pooled.data<float>().ToArray());
            if (projection)
            {
                using var projected = clip.Encode(rows, new() { ProjectPooled = true });
                using var expected = nn.functional.linear(defaultResult.Pooled,
                    captured.Tensors[SdCheckpointComponent.Clip][ClipWeightSchema.Projection]);
                Assert.Equal(expected.data<float>().ToArray(), projected.Pooled.data<float>().ToArray());
                Assert.False(defaultResult.Pooled.data<float>().ToArray().SequenceEqual(projected.Pooled.data<float>().ToArray()));
            }
            else
                Assert.Throws<InvalidOperationException>(() => clip.Encode(rows, new() { ProjectPooled = true }));

            void CheckValues(SdCheckpointComponent component, IEnumerable<(string SourceName, string CanonicalName)> mappings)
            {
                foreach (var (source, canonical) in mappings)
                {
                    var entry = entries[source];
                    var tensor = captured.Tensors[component][canonical];
                    Assert.Equal(ScalarType.Float32, tensor.dtype);
                    Assert.False(tensor.requires_grad);
                    Assert.Equal(entry.Shape, tensor.shape);
                    Assert.True(tensor.is_contiguous());
                    Assert.True(CpuModelWeightBank.IsAligned(tensor));
                    // Read bounded storage directly; never copy a full parameter tensor into managed memory.
                    var bytes = tensor.bytes;
                    int prefixCount = Math.Min(16, bytes.Length / 4);
                    for (int i = 0; i < prefixCount; i++)
                        Assert.Equal(SdCheckpointTestFile.ExpectedValue(entry, i),
                            BinaryPrimitives.ReadSingleLittleEndian(bytes.Slice(i * 4, 4)));
                    if (bytes.Length > 16 * 4)
                        Assert.Equal(0f, BinaryPrimitives.ReadSingleLittleEndian(bytes[^4..]));
                }
            }
        }
        finally { components?.Dispose(); }
        captured.AssertReleased();
    }

    [Theory]
    [InlineData("before-unet", false)]
    [InlineData("before-unet", true)]
    [InlineData("before-vae", false)]
    [InlineData("before-vae", true)]
    [InlineData("partial-unet", false)]
    [InlineData("partial-unet", true)]
    [InlineData("partial-vae", false)]
    [InlineData("partial-vae", true)]
    [InlineData("last-vae-tensor", false)]
    [InlineData("last-vae-tensor", true)]
    [InlineData("before-commit", false)]
    [InlineData("before-commit", true)]
    public void FailureAndCancellationRollBackCompletedAndPartialComponents(string point, bool cancel)
    {
        using var fixture = SdCheckpointTestFile.Create();
        using var file = new SafeTensorFile(fixture.Path);
        var plan = Inspect(file);
        var captured = new Captures();
        using var cancellation = new CancellationTokenSource();
        var injected = new InvalidOperationException("Assembly observer failure");
        bool reached = false;
        void Observer(SdCheckpointLoadStage stage, SdCheckpointComponent? component,
            IReadOnlyDictionary<string, Tensor>? tensors)
        {
            captured.Observe(stage, component, tensors);
            bool fail = point switch
            {
                "before-unet" => stage == SdCheckpointLoadStage.BeforeComponentLoad && component == SdCheckpointComponent.Unet,
                "before-vae" => stage == SdCheckpointLoadStage.BeforeComponentLoad && component == SdCheckpointComponent.Vae,
                "partial-unet" => stage == SdCheckpointLoadStage.TensorMaterialized && component == SdCheckpointComponent.Unet && tensors!.Count == 3,
                "partial-vae" => stage == SdCheckpointLoadStage.TensorMaterialized && component == SdCheckpointComponent.Vae && tensors!.Count == 3,
                "last-vae-tensor" => stage == SdCheckpointLoadStage.TensorMaterialized && component == SdCheckpointComponent.Vae && tensors!.Count == plan.Vae.Mappings.Count,
                "before-commit" => stage == SdCheckpointLoadStage.BeforeCommit,
                _ => false
            };
            if (!fail) return;
            reached = true;
            captured.AssertAlive();
            if (cancel) cancellation.Cancel();
            else throw injected;
        }
        void Load()
        {
            using var unexpectedSuccess = SdCheckpointAssemblyLoading.Load(file, plan,
                plan.EstimatedPeakWeightBytes, cancellation.Token, Observer);
        }
        if (cancel)
            Assert.Equal(cancellation.Token, Assert.Throws<OperationCanceledException>(Load).CancellationToken);
        else
            Assert.Same(injected, Assert.Throws<InvalidOperationException>(Load));
        Assert.True(reached);
        Assert.Equal(plan.Clip.Mappings.Count, captured.Tensors[SdCheckpointComponent.Clip].Count);
        Assert.Equal(point switch { "before-unet" => 0, "partial-unet" => 3, _ => plan.Unet.Mappings.Count },
            captured.Tensors[SdCheckpointComponent.Unet].Count);
        Assert.Equal(point switch { "partial-vae" => 3, "last-vae-tensor" or "before-commit" => plan.Vae.Mappings.Count, _ => 0 },
            captured.Tensors[SdCheckpointComponent.Vae].Count);
        // Same tensor count, different ownership boundary: VAE transfer has not completed in the former case.
        Assert.Equal(point == "before-commit", captured.Completed.Contains(SdCheckpointComponent.Vae));
        captured.AssertReleased();
        file.VerifyOpenSnapshotLength();
        Assert.Equal(plan.SourceBytes, Inspect(file).SourceBytes);
    }

    [Fact]
    public void RetainedAssemblyAndIndependentFactoriesReleaseOnlyTheirLastComponentOwner()
    {
        using var fixture = SdCheckpointTestFile.Create();
        using var file = new SafeTensorFile(fixture.Path);
        var plan = Inspect(file);
        var captured = new Captures();
        using var original = SdCheckpointAssemblyLoading.Load(file, plan, plan.EstimatedPeakWeightBytes,
            observer: captured.Observe);
        using var retained = original.Retain();
        original.Dispose();
        Assert.Throws<ObjectDisposedException>(() => original.Retain());
        Assert.Throws<ObjectDisposedException>(() => original.CreateClipEncoder());
        Assert.Throws<ObjectDisposedException>(() => original.CreateUnet());
        Assert.Throws<ObjectDisposedException>(() => original.CreateImageVae());
        captured.AssertAlive();
        using var clip = retained.CreateClipEncoder();
        using var unet = retained.CreateUnet();
        using var vae = retained.CreateImageVae();
        using var extraClip = clip.Retain();
        using var extraUnet = unet.Retain();
        using var extraVae = vae.Retain();
        Assert.Equal(ClipProfile.Sd1L, clip.Profile);
        Assert.Equal(SdCheckpointTestFile.ReducedProfile.Unet, unet.Config);
        Assert.Equal(SdCheckpointTestFile.ReducedProfile.Vae, vae.Config);
        retained.Dispose();
        clip.Dispose();
        unet.Dispose();
        vae.Dispose();
        captured.AssertAlive();
        // Each graph owns precisely its bank. Releasing CLIP must not release U-Net or VAE.
        extraClip.Dispose();
        Assert.All(captured.Tensors[SdCheckpointComponent.Clip].Values, t => Assert.True(t.IsInvalid));
        Assert.All(captured.Tensors[SdCheckpointComponent.Unet].Values, t => Assert.False(t.IsInvalid));
        Assert.All(captured.Tensors[SdCheckpointComponent.Vae].Values, t => Assert.False(t.IsInvalid));
        extraUnet.Dispose();
        Assert.All(captured.Tensors[SdCheckpointComponent.Unet].Values, t => Assert.True(t.IsInvalid));
        Assert.All(captured.Tensors[SdCheckpointComponent.Vae].Values, t => Assert.False(t.IsInvalid));
        extraVae.Dispose();
        captured.AssertReleased();
    }

    [Fact]
    public async Task BarrierRacesEitherReturnIndependentOwnersOrRejectDisposedAssembly()
    {
        using var fixture = SdCheckpointTestFile.Create();
        using var file = new SafeTensorFile(fixture.Path);
        var plan = Inspect(file);
        for (int operation = 0; operation < 4; operation++)
        {
            // Each race starts with the sole assembly owner. No keeper can mask an
            // acquisition that failed to retain its bank before the last Dispose.
            var captured = new Captures();
            using var raced = SdCheckpointAssemblyLoading.Load(file, plan, plan.EstimatedPeakWeightBytes,
                observer: captured.Observe);
            using var barrier = new Barrier(2);
            int selected = operation;
            var acquire = Task.Run<IDisposable?>(() =>
            {
                Assert.True(barrier.SignalAndWait(TimeSpan.FromSeconds(10)));
                try
                {
                    return selected switch
                    {
                        0 => raced.Retain(),
                        1 => raced.CreateClipEncoder(),
                        2 => raced.CreateUnet(),
                        _ => raced.CreateImageVae()
                    };
                }
                catch (ObjectDisposedException) { return null; }
            });
            var release = Task.Run(() =>
            {
                Assert.True(barrier.SignalAndWait(TimeSpan.FromSeconds(10)));
                raced.Dispose();
            });
            await Task.WhenAll(acquire, release);
            using var acquired = await acquire;
            if (acquired is not null)
            {
                foreach (var (component, tensors) in captured.Tensors)
                {
                    bool shouldBeAlive = selected switch
                    {
                        0 => true,
                        1 => component == SdCheckpointComponent.Clip,
                        2 => component == SdCheckpointComponent.Unet,
                        _ => component == SdCheckpointComponent.Vae
                    };
                    Assert.NotEmpty(tensors);
                    Assert.All(tensors.Values, t => Assert.Equal(!shouldBeAlive, t.IsInvalid));
                }
                using var further = acquired switch
                {
                    SdCheckpointComponents c => (IDisposable)c.Retain(),
                    ComfyClipEncoder c => c.Retain(),
                    SdUnet u => u.Retain(),
                    ComfyImageVae v => v.Retain(),
                    _ => throw new InvalidOperationException("Unexpected owner type")
                };
            }
            Assert.Throws<ObjectDisposedException>(() => raced.Retain());
            Assert.Throws<ObjectDisposedException>(() => raced.CreateClipEncoder());
            Assert.Throws<ObjectDisposedException>(() => raced.CreateUnet());
            Assert.Throws<ObjectDisposedException>(() => raced.CreateImageVae());
            // The temporary further Retain has left scope; now release the only
            // surviving result, or verify that a lost acquisition retained nothing.
            acquired?.Dispose();
            captured.AssertReleased();
        }
    }
}
