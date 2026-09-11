using ComfySharp.Inference;
using Xunit;

namespace ComfySharp.Inference.Tests;

/// <summary>Metadata and admission contracts. These tests do not materialize a tensor.
/// A separate fresh process is required to establish native dependency isolation.</summary>
public sealed class Sd15CheckpointPlanTests
{
    private static SdCheckpointAssemblyPlan Inspect(SafeTensorFile file,
        Sd15UnclaimedTensorHandling handling = Sd15UnclaimedTensorHandling.Reject,
        CancellationToken cancellationToken = default) =>
        SdCheckpointAssemblyLoading.Inspect(file, SdCheckpointTestFile.ReducedProfile, handling, cancellationToken);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CombinedPlanAccountsForEverySelectedTensorWithoutCreatingAStockPlan(bool projection)
    {
        using var fixture = SdCheckpointTestFile.Create(projection, mixedDTypes: true);
        using var file = new SafeTensorFile(fixture.Path);
        var plan = Inspect(file);
        Assert.Same(file, plan.Source);
        Assert.Same(file, plan.Clip.Source);
        Assert.Same(file, plan.Unet.Plan.Source);
        Assert.Same(file, plan.Vae.Plan.Source);
        Assert.False(plan.Clip.RequireProjection);
        Assert.Equal(projection, plan.Clip.HasProjection);
        Assert.Equal(ClipCheckpointLayout.Sd1, plan.Clip.Layout);
        Assert.Equal(UnetCheckpointLayout.ModelDiffusionModel, plan.Unet.Layout);
        Assert.Equal(ClassicalVaeCheckpointLayout.FirstStageModel, plan.Vae.Layout);
        Assert.Equal(projection ? 971 : 970, file.Tensors.Count);
        Assert.Equal(file.Tensors.Values.Sum(t => t.End - t.Start), plan.SourceBytes);
        long expectedResident = fixture.Entries.Sum(t => checked(t.Shape.Aggregate(1L, (a, b) => checked(a * b)) * 4));
        Assert.Equal(expectedResident, plan.ResidentBytes);
        Assert.Equal(Math.Max(plan.Clip.TemporaryBytes, Math.Max(plan.Unet.TemporaryBytes, plan.Vae.TemporaryBytes)), plan.TemporaryBytes);
        Assert.Equal(checked(plan.ResidentBytes + plan.TemporaryBytes), plan.EstimatedPeakWeightBytes);
        Assert.Empty(plan.UnclaimedTensorNames);
        Assert.Throws<ArgumentException>(() => new Sd15CheckpointPlan(plan));
        Assert.Throws<InvalidDataException>(() => Sd15CheckpointLoader.Inspect(file));
    }

    [Fact]
    public void StockFacadeUsesExplicitConfigurationsAndKnownWeightBudgetWithoutAllocatingThem()
    {
        var profile = SdCheckpointAssemblyProfile.Sd15;
        Assert.Equal(ClipTextConfig.Large, profile.Clip);
        Assert.Equal(SdUnetConfig.Sd15, profile.Unet);
        Assert.Equal(ClassicalVaeConfig.Stock, profile.Vae);
        var clip = ClipWeightSchema.Describe(profile.Clip).Where(p => p.Key != ClipWeightSchema.Projection).ToArray();
        var unet = UnetWeightSchema.Describe(profile.Unet);
        var vae = ClassicalVaeWeightSchema.Describe(profile.Vae);
        Assert.Equal(196, clip.Length); Assert.Equal(686, unet.Count); Assert.Equal(248, vae.Count);
        long bytes = clip.Concat(unet).Concat(vae).Sum(p => checked(p.Value.Aggregate(1L, (a, b) => checked(a * b)) * 4));
        Assert.Equal(4_264_941_228L, bytes);
    }

    [Theory]
    [InlineData("cond_stage_model.")]
    [InlineData("model.diffusion_model.")]
    [InlineData("first_stage_model.")]
    public void MissingOrMalformedComponentCannotProduceAnAssemblyPlan(string prefix)
    {
        foreach (bool missing in new[] { true, false })
        {
            using var fixture = SdCheckpointTestFile.Create(edit: entries =>
            {
                int index = entries.FindIndex(e => e.Name.StartsWith(prefix, StringComparison.Ordinal));
                if (missing) entries.RemoveAt(index);
                else entries[index] = entries[index] with { Shape = [1] };
            });
            using var file = new SafeTensorFile(fixture.Path);
            Assert.Throws<InvalidDataException>(() => Inspect(file));
        }
    }

    [Fact]
    public void OutsideNamespaceExceptionReportsEveryOmissionAndNeverHidesAnActiveNamespaceError()
    {
        using var fixture = SdCheckpointTestFile.Create(edit: entries =>
        {
            entries.Add(new("model_ema.decay", []));
            entries.Add(new("unrelated.weight", [2]));
            entries.Add(new("cond_stage_model.logit_scale", []));
        });
        using var file = new SafeTensorFile(fixture.Path);
        Assert.Throws<InvalidDataException>(() => Inspect(file));
        var plan = Inspect(file, Sd15UnclaimedTensorHandling.ReportAndIgnoreOutsideComponents);
        Assert.Equal(new[] { "model_ema.decay", "unrelated.weight" }, plan.UnclaimedTensorNames);
        Assert.Equal(new[] { "cond_stage_model.logit_scale" }, plan.Clip.IgnoredEncoderKeys);
        Assert.Equal(file.Tensors.Values.Sum(t => t.End - t.Start) - 16, plan.SourceBytes);

        using var bad = SdCheckpointTestFile.Create(edit: entries => entries.Add(new("first_stage_model.unknown", [])));
        using var badFile = new SafeTensorFile(bad.Path);
        Assert.Throws<InvalidDataException>(() => Inspect(badFile, Sd15UnclaimedTensorHandling.ReportAndIgnoreOutsideComponents));
        Assert.Throws<ArgumentOutOfRangeException>(() => Inspect(file, (Sd15UnclaimedTensorHandling)99));
    }

    [Fact]
    public void OptionalProjectionStillRejectsWrongShapeAndDuplicateAliases()
    {
        using var fixture = SdCheckpointTestFile.Create(includeProjection: true, edit: entries =>
        {
            int index = entries.FindIndex(e => e.Name.EndsWith(ClipWeightSchema.Projection, StringComparison.Ordinal));
            entries[index] = entries[index] with { Shape = [1] };
        });
        using var file = new SafeTensorFile(fixture.Path);
        Assert.Throws<InvalidDataException>(() => Inspect(file));
        using var duplicate = SdCheckpointTestFile.Create(edit: entries =>
        {
            var existing = entries.First(e => e.Name == "first_stage_model.quant_conv.weight");
            entries.Add(existing with { Name = "first_stage_model.encoder.quant_conv.weight" });
        });
        using var duplicateFile = new SafeTensorFile(duplicate.Path);
        Assert.Throws<InvalidDataException>(() => Inspect(duplicateFile));
    }

    [Fact]
    public void ReaderIdentityBudgetAndCancellationAreRejectedBeforeAnyComponentLoad()
    {
        using var fixture = SdCheckpointTestFile.Create();
        using var file = new SafeTensorFile(fixture.Path);
        using var other = new SafeTensorFile(fixture.Path);
        var plan = Inspect(file);
        int calls = 0;
        SdCheckpointLoadObserver observer = (_, _, _) => calls++;
        Assert.Throws<ArgumentException>(() => SdCheckpointAssemblyLoading.Load(other, plan, long.MaxValue, observer: observer));
        Assert.Throws<ArgumentOutOfRangeException>(() => SdCheckpointAssemblyLoading.Load(file, plan, 0, observer: observer));
        Assert.Throws<InvalidOperationException>(() => SdCheckpointAssemblyLoading.Load(file, plan, plan.EstimatedPeakWeightBytes - 1, observer: observer));
        Assert.Throws<OperationCanceledException>(() => SdCheckpointAssemblyLoading.Load(file, plan, long.MaxValue, new(true), observer));
        Assert.Throws<OperationCanceledException>(() => Inspect(file, cancellationToken: new(true)));
        file.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Inspect(file));
        Assert.Throws<ObjectDisposedException>(() => SdCheckpointAssemblyLoading.Load(file, plan, long.MaxValue, observer: observer));
        Assert.Equal(0, calls);
        other.VerifyOpenSnapshotLength();
    }
}
