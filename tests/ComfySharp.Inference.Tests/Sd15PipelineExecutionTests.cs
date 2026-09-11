using ComfySharp.RuntimeProbe;
using ComfySharp.Tokenization;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

// Behavior tests use a real reduced combined checkpoint. Their sparse synthetic weights
// are separate from the prospective source corpus and cannot establish numerical parity.
[Collection("Classical VAE")]
public sealed class Sd15PipelineExecutionTests : IDisposable
{
    private readonly int previousThreads;

    public Sd15PipelineExecutionTests()
    {
        NativeRuntimeBootstrap.Initialize();
        previousThreads = get_num_threads();
        set_num_threads(1);
    }

    private sealed class Graphs : IDisposable
    {
        internal readonly List<Tensor> Weights = [];
        internal SdCheckpointComponents Components { get; }
        internal ComfyClipEncoder Clip { get; }
        internal SdUnet Unet { get; }
        internal ComfyImageVae Vae { get; }

        internal Graphs()
        {
            using var fixture = SdCheckpointTestFile.Create(includeProjection: true);
            using var file = new SafeTensorFile(fixture.Path);
            var plan = SdCheckpointAssemblyLoading.Inspect(file, SdCheckpointTestFile.ReducedProfile,
                Sd15UnclaimedTensorHandling.Reject);
            var seen = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
            Components = SdCheckpointAssemblyLoading.Load(file, plan, plan.EstimatedPeakWeightBytes,
                observer: (stage, _, borrowed) =>
                {
                    if (stage == SdCheckpointLoadStage.TensorMaterialized)
                        foreach (var tensor in borrowed!.Values) seen.Add(tensor);
                });
            Weights.AddRange(seen);
            try
            {
                Clip = Components.CreateClipEncoder();
                try
                {
                    Unet = Components.CreateUnet();
                    try { Vae = Components.CreateImageVae(); }
                    catch { Unet.Dispose(); throw; }
                }
                catch { Clip.Dispose(); throw; }
            }
            catch { Components.Dispose(); throw; }
        }

        public void Dispose()
        {
            try { Vae.Dispose(); }
            finally
            {
                try { Unet.Dispose(); }
                finally
                {
                    try { Clip.Dispose(); }
                    finally { Components.Dispose(); }
                }
            }
        }
    }

    [Fact]
    public void RealTextAndAllGraphsExecuteWithIndependentResultsAfterCallerDisposal()
    {
        using var callerGrad = set_grad_enabled(true);
        Sd15PipelineResult? result = null;
        try
        {
            float[][] outputValues;
            var boundaries = new List<Sd15PipelineBoundary>();
            var borrowed = new Dictionary<Sd15PipelineBoundary, Tensor>();
            var capturedShapes = new Dictionary<Sd15PipelineBoundary, long[]>();
            using (var scope = NewDisposeScope())
            using (var graphs = new Graphs())
            {
                var conditioning = Sd15PipelineExecution.Tokenize("a (cat:1.5)", "", 3.5, maximumDenoise: false);
                Assert.Equal(ClipProfile.Sd1L, conditioning.Positive.Profile);
                Assert.Equal(49406, conditioning.Negative.Chunks[0][0].Id);
                Assert.Equal(49407, conditioning.Negative.Chunks[0][1].Id);
                var noise = Noise().requires_grad_(true);
                var sigmas = tensor(new[] { 1.5f, 0.25f, 0f }, requires_grad: true);
                var noiseBefore = Values(noise);
                var sigmaBefore = Values(sigmas);
                var steps = new List<long>();
                result = Sd15PipelineExecution.Run(graphs.Clip, graphs.Unet, graphs.Vae, conditioning, noise, sigmas,
                    capture: (stage, value) =>
                    {
                        Assert.False(is_grad_enabled());
                        Assert.False(value.requires_grad);
                        boundaries.Add(stage);
                        borrowed.Add(stage, value);
                        capturedShapes.Add(stage, value.shape);
                        if (stage == Sd15PipelineBoundary.PositiveHidden)
                        {
                            // Run has retained all three graphs, including those not yet executed.
                            graphs.Dispose();
                            Assert.All(graphs.Weights, t => Assert.False(t.IsInvalid));
                        }
                    }, configureSampler: sampler => sampler.DiagnosticObserver = (step, _, _, _) => steps.Add(step));
                Assert.Equal(Enum.GetValues<Sd15PipelineBoundary>(), boundaries);
                Assert.Equal(new long[] { 0, 1 }, steps);
                Assert.Equal(new long[] { 1, 77, 16 }, capturedShapes[Sd15PipelineBoundary.NegativeHidden]);
                Assert.Equal(new long[] { 1, 4, 4, 5 }, result.DiffusionLatent.shape);
                Assert.Equal(new long[] { 1, 4, 4, 5 }, result.RawVaeLatent.shape);
                Assert.Equal(new long[] { 1, 32, 40, 3 }, result.Image.shape);
                Assert.True(result.Image.ge(0).all().item<bool>() && result.Image.le(1).all().item<bool>());
                Assert.All(new[] { result.DiffusionLatent, result.RawVaeLatent, result.Image }, t => Assert.True(t.isfinite().all().item<bool>()));
                Assert.All(graphs.Weights, t => Assert.True(t.IsInvalid));
                Assert.Equal(noiseBefore, Values(noise));
                Assert.Equal(sigmaBefore, Values(sigmas));
                Assert.True(is_grad_enabled());
                outputValues = new[] { Values(result.DiffusionLatent), Values(result.RawVaeLatent), Values(result.Image) };
                foreach (var stage in boundaries.Take(5)) Assert.True(borrowed[stage].IsInvalid);
            }
            using (result)
            {
                Assert.Equal(outputValues[0], Values(result.DiffusionLatent));
                Assert.Equal(outputValues[1], Values(result.RawVaeLatent));
                Assert.Equal(outputValues[2], Values(result.Image));
            }
            Assert.True(result.DiffusionLatent.IsInvalid);
            Assert.True(result.RawVaeLatent.IsInvalid);
            Assert.True(result.Image.IsInvalid);
        }
        finally { result?.Dispose(); }
    }

    [Theory]
    [InlineData((int)Sd15PipelineBoundary.PositiveHidden, false)]
    [InlineData((int)Sd15PipelineBoundary.PositiveHidden, true)]
    [InlineData((int)Sd15PipelineBoundary.InitialDiffusionLatent, false)]
    [InlineData((int)Sd15PipelineBoundary.InitialDiffusionLatent, true)]
    [InlineData((int)Sd15PipelineBoundary.FinalDiffusionLatent, false)]
    [InlineData((int)Sd15PipelineBoundary.FinalDiffusionLatent, true)]
    [InlineData((int)Sd15PipelineBoundary.RawVaeLatent, false)]
    [InlineData((int)Sd15PipelineBoundary.RawVaeLatent, true)]
    [InlineData((int)Sd15PipelineBoundary.Image, false)]
    [InlineData((int)Sd15PipelineBoundary.Image, true)]
    public void BoundaryFailureReleasesAllProducedTensorsAndPreservesCallerGraphs(int failAtValue, bool cancel)
    {
        var failAt = (Sd15PipelineBoundary)failAtValue;
        using var scope = NewDisposeScope();
        using var callerGrad = set_grad_enabled(true);
        using var graphs = new Graphs();
        using var cts = new CancellationTokenSource();
        var conditioning = Sd15PipelineExecution.Tokenize("", "", 1, maximumDenoise: false);
        var noise = Noise();
        var sigmas = tensor(new[] { 1.5f, 0f });
        var inputs = new[] { Values(noise), Values(sigmas) };
        var captured = new List<Tensor>();
        var injected = new InvalidDataException("Injected diagnostic writer failure");
        void Execute()
        {
            using var unexpected = Sd15PipelineExecution.Run(graphs.Clip, graphs.Unet, graphs.Vae,
                conditioning, noise, sigmas, cts.Token, (stage, value) =>
                {
                    captured.Add(value);
                    if (stage != failAt) return;
                    if (cancel) cts.Cancel();
                    else throw injected;
                });
        }
        if (cancel) Assert.Equal(cts.Token, Assert.Throws<OperationCanceledException>(Execute).CancellationToken);
        else Assert.Same(injected, Assert.Throws<InvalidDataException>(Execute));
        Assert.NotEmpty(captured);
        Assert.All(captured, t => Assert.True(t.IsInvalid));
        Assert.All(graphs.Weights, t => Assert.False(t.IsInvalid));
        using (var clip = graphs.Clip.Retain()) { }
        using (var unet = graphs.Unet.Retain()) { }
        using (var vae = graphs.Vae.Retain()) { }
        Assert.Equal(inputs[0], Values(noise));
        Assert.Equal(inputs[1], Values(sigmas));
        Assert.True(is_grad_enabled());
        graphs.Dispose();
        Assert.All(graphs.Weights, t => Assert.True(t.IsInvalid));
    }

    [Fact]
    public void RejectedInputOrCancelledCallCannotReachEncoderOrChangeTheInputs()
    {
        using var scope = NewDisposeScope();
        using var graphs = new Graphs();
        var conditioning = Sd15PipelineExecution.Tokenize("a", "", 1, false);
        var noise = Noise();
        var sigmas = tensor(new[] { 1.5f, 0f });
        var before = new[] { Values(noise), Values(sigmas) };
        int captures = 0;
        Assert.Throws<OperationCanceledException>(() => Sd15PipelineExecution.Run(graphs.Clip, graphs.Unet,
            graphs.Vae, conditioning, noise, sigmas, new(true), (_, _) => captures++));
        Assert.Throws<ArgumentException>(() => Sd15PipelineExecution.Run(graphs.Clip, graphs.Unet,
            graphs.Vae, conditioning, zeros(1, 4, 4, 4), sigmas, capture: (_, _) => captures++));
        Assert.Throws<ArgumentException>(() => Sd15PipelineExecution.Run(graphs.Clip, graphs.Unet,
            graphs.Vae, conditioning, noise, tensor(new[] { 1.5f, 0.5f }), capture: (_, _) => captures++));
        Assert.Equal(0, captures);
        Assert.Equal(before[0], Values(noise));
        Assert.Equal(before[1], Values(sigmas));
        Assert.All(graphs.Weights, t => Assert.False(t.IsInvalid));
    }

    private static Tensor Noise() => (arange(80, dtype: ScalarType.Float32) - 40).reshape(1, 4, 4, 5) * 0.015625;
    private static float[] Values(Tensor value) { using var copy = value.contiguous(); return copy.data<float>().ToArray(); }
    public void Dispose() => set_num_threads(previousThreads);
}
