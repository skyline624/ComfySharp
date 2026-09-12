using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Owns optimizer state and exclusive gradient management for retained LoRA factors.
/// Callers serialize all model, factor and optimizer operations. A complete accumulation window
/// is required before Step. Cancellation discards only the unfinished window, not prior updates.</summary>
public sealed class LoraTrainingOptimizer : IDisposable
{
    private readonly List<TrainableLoraPatch> owners = [];
    private readonly Tensor[] parameters;
    private readonly List<State> states = [];
    private readonly string name;
    private readonly double learningRate;
    private sealed class State : IDisposable
    {
        internal Tensor? Mean, Square;
        internal long Step;
        public void Dispose() { Mean?.Dispose(); Square?.Dispose(); }
    }
    private bool disposed, faulted;
    public int AccumulationSteps { get; }
    public int PendingMicrobatches { get; private set; }
    public long CompletedSteps { get; private set; }

    public LoraTrainingOptimizer(IEnumerable<TrainableLoraPatch> patches, string name, double learningRate, int accumulationSteps = 1)
    {
        ArgumentNullException.ThrowIfNull(patches);
        if (name is not ("Adam" or "AdamW" or "SGD" or "RMSprop")) throw new ArgumentException("Unknown training optimizer.", nameof(name));
        if (!double.IsFinite(learningRate) || learningRate < 1e-7 || learningRate > 1) throw new ArgumentOutOfRangeException(nameof(learningRate));
        if (accumulationSteps < 1 || accumulationSteps > 1024) throw new ArgumentOutOfRangeException(nameof(accumulationSteps));
        AccumulationSteps = accumulationSteps; this.name = name; this.learningRate = learningRate;
        try
        {
            var selected = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
            foreach (var patch in patches)
            {
                ArgumentNullException.ThrowIfNull(patch);
                var owner = patch.Retain(); owners.Add(owner);
                if (!selected.Add(owner.Up)) throw new ArgumentException("An adapter may appear only once in an optimizer.", nameof(patches));
                InferenceDevice.RequireSame(owners[0].Up.device, owner.Up, nameof(patches));
            }
            if (owners.Count == 0) throw new ArgumentException("At least one adapter is required.", nameof(patches));
            parameters = owners.SelectMany(p => new[] { p.Up, p.Down }).ToArray();
            using var scope = NewDisposeScope();
            foreach (var parameter in parameters)
            {
                var state = new State(); states.Add(state);
                if (name is "Adam" or "AdamW") state.Mean = zeros_like(parameter).DetachFromDisposeScope();
                if (name != "SGD") state.Square = zeros_like(parameter).DetachFromDisposeScope();
            }
            ClearGradients();
        }
        catch { foreach (var state in states) state.Dispose(); foreach (var owner in owners) owner.Dispose(); throw; }
    }

    public void Accumulate(Tensor loss, CancellationToken cancellationToken = default)
    {
        ThrowIfUnavailable();
        if (PendingMicrobatches == AccumulationSteps) throw new InvalidOperationException("Step or reset the completed window before accumulating again.");
        try
        {
            cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(loss);
            using var scope = NewDisposeScope(); using var grad = set_grad_enabled(true);
            if (loss.numel() != 1 || loss.dtype != ScalarType.Float32 || !loss.requires_grad)
                throw new ArgumentException("Accumulate requires a differentiable Float32 scalar loss.", nameof(loss));
            InferenceDevice.RequireSame(parameters[0].device, loss, nameof(loss));
            if (!loss.isfinite().all().item<bool>()) throw new ArithmeticException("Nonfinite training loss.");
            using var normalized = loss / AccumulationSteps; normalized.backward();
            cancellationToken.ThrowIfCancellationRequested(); PendingMicrobatches++;
        }
        catch { ClearGradients(); throw; }
    }

    public void Step(CancellationToken cancellationToken = default)
    {
        ThrowIfUnavailable();
        bool updating = false;
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (PendingMicrobatches != AccumulationSteps) throw new IncompleteAccumulationException();
            using var scope = NewDisposeScope(); using var noGrad = no_grad();
            foreach (var parameter in parameters)
            {
                using var gradient = parameter.grad;
                if (gradient is not null && (!gradient.isfinite().all().item<bool>() || gradient.is_sparse))
                    throw new ArithmeticException("Nonfinite or sparse adapter gradient; no weights were updated.");
            }
            cancellationToken.ThrowIfCancellationRequested();
            updating = true;
            for (int i = 0; i < parameters.Length; i++)
            {
                using var iteration = NewDisposeScope();
                var parameter = parameters[i]; var state = states[i]; using var gradient = parameter.grad;
                if (gradient is null) continue;
                state.Step++;
                if (name == "SGD") { parameter.add_(gradient, alpha: -learningRate); continue; }
                if (name == "RMSprop")
                {
                    state.Square!.mul_(.99).addcmul_(gradient, gradient, value: .01);
                    parameter.addcdiv_(gradient, state.Square.sqrt().add_(1e-8), value: -learningRate);
                    continue;
                }
                // Frozen torch.optim defaults: AdamW decay=.01, Adam decay=0; beta=(.9,.999), epsilon=1e-8.
                if (name == "AdamW") parameter.mul_(1 - learningRate * .01);
                state.Mean!.mul_(.9).add_(gradient, alpha: .1);
                state.Square!.mul_(.999).addcmul_(gradient, gradient, value: .001);
                var denominator = (state.Square.sqrt() / Math.Sqrt(1 - Math.Pow(.999, state.Step))).add_(1e-8);
                parameter.addcdiv_(state.Mean, denominator, value: -learningRate / (1 - Math.Pow(.9, state.Step)));
            }
            foreach (var parameter in parameters)
                if (!parameter.isfinite().all().item<bool>()) throw new ArithmeticException("An optimizer update produced a nonfinite factor; discard this training session.");
            CompletedSteps++; ClearGradients();
        }
        catch (IncompleteAccumulationException) { throw new InvalidOperationException("A complete gradient accumulation window is required before Step."); }
        catch { faulted = updating; ClearGradients(); throw; }
    }

    public void ResetAccumulation() { ThrowIfUnavailable(); ClearGradients(); }
    private void ClearGradients()
    {
        // A null gradient skips unused parameters, including AdamW decay. Zero tensors would not.
        foreach (var parameter in parameters) parameter.grad = null;
        PendingMicrobatches = 0;
    }
    private void ThrowIfUnavailable()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (faulted) throw new InvalidOperationException("The optimizer failed during an update and must be disposed.");
    }
    private sealed class IncompleteAccumulationException : Exception;
    public void Dispose()
    {
        if (disposed) return; disposed = true;
        try { ClearGradients(); foreach (var state in states) state.Dispose(); }
        finally { foreach (var owner in owners) owner.Dispose(); }
    }
}
