using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Mean Float32 losses used by the frozen TrainLoraNode. Preserves autograd.</summary>
public static class TrainingLoss
{
    public static Tensor Calculate(string name, Tensor prediction, Tensor target)
    {
        ArgumentNullException.ThrowIfNull(prediction); ArgumentNullException.ThrowIfNull(target);
        if (name is not ("MSE" or "L1" or "Huber" or "SmoothL1")) throw new ArgumentException("Unknown training loss.", nameof(name));
        if (!prediction.shape.SequenceEqual(target.shape) || prediction.numel() == 0)
            throw new ArgumentException("Training predictions and targets require the same nonempty shape.");
        foreach (var tensor in new[] { prediction, target })
            if (tensor.is_sparse || tensor.dtype is not (ScalarType.Float32 or ScalarType.Float16 or ScalarType.BFloat16) || !InferenceDevice.IsSupported(tensor.device_type))
                throw new ArgumentException("Training losses require dense floating CPU/CUDA inputs.");
        InferenceDevice.RequireSame(prediction.device, target, nameof(target));
        using var scope = NewDisposeScope();
        var input = prediction.to_type(ScalarType.Float32); var expected = target.to_type(ScalarType.Float32);
        Tensor loss = name switch
        {
            "MSE" => nn.functional.mse_loss(input, expected, reduction: nn.Reduction.Mean),
            "L1" => nn.functional.l1_loss(input, expected, reduction: nn.Reduction.Mean),
            "Huber" => nn.functional.huber_loss(input, expected, delta: 1, reduction: nn.Reduction.Mean),
            _ => nn.functional.smooth_l1_loss(input, expected, reduction: nn.Reduction.Mean, beta: 1)
        };
        return loss.MoveToOuterDisposeScope();
    }
}
