using static TorchSharp.torch;
namespace ComfySharp.Inference;

/// <summary>Float32 LoRA/LoCon/DoRA weight arithmetic. Inputs are borrowed and never mutated.
/// Ambient autograd is preserved so the same expressions can serve adapter optimization.</summary>
public static class LoraMath
{
    public static Tensor Apply(Tensor weight, Tensor up, Tensor down, double strength = 1, double? alpha = null,
        Tensor? mid = null, Tensor? doraScale = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        Validate(weight, nameof(weight)); Validate(up, nameof(up)); Validate(down, nameof(down));
        if (!double.IsFinite(strength) || alpha is not null && !double.IsFinite(alpha.Value))
            throw new ArgumentOutOfRangeException(nameof(strength), "Strength and alpha must be finite.");
        InferenceDevice.RequireSame(weight.device,up,nameof(up)); InferenceDevice.RequireSame(weight.device,down,nameof(down));
        double scale = alpha is null ? 1.0 : alpha.Value / down.shape[0];
        var right = down;
        if (mid is not null)
        {
            Validate(mid,nameof(mid)); InferenceDevice.RequireSame(weight.device,mid,nameof(mid));
            if (mid.dim()!=4) throw new ArgumentException("LoCon mid must have four dimensions.",nameof(mid));
            right = mm(down.transpose(0,1).flatten(1),mid.transpose(0,1).flatten(1))
                .reshape(down.shape[1],down.shape[0],mid.shape[2],mid.shape[3]).transpose(0,1);
        }
        cancellationToken.ThrowIfCancellationRequested();
        var leftMatrix = up.flatten(1); var rightMatrix = right.flatten(1);
        if (leftMatrix.shape[1]!=rightMatrix.shape[0] || checked(leftMatrix.shape[0]*rightMatrix.shape[1])!=weight.numel())
            throw new ArgumentException("LoRA factors do not match the destination weight shape.");
        var diff = mm(leftMatrix,rightMatrix).reshape(weight.shape);
        Tensor result;
        if (doraScale is null) result = weight + (strength * scale) * diff;
        else
        {
            ArgumentNullException.ThrowIfNull(doraScale);
            if (doraScale.dtype!=ScalarType.Float32 || doraScale.is_sparse || doraScale.dim()<1 || doraScale.shape.Any(v=>v<=0))
                throw new ArgumentException("DoRA scale must be a dense nonempty Float32 tensor.",nameof(doraScale));
            InferenceDevice.RequireSame(weight.device,doraScale,nameof(doraScale));
            var calculated = weight + diff * scale;
            Tensor norm;
            // Preserve the frozen source's asymmetry: output-axis norm uses ORIGINAL weight.
            if(doraScale.shape[0]==weight.shape[0])
                norm=weight.reshape(weight.shape[0],-1).norm(1,true,2).reshape(new[]{weight.shape[0]}.Concat(Enumerable.Repeat(1L,checked((int)weight.dim()-1))).ToArray());
            else
                norm=calculated.transpose(0,1).reshape(weight.shape[1],-1).norm(1,true,2)
                    .reshape(new[]{weight.shape[1]}.Concat(Enumerable.Repeat(1L,checked((int)weight.dim()-1))).ToArray()).transpose(0,1);
            var normalized=calculated * (doraScale / (norm + 1.1920928955078125e-7));
            if(!normalized.shape.SequenceEqual(weight.shape)) throw new ArgumentException("DoRA scale broadcasts beyond the destination weight shape.",nameof(doraScale));
            result=strength==1.0 ? normalized : weight + strength * (normalized-weight);
        }
        cancellationToken.ThrowIfCancellationRequested();
        if (!result.isfinite().all().item<bool>()) throw new ArithmeticException("LoRA produced nonfinite weights.");
        return result.DetachFromDisposeScope();
    }

    private static void Validate(Tensor tensor,string name)
    {
        ArgumentNullException.ThrowIfNull(tensor,name);
        if(!InferenceDevice.IsSupported(tensor.device_type)||tensor.dtype!=ScalarType.Float32||tensor.is_sparse||tensor.dim()<2||tensor.shape.Any(v=>v<=0))
            throw new ArgumentException("LoRA weight/factors require dense nonempty CPU/CUDA Float32 tensors of rank at least two.",name);
    }
}
