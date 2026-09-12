using static TorchSharp.torch;
namespace ComfySharp.Inference;

/// <summary>An immutable, independently disposable snapshot of LoRA factors for inference baking.
/// Low-level trainable operations use LoraMath directly; this owner intentionally freezes its factors.</summary>
public sealed class LoraWeightPatch : IDisposable
{
    private readonly object gate=new();
    private Tensor? up,down,mid,dora;
    public double Strength { get; }
    public double? Alpha { get; }
    public LoraWeightPatch(Tensor up,Tensor down,double strength=1,double? alpha=null,Tensor? mid=null,Tensor? doraScale=null)
    {
        NativeRuntimeBootstrap.Initialize(); using var scope=NewDisposeScope(); using var noGrad=no_grad();
        if(!double.IsFinite(strength)||alpha is not null&&!double.IsFinite(alpha.Value))throw new ArgumentOutOfRangeException(nameof(strength));
        var u=Copy(up);var d=Copy(down);var m=mid is null?null:Copy(mid);var s=doraScale is null?null:Copy(doraScale);
        this.up=u.DetachFromDisposeScope();this.down=d.DetachFromDisposeScope();this.mid=m?.DetachFromDisposeScope();dora=s?.DetachFromDisposeScope();
        Strength=strength;Alpha=alpha;
    }
    private LoraWeightPatch(LoraWeightPatch source)
    {
        using var scope=NewDisposeScope();
        var u=source.up!.alias();var d=source.down!.alias();var m=source.mid?.alias();var s=source.dora?.alias();
        up=u.DetachFromDisposeScope();down=d.DetachFromDisposeScope();mid=m?.DetachFromDisposeScope();dora=s?.DetachFromDisposeScope();
        Strength=source.Strength;Alpha=source.Alpha;
    }
    public LoraWeightPatch Retain()
    {
        lock(gate){ObjectDisposedException.ThrowIf(up is null,this);return new(this);}
    }
    public Tensor Apply(Tensor weight,CancellationToken cancellationToken=default)
    {
        cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(weight);using var operation=Retain();
        using var scope=NewDisposeScope();using var noGrad=no_grad();
        // Copy factors to the weight device only for this operation; source snapshots remain immutable.
        var u=operation.up!.to(weight.device);var d=operation.down!.to(weight.device);
        var m=operation.mid?.to(weight.device);var s=operation.dora?.to(weight.device);
        return LoraMath.Apply(weight,u,d,Strength,Alpha,m,s,cancellationToken);
    }
    private static Tensor Copy(Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if(!InferenceDevice.IsSupported(tensor.device_type)||tensor.is_sparse||tensor.dtype is not(ScalarType.Float32 or ScalarType.Float16 or ScalarType.BFloat16))
            throw new ArgumentException("LoRA snapshots support dense CPU/CUDA F32, F16 and BF16 factors.",nameof(tensor));
        return tensor.to_type(ScalarType.Float32).clone();
    }
    public void Dispose()
    {
        Tensor? u,d,m,s;
        lock(gate){u=up;d=down;m=mid;s=dora;up=down=mid=dora=null;}
        u?.Dispose();d?.Dispose();m?.Dispose();s?.Dispose();
    }
}
