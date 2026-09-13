using static TorchSharp.torch;
namespace ComfySharp.Inference;

/// <summary>An immutable, independently disposable snapshot of LoRA/LoHa/LoKr factors or an additive difference for inference baking.
/// Low-level trainable operations use LoraMath directly; this owner intentionally freezes its factors.</summary>
public sealed class LoraWeightPatch : IDisposable
{
    private readonly object gate=new();
    private Tensor? up,down,mid,dora,difference,up2,down2,t1,t2;
    private Dictionary<string,Tensor>? lokr;
    public double Strength { get; }
    public double? Alpha { get; }
    internal bool IsDifference { get { lock(gate) { ObjectDisposedException.ThrowIf(up is null && difference is null && lokr is null,this); return difference is not null; } } }
    internal LoraWeightPatch To(Device device)
    {
        using var owner = Retain(); using var scope = NewDisposeScope();
        if (owner.difference is { } diff) return FromDifference(diff.to(device), Strength);
        if (owner.lokr is not null) return FromLokr(owner.lokr.ToDictionary(p=>p.Key,p=>p.Value.to(device),StringComparer.Ordinal),Strength,Alpha,owner.dora?.to(device));
        if (owner.up2 is not null) return FromLoha(owner.up!.to(device), owner.down!.to(device), owner.up2.to(device), owner.down2!.to(device),
            Strength, Alpha, owner.t1?.to(device), owner.t2?.to(device), owner.dora?.to(device));
        return new(owner.up!.to(device), owner.down!.to(device), Strength, Alpha, owner.mid?.to(device), owner.dora?.to(device));
    }
    internal Tensor ApplyBypass(Tensor input, Tensor baseOutput, IReadOnlyList<long>? kernelSize = null, long stride = 1, long padding = 0)
    {
        using var owner = Retain();
        if (owner.difference is not null) throw new InvalidOperationException("Additive differences use ordinary weight patching.");
        if (owner.lokr is not null) return LokrBypassMath.Apply(input,baseOutput,owner.lokr,Strength,Alpha,kernelSize,stride,padding);
        if (owner.up2 is not null) return LohaMath.ApplyBypass(input, baseOutput, owner.up!, owner.down!, owner.up2, owner.down2!,
            Strength, Alpha, owner.t1, owner.t2, kernelSize, stride, padding);
        // Frozen LoRAAdapter.h uses up/down/alpha/mid; its inherited g is identity, including when a DoRA field is present.
        return LoraBypassMath.Apply(input, baseOutput, owner.up!, owner.down!, Strength, Alpha, owner.mid, kernelSize, stride, padding);
    }
    private LoraWeightPatch(Tensor difference,double strength)
    {
        NativeRuntimeBootstrap.Initialize();using var scope=NewDisposeScope();using var noGrad=no_grad();
        if(!double.IsFinite(strength))throw new ArgumentOutOfRangeException(nameof(strength));
        var copy=Copy(difference);
        if(copy.dim()<1||copy.shape.Any(d=>d<=0)||!copy.isfinite().all().item<bool>())
            throw new InvalidDataException("An additive adapter requires finite nonempty tensor values.");
        this.difference=copy.DetachFromDisposeScope();Strength=strength;
    }
    public static LoraWeightPatch FromDifference(Tensor difference,double strength=1)=>new(difference,strength);
    public static LoraWeightPatch FromLokr(IReadOnlyDictionary<string,Tensor> factors,double strength=1,double? alpha=null,Tensor? doraScale=null)
        =>new(factors,strength,alpha,doraScale);
    private LoraWeightPatch(IReadOnlyDictionary<string,Tensor> factors,double strength,double? alpha,Tensor? doraScale)
    {
        ArgumentNullException.ThrowIfNull(factors);NativeRuntimeBootstrap.Initialize();using var scope=NewDisposeScope();using var noGrad=no_grad();
        if(!double.IsFinite(strength)||alpha is not null&&!double.IsFinite(alpha.Value))throw new ArgumentOutOfRangeException(nameof(strength));
        if(factors.Keys.Any(k=>!LokrMath.Names.Contains(k,StringComparer.Ordinal)))throw new ArgumentException("Unknown LoKr factor key.");
        var copied=factors.ToDictionary(p=>p.Key,p=>Copy(p.Value),StringComparer.Ordinal);var scale=doraScale is null?null:Copy(doraScale);
        LokrBypassGeometry.ValidateStorage(copied.ToDictionary(p=>p.Key,p=>(IReadOnlyList<long>)p.Value.shape,StringComparer.Ordinal));
        foreach(var value in copied.Values.Append(scale).OfType<Tensor>())if(!value.isfinite().all().item<bool>())throw new InvalidDataException("LoKr snapshots require finite factors.");
        foreach(var value in copied.Values)value.DetachFromDisposeScope();lokr=copied;dora=scale?.DetachFromDisposeScope();Strength=strength;Alpha=alpha;
    }
    public static LoraWeightPatch FromLoha(Tensor w1a, Tensor w1b, Tensor w2a, Tensor w2b,
        double strength=1, double? alpha=null, Tensor? t1=null, Tensor? t2=null, Tensor? doraScale=null)
        => new(w1a,w1b,w2a,w2b,strength,alpha,t1,t2,doraScale);
    private LoraWeightPatch(Tensor w1a, Tensor w1b, Tensor w2a, Tensor w2b,
        double strength, double? alpha, Tensor? core1, Tensor? core2, Tensor? doraScale)
    {
        NativeRuntimeBootstrap.Initialize(); using var scope=NewDisposeScope(); using var noGrad=no_grad();
        if(!double.IsFinite(strength)||alpha is not null&&!double.IsFinite(alpha.Value))throw new ArgumentOutOfRangeException(nameof(strength));
        var a=Copy(w1a); var b=Copy(w1b); var c=Copy(w2a); var d=Copy(w2b);
        var first=core1 is null?null:Copy(core1); var second=core2 is null?null:Copy(core2); var scale=doraScale is null?null:Copy(doraScale);
        LohaMath.ReconstructedShape(a.shape,b.shape,c.shape,d.shape,first?.shape,second?.shape);
        foreach(var value in new[]{a,b,c,d,first,second,scale}.OfType<Tensor>())
            if(!value.isfinite().all().item<bool>())throw new InvalidDataException("LoHa snapshots require finite factors.");
        up=a.DetachFromDisposeScope(); down=b.DetachFromDisposeScope(); up2=c.DetachFromDisposeScope(); down2=d.DetachFromDisposeScope();
        t1=first?.DetachFromDisposeScope(); t2=second?.DetachFromDisposeScope(); dora=scale?.DetachFromDisposeScope();
        Strength=strength; Alpha=alpha;
    }
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
        var u=source.up?.alias();var d=source.down?.alias();var m=source.mid?.alias();var s=source.dora?.alias();var diff=source.difference?.alias();
        var u2=source.up2?.alias();var d2=source.down2?.alias();var first=source.t1?.alias();var second=source.t2?.alias();
        var kr=source.lokr?.ToDictionary(p=>p.Key,p=>p.Value.alias(),StringComparer.Ordinal);
        up=u?.DetachFromDisposeScope();down=d?.DetachFromDisposeScope();mid=m?.DetachFromDisposeScope();dora=s?.DetachFromDisposeScope();difference=diff?.DetachFromDisposeScope();
        up2=u2?.DetachFromDisposeScope();down2=d2?.DetachFromDisposeScope();t1=first?.DetachFromDisposeScope();t2=second?.DetachFromDisposeScope();
        if(kr is not null)foreach(var value in kr.Values)value.DetachFromDisposeScope();lokr=kr;
        Strength=source.Strength;Alpha=source.Alpha;
    }
    public LoraWeightPatch Retain()
    {
        lock(gate){ObjectDisposedException.ThrowIf(up is null&&difference is null&&lokr is null,this);return new(this);}
    }
    public Tensor Apply(Tensor weight,CancellationToken cancellationToken=default)
    {
        cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(weight);using var operation=Retain();
        using var scope=NewDisposeScope();using var noGrad=no_grad();
        if(operation.difference is { } diff)
        {
            if(weight.dtype!=ScalarType.Float32||weight.is_sparse||!weight.shape.SequenceEqual(diff.shape))
                throw new ArgumentException("Difference and target require identical dense Float32 shapes.",nameof(weight));
            var result=weight+diff.to(weight.device)*Strength;
            cancellationToken.ThrowIfCancellationRequested();return result.MoveToOuterDisposeScope();
        }
        // Copy factors to the weight device only for this operation; source snapshots remain immutable.
        if(operation.lokr is not null)return LokrMath.Apply(weight,operation.lokr.ToDictionary(p=>p.Key,p=>p.Value.to(weight.device),StringComparer.Ordinal),
            Strength,Alpha,operation.dora?.to(weight.device),cancellationToken);
        var u=operation.up!.to(weight.device);var d=operation.down!.to(weight.device);
        var m=operation.mid?.to(weight.device);var s=operation.dora?.to(weight.device);
        if(operation.up2 is not null) return LohaMath.Apply(weight,u,d,operation.up2.to(weight.device),operation.down2!.to(weight.device),
            Strength,Alpha,operation.t1?.to(weight.device),operation.t2?.to(weight.device),s,cancellationToken);
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
        Tensor? u,d,m,s,diff,u2,d2,first,second;
        Dictionary<string,Tensor>? kr;
        lock(gate){u=up;d=down;m=mid;s=dora;diff=difference;u2=up2;d2=down2;first=t1;second=t2;kr=lokr;lokr=null;up=down=mid=dora=difference=up2=down2=t1=t2=null;}
        u?.Dispose();d?.Dispose();m?.Dispose();s?.Dispose();diff?.Dispose();
        u2?.Dispose();d2?.Dispose();first?.Dispose();second?.Dispose();
        if(kr is not null)foreach(var value in kr.Values)value.Dispose();
    }
}
