using System.Collections.ObjectModel;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Owned Float32 LokrDiff parameters. Direct factors take precedence while
/// decomposed parameters remain registered, as in the frozen source constructor.</summary>
public sealed class TrainableLokrPatch : TrainableWeightPatch
{
    private sealed class Shared(Dictionary<string,Tensor> values)
    {
        internal readonly object Gate=new();
        internal readonly IReadOnlyDictionary<string,Tensor> Named=new ReadOnlyDictionary<string,Tensor>(values);
        internal readonly IReadOnlyList<Tensor> Parameters=Array.AsReadOnly(values.Values.ToArray());
        internal int Owners=1;
    }
    private readonly Shared shared;
    private bool disposed;
    private TrainableLokrPatch(Shared shared)=>this.shared=shared;

    internal static (long Small,long Large) Factorize(long dimension,int factor)
    {
        if(dimension<=0)throw new ArgumentOutOfRangeException(nameof(dimension));
        if(factor>0&&dimension%factor==0&&dimension>=(long)factor*factor)return(factor,dimension/factor);
        long limit=factor<0?dimension:factor,m=1,n=dimension;
        while(m<n)
        {
            long next=m+1;while(dimension%next!=0)next++;
            long other=dimension/next;
            if(next>limit||next>dimension-other+1)break;
            m=next;n=other;
        }
        return m>n?(n,m):(m,n);
    }

    public TrainableLokrPatch(Tensor? w1,Tensor? w2,double alpha=1,Tensor? w1a=null,Tensor? w1b=null,
        Tensor? w2a=null,Tensor? w2b=null,Tensor? t2=null)
    {
        if(!double.IsFinite(alpha)||!float.IsFinite((float)alpha))throw new ArgumentOutOfRangeException(nameof(alpha));
        if((w1a is null)!=(w1b is null)||(w2a is null)!=(w2b is null)||
            (w1 is null&&w1a is null)||(w2 is null&&w2a is null)||t2 is not null&&w2a is null)
            throw new ArgumentException("LoKr requires complete direct or decomposed sides and Tucker factors for a core.");
        NativeRuntimeBootstrap.Initialize();using var scope=NewDisposeScope();using var noGrad=no_grad();
        var original=new Dictionary<string,Tensor>(StringComparer.Ordinal);
        if(w1a is not null){original.Add("lokr_w1_a",w1a);original.Add("lokr_w1_b",w1b!);}
        if(w2a is not null){original.Add("lokr_w2_a",w2a);original.Add("lokr_w2_b",w2b!);if(t2 is not null)original.Add("lokr_t2",t2);}
        if(w1 is not null)original.Add("lokr_w1",w1);
        if(w2 is not null)original.Add("lokr_w2",w2);
        var device=original.First().Value.device;
        foreach(var (name,value) in original)
        {
            if(value.dtype!=ScalarType.Float32||value.is_sparse||!InferenceDevice.IsSupported(value.device_type)||
                value.dim()<2||value.shape.Any(n=>n<=0)||!value.isfinite().all().item<bool>())
                throw new ArgumentException("LoKr requires finite dense Float32 factors: "+name);
            InferenceDevice.RequireSame(device,value,name);
            if(name is not("lokr_w2" or "lokr_t2")&&value.dim()!=2)throw new ArgumentException("LoKr side factors must be matrices.");
        }
        if(w1 is null&&w1a!.shape[1]!=w1b!.shape[0])throw new ArgumentException("LoKr first side matrix dimensions differ.");
        if(w2 is null)
        {
            if(t2 is null&&w2a!.shape[1]!=w2b!.shape[0])throw new ArgumentException("LoKr second side matrix dimensions differ.");
            if(t2 is not null&&(t2.dim()!=4||t2.shape[0]!=w2a!.shape[0]||t2.shape[1]!=w2b!.shape[0]))
                throw new ArgumentException("LoKr Tucker requires a four-dimensional core with matching input factor rows.");
        }
        var snapshots=original.ToDictionary(p=>p.Key,p=>p.Value.detach().clone().requires_grad_(),StringComparer.Ordinal);
        // The training factory enables all parameters, including alpha. Direct sides
        // ignore alpha; each rebuilt side independently applies its own alpha/rank.
        snapshots.Add("alpha",tensor((float)alpha,device:device).requires_grad_());
        shared=new(snapshots);foreach(var value in snapshots.Values)value.DetachFromDisposeScope();
    }
    public IReadOnlyDictionary<string,Tensor> NamedParameters {get{lock(shared.Gate){ThrowIfDisposed();return shared.Named;}}}
    public override IReadOnlyList<Tensor> Parameters {get{lock(shared.Gate){ThrowIfDisposed();return shared.Parameters;}}}
    public override TrainableLokrPatch Retain(){lock(shared.Gate){ThrowIfDisposed();shared.Owners=checked(shared.Owners+1);return new(shared);}}

    private (Tensor W1,Tensor W2) Sides()
    {
        var p=shared.Named;
        Tensor first=p.TryGetValue("lokr_w1",out var w1)?w1:
            (p["lokr_w1_a"].matmul(p["lokr_w1_b"]))*(p["alpha"]/p["lokr_w1_b"].shape[0]);
        Tensor second;
        if(p.TryGetValue("lokr_w2",out var w2))second=w2;
        else
        {
            second=p.TryGetValue("lokr_t2",out var t2)?einsum("ijkl,jr,ip->prkl",t2,p["lokr_w2_b"],p["lokr_w2_a"]):
                p["lokr_w2_a"].matmul(p["lokr_w2_b"]);
            second=second*(p["alpha"]/p["lokr_w2_b"].shape[0]);
        }
        return(first,second);
    }
    public Tensor Apply(Tensor weight)=>Apply(weight,default);
    internal override Tensor Apply(Tensor weight,CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(weight);
        using var owner=Retain();using var scope=NewDisposeScope();
        if(weight.is_sparse||weight.dim()<2||weight.dtype is not(ScalarType.Float32 or ScalarType.Float16 or ScalarType.BFloat16))
            throw new ArgumentException("LoKr requires a dense floating-point target weight.");
        InferenceDevice.RequireSame(shared.Parameters[0].device,weight,nameof(weight));
        var(first,second)=Sides();
        for(long i=first.dim();i<second.dim();i++)first=first.unsqueeze(-1);
        var diff=kron(first,second);
        if(diff.numel()!=weight.numel())throw new ArgumentException("LoKr reconstruction size differs from target.");
        var result=weight+diff.reshape(weight.shape).to_type(weight.dtype);
        cancellationToken.ThrowIfCancellationRequested();return result.MoveToOuterDisposeScope();
    }
    internal override Tensor ApplyBypass(Tensor input,Tensor baseOutput,IReadOnlyList<long>? kernelSize,long stride,long padding)
    {
        using var owner=Retain();using var scope=NewDisposeScope();
        int dims=LokrBypassMath.Validate(input,baseOutput,kernelSize,stride,padding);
        if(input.dtype!=ScalarType.Float32)throw new ArgumentException("Trainable LoKr bypass currently requires Float32 activations.");
        InferenceDevice.RequireSame(input.device,shared.Parameters[0],"factor");
        var(first,second)=Sides();var grouped=LokrBypassMath.Group(input,first.shape[1],dims);
        if(dims>0&&second.dim()==2)second=second.view(second.shape.Concat(Enumerable.Repeat(1L,dims)).ToArray());
        var hidden=LokrBypassMath.Op(grouped,second,dims,stride,padding);
        // Source LokrDiff.h always multiplies by the injected multiplier (1.0
        // for the training group), retaining this native operation and its layout.
        var delta=LokrBypassMath.Cross(hidden,first,input.shape[0],dims)*1.0;
        return LokrBypassMath.Combine(delta,baseOutput).MoveToOuterDisposeScope();
    }
    private void ThrowIfDisposed()=>ObjectDisposedException.ThrowIf(disposed,this);
    public override void Dispose(){lock(shared.Gate){if(disposed)return;disposed=true;if(--shared.Owners==0)foreach(var value in shared.Parameters)value.Dispose();}}
}
