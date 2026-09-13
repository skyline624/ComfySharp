using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Frozen LoKrAdapter Float32 weight reconstruction. Training LokrDiff
/// has different alpha semantics and remains a separate implementation.</summary>
public static class LokrMath
{
    internal static readonly string[] Names=["lokr_w1","lokr_w2","lokr_w1_a","lokr_w1_b","lokr_w2_a","lokr_w2_b","lokr_t2"];
    internal static long[] ReconstructedShape(IReadOnlyDictionary<string,IReadOnlyList<long>> factors)
    {
        IReadOnlyList<long> Required(string key)=>factors.TryGetValue(key,out var value)?value:throw new ArgumentException("Missing LoKr factor: "+key);
        void Matrix(IReadOnlyList<long> shape){if(shape.Count!=2||shape.Any(v=>v<=0))throw new ArgumentException("LoKr active side factors must be nonempty matrices.");}
        long[] Side(string direct,string a,string b,bool second)
        {
            if(factors.TryGetValue(direct,out var full))
            {
                if(full.Count<2||full.Count>(second?5:2)||full.Any(v=>v<=0))throw new ArgumentException("Unsupported direct LoKr side shape.");
                return full.ToArray();
            }
            var left=Required(a);var right=Required(b);Matrix(left);
            if(!second)Matrix(right);
            if(second&&factors.TryGetValue("lokr_t2",out var core))
            {
                Matrix(right);
                if(core.Count!=4||core.Any(v=>v<=0)||core[0]!=left[0]||core[1]!=right[0])throw new ArgumentException("LoKr Tucker core differs from its factors.");
                return[left[1],right[1],core[2],core[3]];
            }
            // A spatial second B factor is used by LoKrAdapter.h. Its ordinary
            // calculate_weight mm still fails as in source; only bypass accepts it.
            if(second&&(right.Count<2||right.Count>5||right.Any(v=>v<=0)))throw new ArgumentException("Invalid LoKr second side factor.");
            if(left[1]!=right[0])throw new ArgumentException("LoKr matrix ranks differ.");
            return new[]{left[0]}.Concat(right.Skip(1)).ToArray();
        }
        var first=Side("lokr_w1","lokr_w1_a","lokr_w1_b",false);var second=Side("lokr_w2","lokr_w2_a","lokr_w2_b",true);
        // Inference source appends axes ONLY for four-dimensional w2. Other
        // dimensions use torch.kron's leading-axis padding before target reshape.
        if(second.Length==4)first=[first[0],first[1],1,1];
        int count=Math.Max(first.Length,second.Length);var result=new long[count];
        for(int i=1;i<=count;i++)result[^i]=checked((i<=first.Length?first[^i]:1)*(i<=second.Length?second[^i]:1));
        return result;
    }
    public static Tensor Apply(Tensor weight,IReadOnlyDictionary<string,Tensor> factors,double strength=1,double? alpha=null,
        Tensor? doraScale=null,CancellationToken cancellationToken=default)
    {
        cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(weight);ArgumentNullException.ThrowIfNull(factors);
        NativeRuntimeBootstrap.Initialize();
        if(!double.IsFinite(strength)||alpha is not null&&!double.IsFinite(alpha.Value))throw new ArgumentOutOfRangeException(nameof(strength));
        if(weight.dtype!=ScalarType.Float32||weight.is_sparse||weight.dim()<2||!InferenceDevice.IsSupported(weight.device_type))
            throw new ArgumentException("LoKr inference requires a dense Float32 target weight.");
        foreach(var value in factors.Values)
        {
            if(value.dtype!=ScalarType.Float32||value.is_sparse)throw new ArgumentException("LoKr arithmetic requires dense Float32 factors.");
            InferenceDevice.RequireSame(weight.device,value,"factor");
        }
        var shape=ReconstructedShape(factors.ToDictionary(p=>p.Key,p=>(IReadOnlyList<long>)p.Value.shape,StringComparer.Ordinal));
        if(shape.Aggregate(1L,(a,b)=>checked(a*b))!=weight.numel())throw new ArgumentException("LoKr reconstruction cannot reshape to the target.");
        using var scope=NewDisposeScope();long? rank=null;
        Tensor first;
        if(factors.TryGetValue("lokr_w1",out var full1))first=full1;
        else{rank=factors["lokr_w1_b"].shape[0];first=mm(factors["lokr_w1_a"],factors["lokr_w1_b"]);}
        Tensor second;
        if(factors.TryGetValue("lokr_w2",out var full2))second=full2;
        else
        {
            rank=factors["lokr_w2_b"].shape[0];
            second=factors.TryGetValue("lokr_t2",out var core)?einsum("ijkl,jr,ip->prkl",core,factors["lokr_w2_b"],factors["lokr_w2_a"]):mm(factors["lokr_w2_a"],factors["lokr_w2_b"]);
        }
        if(second.dim()==4)first=first.unsqueeze(2).unsqueeze(2);
        // Frozen inference scales once; the second rebuilt side overwrites rank.
        double scale=alpha is not null&&rank is not null?alpha.Value/rank.Value:1;
        var diff=kron(first,second).reshape(weight.shape);
        var result=LoraMath.ApplyDifference(weight,diff,strength,scale,doraScale);
        if(!result.isfinite().all().item<bool>())throw new ArithmeticException("LoKr produced nonfinite weights.");
        cancellationToken.ThrowIfCancellationRequested();return result.DetachFromDisposeScope();
    }
}
