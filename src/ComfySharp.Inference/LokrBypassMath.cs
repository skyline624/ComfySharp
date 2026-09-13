using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Frozen LoKrAdapter.h using grouped operators, never a materialized Kronecker weight.
/// Training LokrDiff.h uses separately reconstructed/scaled sides.</summary>
public static class LokrBypassMath
{
    internal static int Validate(Tensor input,Tensor baseOutput,IReadOnlyList<long>? kernelSize,long stride,long padding)
    {
        ArgumentNullException.ThrowIfNull(input);ArgumentNullException.ThrowIfNull(baseOutput);
        if(input.is_sparse||baseOutput.is_sparse||input.dtype!=baseOutput.dtype||
            input.dtype is not(ScalarType.Float32 or ScalarType.Float16 or ScalarType.BFloat16)||input.shape.Any(n=>n<=0))
            throw new ArgumentException("LoKr bypass requires dense floating-point activations with matching dtypes.");
        InferenceDevice.RequireSame(input.device,baseOutput,nameof(baseOutput));
        int dims=kernelSize?.Count??0;
        if(stride<1||padding<0||dims>3||kernelSize is not null&&(dims<1||kernelSize.Any(n=>n<=0))||
            dims>0&&input.dim()!=dims+2||dims==0&&input.dim()<1)
            throw new ArgumentException("LoKr bypass requires linear or Conv1d/2d/3d geometry.");
        return dims;
    }
    internal static Tensor Op(Tensor input,Tensor weight,int dims,long stride=1,long padding=0)=>dims switch
    {
        0=>nn.functional.linear(input,weight),
        1=>nn.functional.conv1d(input,weight,stride:stride,padding:padding),
        2=>nn.functional.conv2d(input,weight,strides:[stride,stride],padding:[padding,padding]),
        3=>nn.functional.conv3d(input,weight,strides:[stride,stride,stride],padding:[padding,padding,padding]),
        _=>throw new ArgumentOutOfRangeException(nameof(dims))
    };
    internal static Tensor Group(Tensor input,long groups,int dims)=>dims>0
        ?input.reshape(new[]{checked(input.shape[0]*groups),-1L}.Concat(input.shape.Skip(2)).ToArray())
        :input.reshape(input.shape.Take(checked((int)input.dim()-1)).Concat(new[]{groups,-1L}).ToArray());
    internal static Tensor Cross(Tensor hidden,Tensor first,long batch,int dims)
    {
        using var scope=NewDisposeScope();
        if(dims>0)hidden=hidden.view(new[]{batch,-1L}.Concat(hidden.shape.Skip(1)).ToArray());
        var cross=dims>0?hidden.transpose(1,-1):hidden.transpose(-1,-2);
        var projected=nn.functional.linear(cross,first);
        projected=dims>0?projected.transpose(1,-1):projected.transpose(-1,-2);
        var output=dims>0?projected.reshape(new[]{batch,-1L}.Concat(projected.shape.Skip(3)).ToArray())
            :projected.reshape(projected.shape.Take(checked((int)projected.dim()-2)).Concat(new[]{-1L}).ToArray());
        return output.MoveToOuterDisposeScope();
    }
    internal static Tensor Combine(Tensor delta,Tensor baseOutput)
    {
        // Frozen bypass injection adds h(x) with native broadcasting, including
        // channel/spatial dimensions of size one; incompatible dimensions still fail.
        return baseOutput+delta;
    }
    public static Tensor Apply(Tensor input,Tensor baseOutput,IReadOnlyDictionary<string,Tensor> factors,double strength=1,double? alpha=null,
        IReadOnlyList<long>? kernelSize=null,long stride=1,long padding=0,CancellationToken cancellationToken=default)
    {
        cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(factors);
        if(!double.IsFinite(strength)||alpha is not null&&!double.IsFinite(alpha.Value))throw new ArgumentOutOfRangeException(nameof(strength));
        int dims=Validate(input,baseOutput,kernelSize,stride,padding);
        foreach(var value in factors.Values)
        {
            if(value.is_sparse||value.dtype is not(ScalarType.Float32 or ScalarType.Float16 or ScalarType.BFloat16))throw new ArgumentException("LoKr bypass requires dense floating-point factors.");
            InferenceDevice.RequireSame(input.device,value,"factor");
        }
        using var scope=NewDisposeScope();
        bool direct1=factors.TryGetValue("lokr_w1",out var first),direct2=factors.TryGetValue("lokr_w2",out var second);
        // Source h chooses the FIRST rebuilt rank, unlike calculate_weight.
        double? rank=!direct1?factors["lokr_w1_b"].shape[0]:!direct2?factors["lokr_w2_b"].shape[0]:alpha;
        if(alpha is not null&&rank==0)throw new DivideByZeroException("Source LoKr bypass alpha/rank divides by zero for direct factors with alpha zero.");
        double scale=(alpha is null?1:alpha.Value/rank!.Value)*strength;
        Tensor Cast(string name)=>factors[name].to_type(input.dtype);
        first=direct1?first!.to_type(input.dtype):mm(Cast("lokr_w1_a"),Cast("lokr_w1_b"));
        var grouped=Group(input,first.shape[1],dims);Tensor hidden;
        if(direct2)hidden=Op(grouped,second!.to_type(input.dtype),dims,stride,padding);
        else
        {
            var a=Cast("lokr_w2_b");var b=Cast("lokr_w2_a");bool tucker=factors.ContainsKey("lokr_t2");
            Tensor Spatial(Tensor value)=>value.dim()==2?value.view(value.shape.Concat(Enumerable.Repeat(1L,dims)).ToArray()):value;
            if(dims>0)
            {
                if(tucker)a=Spatial(a);b=Spatial(b);
                if(tucker)
                {
                    var t=Spatial(Cast("lokr_t2"));
                    hidden=Op(Op(Op(grouped,a,dims),t,dims,stride,padding),b,dims);
                }
                else hidden=Op(Op(grouped,a,dims,stride,padding),b,dims);
            }
            else hidden=Op(Op(grouped,a,0),b,0);
        }
        var result=Combine(Cross(hidden,first,input.shape[0],dims)*scale,baseOutput);
        cancellationToken.ThrowIfCancellationRequested();return result.MoveToOuterDisposeScope();
    }
}
