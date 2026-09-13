namespace ComfySharp.Inference;

internal static class AdapterBypassGeometry
{
    internal static void OutputChannels(long output,long target)
    {
        if(output!=target&&output!=1&&target!=1)throw new ArgumentException("Bypass output channels cannot broadcast with the target.");
    }
    internal static void Lora(IReadOnlyList<long> up,IReadOnlyList<long> down,IReadOnlyList<long>? mid,IReadOnlyList<long> target)
    {
        if(target.Count is <2 or >5)throw new ArgumentException("Bypass requires linear or Conv1d/2d/3d geometry.");
        bool convolution=target.Count>2;
        void Operator(IReadOnlyList<long> shape)
        {
            if(shape.Count!=(convolution&&shape.Count!=2?target.Count:2)||shape.Any(n=>n<=0))throw new ArgumentException("Bypass factor rank differs from the module.");
        }
        Operator(up);Operator(down);if(mid is not null)Operator(mid);
        long columns=convolution&&down.Count==2?target.Skip(1).Aggregate(1L,(a,b)=>checked(a*b)):target[1];
        if(down[1]!=columns)throw new ArgumentException("Bypass down factor input channels/kernel differ.");
        long intermediate=down[0];
        if(mid is not null)
        {
            if(mid[1]!=intermediate)throw new ArgumentException("Bypass mid factor channels differ.");
            intermediate=mid[0];
        }
        if(up[1]!=intermediate)throw new ArgumentException("Bypass up factor channels differ.");
        OutputChannels(up[0],target[0]);
    }
    internal static void Loha(IReadOnlyList<long> difference,IReadOnlyList<long> target)
    {
        if(target.Count is <2 or >5)throw new ArgumentException("Bypass requires linear or Conv1d/2d/3d geometry.");
        if(difference.Count==2)
        {
            long columns=target.Skip(1).Aggregate(1L,(a,b)=>checked(a*b));
            if(difference[1]!=columns)throw new ArgumentException("LoHa bypass reconstruction cannot reshape to the module inputs.");
        }
        else if(difference.Count!=target.Count||difference[1]!=target[1])throw new ArgumentException("LoHa bypass kernel rank/input channels differ.");
        OutputChannels(difference[0],target[0]);
    }
}
