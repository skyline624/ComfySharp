namespace ComfySharp.Inference;

/// <summary>Metadata for the operators in frozen LoKrAdapter.h, independent of weight reconstruction.</summary>
internal static class LokrBypassGeometry
{
    internal static (long InputChannels,long OutputChannels) Channels(IReadOnlyDictionary<string,IReadOnlyList<long>> factors,int dimensions)
    {
        if(dimensions is <0 or >3)throw new ArgumentException("LoKr bypass requires a linear or Conv1d/2d/3d module.");
        IReadOnlyList<long> Required(string key)=>factors.TryGetValue(key,out var shape)?shape:throw new ArgumentException("Missing LoKr factor: "+key);
        void Rank(IReadOnlyList<long> shape,int rank){if(shape.Count!=rank||shape.Any(n=>n<=0))throw new ArgumentException("LoKr bypass factor has an incompatible rank.");}
        IReadOnlyList<long> first;
        if(factors.TryGetValue("lokr_w1",out var direct)){Rank(direct,2);first=direct;}
        else
        {
            var a=Required("lokr_w1_a");var b=Required("lokr_w1_b");Rank(a,2);Rank(b,2);
            if(a[1]!=b[0])throw new ArgumentException("LoKr first side matrix ranks differ.");
            first=new[]{a[0],b[1]};
        }
        long input,output;
        if(factors.TryGetValue("lokr_w2",out var second)){Rank(second,dimensions+2);input=second[1];output=second[0];}
        else
        {
            var a=Required("lokr_w2_b");var b=Required("lokr_w2_a");
            bool tucker=dimensions>0&&factors.ContainsKey("lokr_t2");
            // In convolution h, b gets unit spatial axes; a does so only for Tucker.
            void Operator(IReadOnlyList<long> shape,bool append){Rank(shape,append&&shape.Count==2?2:dimensions+2);}
            Operator(a,tucker);Operator(b,dimensions>0);
            input=a[1];output=b[0];long intermediate=a[0];
            if(tucker)
            {
                var core=Required("lokr_t2");Operator(core,true);
                if(core[1]!=intermediate)throw new ArgumentException("LoKr Tucker input channels differ.");
                intermediate=core[0];
            }
            if(b[1]!=intermediate)throw new ArgumentException("LoKr bypass intermediate channels differ.");
        }
        return(checked(first[1]*input),checked(first[0]*output));
    }

    internal static void ValidateTarget(IReadOnlyDictionary<string,IReadOnlyList<long>> factors,IReadOnlyList<long> target)
    {
        var channels=Channels(factors,target.Count-2);
        if(channels.InputChannels!=target[1])throw new ArgumentException("LoKr bypass input channels differ from the target.");
        if(channels.OutputChannels!=target[0]&&channels.OutputChannels!=1&&target[0]!=1)
            throw new ArgumentException("LoKr bypass output channels cannot broadcast with the target.");
        // Spatial output sizes depend on runtime inputs, stride and padding. The
        // actual native operators and broadcast addition validate those dimensions.
    }

    internal static void ValidateStorage(IReadOnlyDictionary<string,IReadOnlyList<long>> factors)
    {
        try{LokrMath.ReconstructedShape(factors);return;}catch(ArgumentException){}
        for(int dimensions=0;dimensions<=3;dimensions++)
            try{Channels(factors,dimensions);return;}catch(ArgumentException){}
        throw new ArgumentException("LoKr factors support neither reconstruction nor a bypass operator chain.");
    }
}
