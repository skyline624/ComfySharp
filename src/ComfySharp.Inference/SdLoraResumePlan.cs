namespace ComfySharp.Inference;

/// <summary>Frozen training factory selection. Deliberately distinct from inference aliases,
/// alpha keys and difference loading. Metadata only; supplied source is borrowed.</summary>
internal sealed class SdLoraResumePlan
{
    internal sealed record Factors(string? Up, string? Down, long Rank, LohaFactorKeys? Loha=null, long ParameterElements=0,
        IReadOnlyDictionary<string,string>? Lokr=null);
    internal Dictionary<string, Factors> Targets { get; } = new(StringComparer.Ordinal);
    internal Dictionary<string, string> Alphas { get; } = new(StringComparer.Ordinal);
    internal IReadOnlyList<string> IgnoredKeys { get; private set; } = Array.Empty<string>();

    internal static SdLoraResumePlan Inspect(ILoraTensorSource source, IReadOnlyDictionary<string,IReadOnlyList<long>> schema,
        CancellationToken token)
    {
        var plan = new SdLoraResumePlan(); var consumed = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (name, shape) in schema)
        {
            token.ThrowIfCancellationRequested();
            if (shape.Count < 2) continue; // Source always constructs fresh BiasDiff leaves.
            string prefix = "diffusion_model." + name[..^7];
            string alpha = prefix + ".weight.alpha";
            if (source.Tensors.TryGetValue(alpha, out var info))
            {
                if (!SafeTensorFile.SupportsTensorDType(info.DType) || info.Shape.Aggregate(1L,(a,b)=>checked(a*b)) != 1)
                    throw new InvalidDataException("Resume alpha must contain one readable scalar: " + alpha);
                plan.Alphas.Add(name,alpha); consumed.Add(alpha);
            }
            bool selected = false;
            foreach (var format in LoraFileLoader.Formats)
            {
                string up = prefix + format.Up, down = prefix + format.Down;
                if (!source.Tensors.TryGetValue(up, out var u)) continue;
                if (!source.Tensors.TryGetValue(down, out var d)) throw new InvalidDataException("Resume factor has no matching down matrix: " + up);
                if (format.Mid && source.Tensors.ContainsKey(prefix + ".lora_mid.weight"))
                    throw new NotSupportedException("Training resume of LoCon mid factors remains unported: " + prefix);
                long columns = shape.Skip(1).Aggregate(1L,(a,b)=>checked(a*b));
                if (!SafeTensorFile.SupportsTensorDType(u.DType) || !SafeTensorFile.SupportsTensorDType(d.DType) ||
                    u.Shape.Count != 2 || d.Shape.Count != 2 || u.Shape[0] != shape[0] || d.Shape[1] != columns ||
                    u.Shape[1] < 1 || u.Shape[1] != d.Shape[0])
                    throw new InvalidDataException("Resume matrices must match the SD target and share a positive rank: " + prefix);
                plan.Targets.Add(name,new(up,down,d.Shape[0])); consumed.Add(up); consumed.Add(down);
                selected = true; break;
            }
            // Training stops at the first provider (LoRA, then LoHa, then LoKr), unlike inference load_lora.
            if(!selected&&source.Tensors.ContainsKey(prefix+".hada_w1_a"))
            {
                string Required(string suffix)
                {
                    string key=prefix+suffix;
                    if(!source.Tensors.TryGetValue(key,out var value))throw new InvalidDataException("Missing LoHa resume factor: "+key);
                    if(value.DType is not("F32" or "F16" or "BF16" or "F64")||value.Shape.Any(n=>n<=0))
                        throw new InvalidDataException("LoHa resume requires nonempty floating-point factors: "+key);
                    consumed.Add(key);return key;
                }
                string first=Required(".hada_w1_a"),second=Required(".hada_w1_b");
                var loha=new LohaFactorKeys(Required(".hada_w2_a"),Required(".hada_w2_b"),
                    source.Tensors.ContainsKey(prefix+".hada_t1")?Required(".hada_t1"):null,
                    source.Tensors.ContainsKey(prefix+".hada_t1")?Required(".hada_t2"):null);
                var a=source.Tensors[first].Shape;var b=source.Tensors[second].Shape;
                var c=source.Tensors[loha.W2A].Shape;var d=source.Tensors[loha.W2B].Shape;
                var t1=loha.T1 is null?null:source.Tensors[loha.T1].Shape;var t2=loha.T2 is null?null:source.Tensors[loha.T2].Shape;
                try {TrainableLohaPatch.ValidateGeometry(a,b,c,d,t1,t2,shape);}
                catch(ArgumentException error){throw new InvalidDataException("LoHa resume geometry differs from the target: "+prefix,error);}
                long Elements(IReadOnlyList<long> dims)=>dims.Aggregate(1L,(x,y)=>checked(x*y));
                long count=checked(Elements(a)+Elements(b)+Elements(c)+Elements(d)+(t1 is null?0:Elements(t1)+Elements(t2!))+1);
                plan.Targets.Add(name,new(first,second,b[0],loha,count));selected=true;
            }
            if(!selected&&new[]{"lokr_w1","lokr_w2","lokr_w1_a","lokr_w2_a"}.Any(k=>source.Tensors.ContainsKey(prefix+"."+k)))
            {
                var keys=new Dictionary<string,string>(StringComparer.Ordinal);
                bool Present(string key)=>source.Tensors.ContainsKey(prefix+"."+key);
                void Required(string key)
                {
                    string full=prefix+"."+key;
                    if(!source.Tensors.TryGetValue(full,out var value))throw new InvalidDataException("Missing LoKr resume factor: "+full);
                    if(value.DType is not("F32" or "F16" or "BF16" or "F64")||value.Shape.Count<2||value.Shape.Any(n=>n<=0)||
                        key is not("lokr_w2" or "lokr_t2")&&value.Shape.Count!=2)
                        throw new InvalidDataException("LoKr resume requires nonempty floating-point factors with source dimensions: "+full);
                    keys.Add(key,full);consumed.Add(full);
                }
                // LokrDiff registers decomposed pairs even when direct sides win.
                // Orphan b/core tensors are loaded by LoKrAdapter but never registered.
                if(Present("lokr_w1_a")){Required("lokr_w1_a");Required("lokr_w1_b");}
                if(Present("lokr_w2_a")){Required("lokr_w2_a");Required("lokr_w2_b");if(Present("lokr_t2"))Required("lokr_t2");}
                if(Present("lokr_w1"))Required("lokr_w1");if(Present("lokr_w2"))Required("lokr_w2");
                long Elements(IReadOnlyList<long> dims)=>dims.Aggregate(1L,(x,y)=>checked(x*y));
                try
                {
                    var rebuilt=LokrMath.ReconstructedShape(keys.ToDictionary(p=>p.Key,p=>source.Tensors[p.Value].Shape,StringComparer.Ordinal));
                    if(Elements(rebuilt)!=Elements(shape))throw new ArgumentException("Reconstructed element count differs.");
                }
                catch(ArgumentException error){throw new InvalidDataException("LoKr resume geometry differs from the target: "+prefix,error);}
                long count=checked(keys.Values.Sum(k=>Elements(source.Tensors[k].Shape))+1);
                plan.Targets.Add(name,new(null,null,0,ParameterElements:count,Lokr:new System.Collections.ObjectModel.ReadOnlyDictionary<string,string>(keys)));selected=true;
            }
            // Later adapter loaders would claim these formats; never silently replace them by fresh LoRA.
            bool otherAlgorithm = new[]{".a1.weight"}
                .Any(suffix => source.Tensors.ContainsKey(prefix + suffix)) ||
                source.Tensors.TryGetValue(prefix + ".oft_blocks",out var blocks) && blocks.Shape.Count is 3 or 4;
            if (!selected && otherAlgorithm)
                throw new NotSupportedException("Training resume requires an unported adapter algorithm: " + prefix);
        }
        plan.IgnoredKeys = Array.AsReadOnly(source.Tensors.Keys.Where(k=>!consumed.Contains(k)).Order(StringComparer.Ordinal).ToArray());
        return plan;
    }
}
