using static TorchSharp.torch;
namespace ComfySharp.Inference;

public sealed record LoraTarget(string Component,string Weight,IReadOnlyList<long> Shape);
public sealed record LoraAlias(string Prefix,LoraTarget Target);
public sealed record LohaFactorKeys(string W2A,string W2B,string? T1=null,string? T2=null);
public enum LoraLoadMode { Weights, Bypass }
public sealed record LoraBinding(string Prefix,LoraTarget Target,string? Up,string? Down,string? Mid,string? Alpha,string? Dora,string? Difference=null,LohaFactorKeys? Loha=null,IReadOnlyDictionary<string,string>? Lokr=null);

/// <summary>Metadata-only selection tied to one tensor source. Alias order is authoritative:
/// later aliases overwrite earlier bindings for the same target, as frozen load_lora does.</summary>
public sealed class LoraLoadPlan
{
    internal ILoraTensorSource Source { get; }
    public IReadOnlyList<LoraBinding> Bindings { get; }
    public IReadOnlyList<string> UnclaimedKeys { get; }
    public IReadOnlyList<string> ShadowedPrefixes { get; }
    public IReadOnlyList<string> Components { get; }
    public long ResidentFactorBytes { get; }
    public LoraLoadMode Mode { get; }
    internal LoraLoadPlan(ILoraTensorSource file,LoraBinding[] bindings,string[] unclaimed,string[] shadowed,string[] components,long bytes,LoraLoadMode mode)
    {Source=file;Bindings=Array.AsReadOnly(bindings);UnclaimedKeys=Array.AsReadOnly(unclaimed);ShadowedPrefixes=Array.AsReadOnly(shadowed);Components=Array.AsReadOnly(components);ResidentFactorBytes=bytes;Mode=mode;}
}

public static class LoraFileLoader
{
    // This is source LoRAAdapter.load order, independent of safetensors header order.
    internal static readonly (string Up,string Down,bool Mid)[] Formats=[
        (".lora_up.weight",".lora_down.weight",true),("_lora.up.weight","_lora.down.weight",false),
        (".lora_B.weight",".lora_A.weight",false),(".lora.up.weight",".lora.down.weight",false),
        (".lora_B",".lora_A",false),(".lora_linear_layer.up.weight",".lora_linear_layer.down.weight",false),
        (".lora_B.default.weight",".lora_A.default.weight",false)];

    public static LoraLoadPlan Inspect(ILoraTensorSource file,IReadOnlyList<LoraAlias> aliases,
        bool allowUnclaimedKeys=false,long maxResidentFactorBytes=512L*1024*1024,CancellationToken cancellationToken=default,LoraLoadMode mode=LoraLoadMode.Weights)
    {
        ArgumentNullException.ThrowIfNull(file);ArgumentNullException.ThrowIfNull(aliases);
        cancellationToken.ThrowIfCancellationRequested();
        if(!Enum.IsDefined(mode))throw new ArgumentOutOfRangeException(nameof(mode));
        if(maxResidentFactorBytes<0)throw new ArgumentOutOfRangeException(nameof(maxResidentFactorBytes));
        var selected=new Dictionary<(string,string),LoraBinding>();var claimed=new HashSet<string>(StringComparer.Ordinal);
        var prefixes=new HashSet<string>(StringComparer.Ordinal);var components=new HashSet<string>(StringComparer.Ordinal);var shadowed=new List<string>();
        foreach(var alias in aliases)
        {
            cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(alias);
            if(string.IsNullOrWhiteSpace(alias.Prefix)||!prefixes.Add(alias.Prefix))throw new ArgumentException("Alias prefixes must be nonempty and unique.",nameof(aliases));
            ArgumentNullException.ThrowIfNull(alias.Target);
            ArgumentNullException.ThrowIfNull(alias.Target.Shape);
            if(string.IsNullOrWhiteSpace(alias.Target.Component)||string.IsNullOrWhiteSpace(alias.Target.Weight)||alias.Target.Shape.Count<1||alias.Target.Shape.Any(v=>v<=0))
                throw new ArgumentException("Adapter targets need a component, canonical weight and positive shape.",nameof(aliases));
            var target=alias.Target with{Shape=Array.AsReadOnly(alias.Target.Shape.ToArray())};
            components.Add(target.Component);
            foreach(var format in Formats)
            {
                string up=alias.Prefix+format.Up;
                if(!file.Tensors.ContainsKey(up))continue;
                string down=alias.Prefix+format.Down;
                if(!file.Tensors.ContainsKey(down))throw new InvalidDataException($"LoRA up factor '{up}' has no matching down factor '{down}'.");
                if(file.Tensors.ContainsKey(alias.Prefix+".reshape_weight"))throw new NotSupportedException($"LoRA reshape_weight is not implemented for '{alias.Prefix}'.");
                string? Present(string suffix)=>file.Tensors.ContainsKey(alias.Prefix+suffix)?alias.Prefix+suffix:null;
                var binding=new LoraBinding(alias.Prefix,target,up,down,format.Mid?Present(".lora_mid.weight"):null,Present(".alpha"),Present(".dora_scale"));
                Validate(file,binding,mode);
                foreach(string key in FactorKeys(binding))claimed.Add(key);
                if(binding.Alpha is not null)claimed.Add(binding.Alpha);
                var identity=(target.Component,target.Weight);
                if(selected.TryGetValue(identity,out var prior))shadowed.Add(prior.Prefix);
                selected[identity]=binding;break;
            }
            // Frozen load_lora continues through providers: LoHa overwrites LoRA for the same alias.
            if(file.Tensors.ContainsKey(alias.Prefix+".hada_w1_a"))
            {
                string Required(string suffix)
                {
                    string key=alias.Prefix+suffix;
                    if(!file.Tensors.ContainsKey(key))throw new InvalidDataException($"LoHa adapter '{alias.Prefix}' is missing '{key}'.");
                    return key;
                }
                string? Present(string suffix)=>file.Tensors.ContainsKey(alias.Prefix+suffix)?alias.Prefix+suffix:null;
                var first=Present(".hada_t1");
                var factors=new LohaFactorKeys(Required(".hada_w2_a"),Required(".hada_w2_b"),first,first is null?null:Required(".hada_t2"));
                var binding=new LoraBinding(alias.Prefix,target,Required(".hada_w1_a"),Required(".hada_w1_b"),null,Present(".alpha"),Present(".dora_scale"),Loha:factors);
                Validate(file,binding,mode);
                foreach(string key in FactorKeys(binding))claimed.Add(key);
                if(binding.Alpha is not null)claimed.Add(binding.Alpha);
                var identity=(target.Component,target.Weight);
                if(selected.TryGetValue(identity,out var prior)&&prior.Prefix!=alias.Prefix)shadowed.Add(prior.Prefix);
                selected[identity]=binding;
            }
            // The next frozen provider is LoKr, which overwrites LoRA/LoHa for this alias.
            if(new[]{"lokr_w1","lokr_w2","lokr_w1_a","lokr_w2_a"}.Any(k=>file.Tensors.ContainsKey(alias.Prefix+"."+k)))
            {
                var keys=LokrMath.Names.Where(k=>file.Tensors.ContainsKey(alias.Prefix+"."+k))
                    .ToDictionary(k=>k,k=>alias.Prefix+"."+k,StringComparer.Ordinal);
                string? Present(string suffix)=>file.Tensors.ContainsKey(alias.Prefix+suffix)?alias.Prefix+suffix:null;
                var binding=new LoraBinding(alias.Prefix,target,null,null,null,Present(".alpha"),Present(".dora_scale"),
                    Lokr:new System.Collections.ObjectModel.ReadOnlyDictionary<string,string>(keys));
                Validate(file,binding,mode);foreach(string key in FactorKeys(binding))claimed.Add(key);if(binding.Alpha is not null)claimed.Add(binding.Alpha);
                var identity=(target.Component,target.Weight);
                if(selected.TryGetValue(identity,out var prior)&&prior.Prefix!=alias.Prefix)shadowed.Add(prior.Prefix);
                selected[identity]=binding;
            }
            void Difference(string suffix,bool bias=false)
            {
                string key=alias.Prefix+suffix;
                if(!file.Tensors.ContainsKey(key))return;
                if(bias&&!target.Weight.EndsWith(".weight",StringComparison.Ordinal))
                    throw new InvalidDataException("A bias difference alias must identify its module weight.");
                var destination=bias?target with{Weight=target.Weight[..^7]+".bias",Shape=Array.AsReadOnly(new[]{target.Shape[0]})}:target;
                var binding=new LoraBinding(alias.Prefix,destination,null,null,null,null,null,key);
                Validate(file,binding,mode);claimed.Add(key);
                var identity=(destination.Component,destination.Weight);
                if(selected.TryGetValue(identity,out var prior)&&prior.Prefix!=alias.Prefix)shadowed.Add(prior.Prefix);
                selected[identity]=binding;
            }
            // Frozen load_lora ordering: norm differences, then explicit weight/bias differences.
            if(file.Tensors.ContainsKey(alias.Prefix+".w_norm")){Difference(".w_norm");Difference(".b_norm",true);}
            Difference(".diff");Difference(".diff_b",true);
        }
        var unclaimed=file.Tensors.Keys.Where(k=>!claimed.Contains(k)).Order(StringComparer.Ordinal).ToArray();
        if(!allowUnclaimedKeys&&unclaimed.Length>0)throw new NotSupportedException($"Unclaimed LoRA tensors ({unclaimed.Length}): {string.Join(", ",unclaimed.Take(8))}. No adapter is loaded.");
        if(selected.Count==0)throw new NotSupportedException("No adapter tensors match the supplied target aliases.");
        long resident=0;
        foreach(var binding in selected.Values)
            foreach(string key in FactorKeys(binding))resident=checked(resident+Elements(file.Tensors[key].Shape)*4);
        if(resident>maxResidentFactorBytes)throw new NotSupportedException("LoRA resident factors exceed the configured allowance; read/conversion temporaries are additional.");
        cancellationToken.ThrowIfCancellationRequested();
        return new(file,selected.Values.ToArray(),unclaimed,shadowed.ToArray(),components.ToArray(),resident,mode);
    }

    public static LoraAdapterSet Load(ILoraTensorSource file,LoraLoadPlan plan,
        IReadOnlyDictionary<string,double>? componentStrengths=null,CancellationToken cancellationToken=default)
    {
        ArgumentNullException.ThrowIfNull(file);ArgumentNullException.ThrowIfNull(plan);
        if(!ReferenceEquals(file,plan.Source))throw new ArgumentException("Load must use the same tensor source as inspection.",nameof(file));
        cancellationToken.ThrowIfCancellationRequested();
        var strengths=componentStrengths is null?new Dictionary<string,double>():new Dictionary<string,double>(componentStrengths,StringComparer.Ordinal);
        foreach(var pair in strengths)
            if(!double.IsFinite(pair.Value)||!plan.Components.Contains(pair.Key,StringComparer.Ordinal))throw new ArgumentException("Strengths must be finite and identify a declared component.",nameof(componentStrengths));
        NativeRuntimeBootstrap.Initialize();using var noGrad=no_grad();
        var patches=new Dictionary<(string,string),LoraWeightPatch>();
        try
        {
            foreach(var binding in plan.Bindings)
            {
                cancellationToken.ThrowIfCancellationRequested();using var scope=NewDisposeScope();
                if(binding.Difference is { } difference)
                {
                    var snapshot=LoraWeightPatch.FromDifference(file.ReadTensor(difference,cancellationToken),strengths.GetValueOrDefault(binding.Target.Component,1.0));
                    patches.Add((binding.Target.Component,binding.Target.Weight),snapshot);continue;
                }
                var mid=binding.Mid is null?null:file.ReadTensor(binding.Mid,cancellationToken);
                var dora=binding.Dora is null?null:file.ReadTensor(binding.Dora,cancellationToken);
                double? alpha=null;
                if(binding.Alpha is not null)
                {
                    var value=file.ReadTensor(binding.Alpha,cancellationToken).to_type(ScalarType.Float64);alpha=value.item<double>();
                    if(!double.IsFinite(alpha.Value))throw new InvalidDataException($"Nonfinite LoRA alpha '{binding.Alpha}'.");
                }
                double strength=strengths.GetValueOrDefault(binding.Target.Component,1.0);
                if(binding.Lokr is { } lokr)
                {
                    patches.Add((binding.Target.Component,binding.Target.Weight),LoraWeightPatch.FromLokr(
                        lokr.ToDictionary(p=>p.Key,p=>file.ReadTensor(p.Value,cancellationToken),StringComparer.Ordinal),strength,alpha,dora));
                    continue;
                }
                var up=file.ReadTensor(binding.Up!,cancellationToken);var down=file.ReadTensor(binding.Down!,cancellationToken);
                var patch=binding.Loha is { } loha
                    ?LoraWeightPatch.FromLoha(up,down,file.ReadTensor(loha.W2A,cancellationToken),file.ReadTensor(loha.W2B,cancellationToken),strength,alpha,
                        loha.T1 is null?null:file.ReadTensor(loha.T1,cancellationToken),loha.T2 is null?null:file.ReadTensor(loha.T2,cancellationToken),dora)
                    :new LoraWeightPatch(up,down,strength,alpha,mid,dora);
                patches.Add((binding.Target.Component,binding.Target.Weight),patch);
            }
            cancellationToken.ThrowIfCancellationRequested();var result=new LoraAdapterSet(plan,patches);patches=[];return result;
        }
        finally{foreach(var patch in patches.Values)patch.Dispose();}
    }

    private static IEnumerable<string> FactorKeys(LoraBinding binding)
    {
        if(binding.Difference is { } diff){yield return diff;yield break;}
        if(binding.Lokr is { } lokr){foreach(string key in lokr.Values)yield return key;}
        else {yield return binding.Up!;yield return binding.Down!;}
        if(binding.Loha is { } loha){yield return loha.W2A;yield return loha.W2B;if(loha.T1 is not null)yield return loha.T1;if(loha.T2 is not null)yield return loha.T2;}
        if(binding.Mid is not null)yield return binding.Mid;if(binding.Dora is not null)yield return binding.Dora;
    }
    private static long Elements(IReadOnlyList<long> shape)=>shape.Aggregate(1L,(n,d)=>checked(n*d));
    private static void Validate(ILoraTensorSource file,LoraBinding binding,LoraLoadMode mode)
    {
        foreach(string key in FactorKeys(binding))
        {
            var info=file.Tensors[key];
            if(info.DType is not("F32" or "F16" or "BF16")||info.Shape.Any(v=>v<=0)||info.End-info.Start>int.MaxValue)
                throw new InvalidDataException($"Unsupported LoRA factor dtype, shape or size: '{key}'.");
        }
        if(binding.Difference is { } diff)
        {
            if(!file.Tensors[diff].Shape.SequenceEqual(binding.Target.Shape))throw new InvalidDataException("Additive adapter shape differs from its target.");
            return;
        }
        if(binding.Target.Shape.Count<2)throw new InvalidDataException("LoRA factors require a target with at least two dimensions.");
        if(binding.Lokr is { } lokr)
        {
            try
            {
                var factors=lokr.ToDictionary(p=>p.Key,p=>file.Tensors[p.Value].Shape,StringComparer.Ordinal);
                if(mode==LoraLoadMode.Bypass)LokrBypassGeometry.ValidateTarget(factors,binding.Target.Shape);
                else if(Elements(LokrMath.ReconstructedShape(factors))!=Elements(binding.Target.Shape))throw new InvalidDataException($"LoKr factors cannot reshape to '{binding.Target.Weight}'.");
            }
            catch(ArgumentException error){throw new InvalidDataException("LoKr factor geometry is incompatible.",error);}
        }
        else if(binding.Loha is { } loha)
        {
            var up=file.Tensors[binding.Up!].Shape;var down=file.Tensors[binding.Down!].Shape;
            long[] shape;
            try {shape=LohaMath.ReconstructedShape(up,down,file.Tensors[loha.W2A].Shape,file.Tensors[loha.W2B].Shape,
                loha.T1 is null?null:file.Tensors[loha.T1].Shape,loha.T2 is null?null:file.Tensors[loha.T2].Shape);}
            catch(ArgumentException error){throw new InvalidDataException("LoHa factor geometry is incompatible.",error);}
            if(mode==LoraLoadMode.Bypass)
            {
                try{AdapterBypassGeometry.Loha(shape,binding.Target.Shape);}
                catch(ArgumentException error){throw new InvalidDataException("LoHa bypass geometry is incompatible.",error);}
            }
            else if(Elements(shape)!=Elements(binding.Target.Shape))throw new InvalidDataException($"LoHa factors cannot reshape to '{binding.Target.Weight}'.");
        }
        else
        {
            var up=file.Tensors[binding.Up!].Shape;var down=file.Tensors[binding.Down!].Shape;
            if(mode==LoraLoadMode.Bypass)
            {
                try{AdapterBypassGeometry.Lora(up,down,binding.Mid is null?null:file.Tensors[binding.Mid].Shape,binding.Target.Shape);}
                catch(ArgumentException error){throw new InvalidDataException("LoRA bypass geometry is incompatible.",error);}
            }
            else
            {
            if(up.Count<2||down.Count<2)throw new InvalidDataException("LoRA up/down factors require at least two dimensions.");
            long columns=Elements(down)/down[0];
            if(binding.Mid is not null)
            {
                var mid=file.Tensors[binding.Mid].Shape;
                if(mid.Count!=4||Elements(down)/down[1]!=mid[1]||mid[0]!=down[0])throw new InvalidDataException("LoCon mid/down shapes are incompatible.");
                columns=checked(down[1]*mid[2]*mid[3]);
            }
            if(Elements(up)/up[0]!=down[0]||checked(up[0]*columns)!=Elements(binding.Target.Shape))throw new InvalidDataException($"LoRA factors cannot reshape to '{binding.Target.Weight}'.");
            }
        }
        if(binding.Alpha is not null)
        {
            var alpha=file.Tensors[binding.Alpha];
            if(!SafeTensorFile.SupportsTensorDType(alpha.DType)||Elements(alpha.Shape)!=1)throw new InvalidDataException("LoRA alpha requires a single supported scalar value.");
        }
        if(binding.Dora is not null&&mode!=LoraLoadMode.Bypass)
        {
            var shape=file.Tensors[binding.Dora].Shape;var target=binding.Target.Shape;
            if(shape.Count<1||shape.Count>target.Count)throw new InvalidDataException("DoRA scale cannot broadcast to the target weight.");
            for(int i=1;i<=shape.Count;i++)if(shape[^i]!=1&&shape[^i]!=target[^i])throw new InvalidDataException("DoRA scale cannot broadcast to the target weight.");
        }
    }
}

/// <summary>Owns loaded adapter snapshots independently of the source file. No native model weights are owned here.</summary>
public sealed class LoraAdapterSet : IDisposable
{
    private readonly object gate=new();
    private Dictionary<(string Component,string Weight),LoraWeightPatch>? patches;
    public LoraLoadPlan Plan { get; }
    internal LoraAdapterSet(LoraLoadPlan plan,Dictionary<(string,string),LoraWeightPatch> patches){Plan=plan;this.patches=patches;}
    public Tensor Apply(string component,string weight,Tensor source,CancellationToken cancellationToken=default)
    {
        RequireWeightPlan();
        cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(source);LoraWeightPatch owned;
        var binding=Plan.Bindings.Single(b=>b.Target.Component==component&&b.Target.Weight==weight);
        if(!binding.Target.Shape.SequenceEqual(source.shape))throw new InvalidDataException("The target weight shape differs from the inspected LoRA plan.");
        lock(gate){ObjectDisposedException.ThrowIf(patches is null,this);owned=patches[(component,weight)].Retain();}
        using(owned)return owned.Apply(source,cancellationToken);
    }
    public SdUnet ApplyTo(SdUnet model,string component="model",long maxPatchedWeightBytes=512L*1024*1024,CancellationToken cancellationToken=default)
    {RequireWeightPlan();cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(model);ValidateTargets(component,UnetWeightSchema.Describe(model.Config));return WithPatches(component,p=>model.WithLora(p,maxPatchedWeightBytes,cancellationToken),cancellationToken);}
    public SdUnet ApplyBypassTo(SdUnet model,string component="model",long maxPatchedWeightBytes=512L*1024*1024,CancellationToken cancellationToken=default)
    {cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(model);ValidateTargets(component,UnetWeightSchema.Describe(model.Config));return WithPatches(component,p=>model.WithBypassLora(p,maxPatchedWeightBytes,cancellationToken),cancellationToken);}
    public ComfyClipEncoder ApplyTo(ComfyClipEncoder clip,string component="clip",long maxPatchedWeightBytes=512L*1024*1024,CancellationToken cancellationToken=default)
    {RequireWeightPlan();cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(clip);ValidateTargets(component,ClipWeightSchema.Describe(clip.Config));return WithPatches(component,p=>clip.WithLora(p,maxPatchedWeightBytes,cancellationToken),cancellationToken);}
    public Tensor ApplyBypass(string component,string weight,Tensor input,Tensor baseOutput,long stride=1,long padding=0,CancellationToken cancellationToken=default)
    {
        cancellationToken.ThrowIfCancellationRequested();ArgumentNullException.ThrowIfNull(input);ArgumentNullException.ThrowIfNull(baseOutput);
        var binding=Plan.Bindings.Single(b=>b.Target.Component==component&&b.Target.Weight==weight);
        var shape=binding.Target.Shape;
        if(shape.Count<2||input.dim()<1||baseOutput.dim()<1||
            shape.Count>2&&(input.dim()!=shape.Count||baseOutput.dim()!=shape.Count)||
            input.shape[shape.Count==2?^1:1]!=shape[1]||baseOutput.shape[shape.Count==2?^1:1]!=shape[0])
            throw new InvalidDataException("Bypass activations do not match the inspected module channels/rank.");
        LoraWeightPatch owned;lock(gate){ObjectDisposedException.ThrowIf(patches is null,this);owned=patches[(component,weight)].Retain();}
        using(owned)return owned.ApplyBypass(input,baseOutput,shape.Count==2?null:shape.Skip(2).ToArray(),stride,padding);
    }
    private void RequireWeightPlan()
    {
        if(Plan.Mode!=LoraLoadMode.Weights)throw new InvalidOperationException("A bypass-inspected adapter cannot be baked; inspect again in weight mode.");
    }
    private void ValidateTargets(string component,IReadOnlyDictionary<string,IReadOnlyList<long>> schema)
    {
        foreach(var binding in Plan.Bindings.Where(b=>b.Target.Component==component))
            if(!schema.TryGetValue(binding.Target.Weight,out var shape)||!shape.SequenceEqual(binding.Target.Shape))
                throw new InvalidDataException($"Model weight '{binding.Target.Weight}' does not match the inspected LoRA plan.");
    }
    private T WithPatches<T>(string component,Func<IReadOnlyDictionary<string,LoraWeightPatch>,T> apply,CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();ArgumentException.ThrowIfNullOrWhiteSpace(component);
        if(!Plan.Components.Contains(component,StringComparer.Ordinal))throw new ArgumentException("Component was not declared in the inspected LoRA aliases.",nameof(component));
        var selected=new Dictionary<string,LoraWeightPatch>(StringComparer.Ordinal);
        try
        {
            lock(gate)
            {
                ObjectDisposedException.ThrowIf(patches is null,this);
                foreach(var (target,patch) in patches)
                    if(target.Component==component)selected.Add(target.Weight,patch.Retain());
            }
            return apply(selected);
        }
        finally{foreach(var patch in selected.Values)patch.Dispose();}
    }
    public void Dispose()
    {
        Dictionary<(string,string),LoraWeightPatch>? owned;
        lock(gate){owned=patches;patches=null;}
        if(owned is not null)foreach(var patch in owned.Values)patch.Dispose();
    }
}
