using ComfySharp.Inference;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

/// <summary>Diagnostic-only evaluation on unchanged parameter storage with inference leaf flags.
/// Callers must exclusively own the adapters during this synchronous evaluation.</summary>
internal static class FrozenAdapterEvaluation
{
    internal static Tensor Run(IEnumerable<TrainableWeightPatch> patches,Func<Tensor> evaluate)
    {
        ArgumentNullException.ThrowIfNull(patches);ArgumentNullException.ThrowIfNull(evaluate);
        var flags=patches.SelectMany(p=>p.Parameters).Distinct().Select(value=>(Value:value,Enabled:value.requires_grad)).ToArray();
        using var disabled=no_grad();
        try
        {
            foreach(var item in flags)item.Value.requires_grad_(false);
            return evaluate();
        }
        finally{foreach(var item in flags)item.Value.requires_grad_(item.Enabled);}
    }
}
