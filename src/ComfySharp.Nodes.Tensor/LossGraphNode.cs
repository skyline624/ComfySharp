using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Inference;
using static TorchSharp.torch;

namespace ComfySharp.Nodes.Tensor;

public sealed class LossGraphNode(Func<IReadOnlyList<double>,CancellationToken,DecodedImageBatch> render,
    IImageFileStore store,bool disableMetadata=false,TimeProvider? time=null) : IRuntimeNode
{
    public static NodeSchema Description { get; } = new("LossGraphNode","Plot Loss Graph","model/training",
        [new("loss","LOSS_MAP",Options:new(){["tooltip"]="Loss map from training node."}),
         new("filename_prefix","STRING",Options:new(){["default"]="loss_graph",["tooltip"]="Prefix for the saved loss graph image."})],
        [],OutputNode:true,Experimental:true,SearchAliases:["training chart","training visualization","plot loss"],
        PythonModule:"comfy_extras.nodes_train",V3HiddenInputs:["PROMPT","EXTRA_PNGINFO"],
        V3ObjectInfo:true,OmitEmptyOptionalInputs:true);
    public NodeSchema Schema=>Description;
    public async ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,IReadOnlyDictionary<string,RuntimeValue> inputs,CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var loss=inputs["loss"];double[] values;
        if(loss.Kind==RuntimeValueKind.Map&&loss.Properties.TryGetValue("loss",out var sequence)&&sequence.Kind==RuntimeValueKind.List)
            values=sequence.Items.Select(v=>v.ToJson()!.Deserialize<double>()).ToArray();
        else if(loss.Kind==RuntimeValueKind.Json&&loss.ToJson() is JsonObject obj&&obj["loss"] is JsonArray array)
            values=array.Select(v=>v!.Deserialize<double>()).ToArray();
        else throw new ArgumentException("LOSS_MAP requires a loss sequence.");
        var image=render(values,cancellationToken);NativeRuntimeBootstrap.Initialize();
        using var scope=NewDisposeScope();using var local=new RuntimeNodeContext();
        var pixels=tensor(image.Rgb,new long[]{image.Frames,image.Height,image.Width,3});
        var previewInputs=new Dictionary<string,RuntimeValue>{{"images",local.Own(pixels)}};pixels.DetachFromDisposeScope();
        // V3 ImageSaveHelper uses truthy hidden metadata. filename_prefix is unused by the source node.
        foreach(var (key,value) in new[]{("prompt",(JsonNode?)context.Hidden.Prompt),("extra_pnginfo",context.Hidden.ExtraPngInfo)})
            if(PythonValues.Truth(value))previewInputs[key]=local.Json(value);
        var result=await ImageFileNodes.CreatePreview(store,time,disableMetadata).ExecuteAsync(local,previewInputs,cancellationToken);
        var ui=(JsonObject)result.Ui!.DeepClone();ui["animated"]=new JsonArray(false);
        return new([],ui);
    }
}
