using System.Text.Json.Nodes;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class InpaintWorkflowTests
{
    [Fact]
    public void Api_import_preserves_inpaint_connections_and_growth_through_document_roundtrip()
    {
        const string json = """
        {"load":{"class_type":"LoadImage","inputs":{"image":"input.png"}},
         "model":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"model.safetensors"}},
         "encode":{"class_type":"VAEEncodeForInpaint","inputs":{"pixels":["load",0],"vae":["model",2],"mask":["load",1],"grow_mask_by":6}},
         "mask":{"class_type":"SetLatentNoiseMask","inputs":{"samples":["encode",0],"mask":["load",1]}}}
        """;
        var document = WorkflowDocument.Parse(ApiPromptImport.Parse(json).ToJson());
        var result = PromptCompiler.Compile(document);
        Assert.True(result.Success, string.Join("; ",result.Diagnostics.Select(d=>d.Message)));
        foreach (var pair in JsonNode.Parse(json)!.AsObject())
            Assert.True(JsonNode.DeepEquals(pair.Value!["inputs"], result.Prompt![pair.Key]!["inputs"]));
    }

    [Fact]
    public void Legacy_inpaint_widget_has_one_serialized_growth_value()
    {
        var document = WorkflowDocument.Parse("""{"version":0.4,"nodes":[{"id":5,"type":"VAEEncodeForInpaint","widgets_values":[6]},{"id":9,"type":"SetLatentNoiseMask","widgets_values":[]}],"links":[]}""");
        var result = PromptCompiler.Compile(document); Assert.True(result.Success);
        Assert.Equal(6,result.Prompt!["5"]!["inputs"]!["grow_mask_by"]!.GetValue<int>());
        Assert.Empty(result.Prompt["9"]!["inputs"]!.AsObject());
    }
}
