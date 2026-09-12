using System.Text.Json.Nodes;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class LoraWorkflowTests
{
    [Fact]
    public void Both_Lora_nodes_roundtrip_API_connections_names_and_independent_strengths()
    {
        const string json = """
        {"model":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"model.safetensors"}},
         "lora":{"class_type":"LoraLoader","inputs":{"model":["model",0],"clip":["model",1],"lora_name":"nested/style.safetensors","strength_model":0.75,"strength_clip":-0.5}},
         "next":{"class_type":"LoraLoaderModelOnly","inputs":{"model":["lora",0],"lora_name":"detail.safetensors","strength_model":1.2}}}
        """;
        var document = WorkflowDocument.Parse(ApiPromptImport.Parse(json).ToJson());
        var result = PromptCompiler.Compile(document); Assert.True(result.Success);
        foreach (var pair in JsonNode.Parse(json)!.AsObject()) Assert.True(JsonNode.DeepEquals(pair.Value!["inputs"], result.Prompt![pair.Key]!["inputs"]));
    }

    [Fact]
    public void Legacy_widget_order_does_not_swap_clip_strength_or_model_name()
    {
        var document = WorkflowDocument.Parse("""{"version":0.4,"nodes":[{"id":1,"type":"LoraLoader","widgets_values":["a.safetensors",0.6,-0.2]},{"id":2,"type":"LoraLoaderModelOnly","widgets_values":["b.safetensors",0.9]}],"links":[]}""");
        var result = PromptCompiler.Compile(document); Assert.True(result.Success);
        Assert.Equal(-0.2, result.Prompt!["1"]!["inputs"]!["strength_clip"]!.GetValue<double>());
        Assert.Equal(0.9, result.Prompt["2"]!["inputs"]!["strength_model"]!.GetValue<double>());
        Assert.Equal("b.safetensors", result.Prompt["2"]!["inputs"]!["lora_name"]!.GetValue<string>());
    }
}
