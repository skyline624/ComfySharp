using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class V3HiddenContextTests
{
    private sealed class Capture(bool lists=false,IReadOnlyList<string>? hidden=null) : IRuntimeNode
    {
        public NodeSchema Schema {get;}=new("CaptureV3","Capture V3","test",[new("prompt","STRING")],[new("*")],OutputNode:true,
            InputIsList:lists,V3ObjectInfo:true,V3HiddenInputs:hidden??["PROMPT","EXTRA_PNGINFO","UNIQUE_ID"]);
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,IReadOnlyDictionary<string,RuntimeValue> inputs,CancellationToken token)
        {
            Assert.Equal(new[]{"prompt"},inputs.Keys);
            var value=lists?Assert.Single(inputs["prompt"].Items):inputs["prompt"];
            return ValueTask.FromResult(new NodeExecutionOutput([context.Json(new JsonObject{
                ["argument"]=value.ToJson(),["original"]=context.Hidden.Prompt?.DeepClone(),
                ["png"]=context.Hidden.ExtraPngInfo?.DeepClone(),["id"]=context.Hidden.UniqueId})]));
        }
    }
    [Theory] [InlineData(false)] [InlineData(true)]
    public async Task V3_metadata_stays_outside_argument_lists_and_refreshes_each_job(bool lists)
    {
        var registry=new NodeRegistry();registry.Register(new Capture(lists));var engine=new EngineService(registry);
        var info=registry.ToObjectInfo()["CaptureV3"]!;
        Assert.Equal("{\"prompt\":[\"PROMPT\"],\"extra_pnginfo\":[\"EXTRA_PNGINFO\"],\"unique_id\":[\"UNIQUE_ID\"]}",info["input"]!["hidden"]!.ToJsonString());
        var prompt=JsonNode.Parse("{\"outer:7\":{\"class_type\":\"CaptureV3\",\"inputs\":{\"prompt\":\"widget-value\"}}}")!.AsObject();
        foreach(string label in new[]{"first","second"})
        {
            var extra=new JsonObject{["extra_pnginfo"]=new JsonObject{["tag"]=label},["unrequested"]="private"};
            var result=await engine.ExecuteAsync(prompt,["outer:7"],extraData:extra);Assert.Equal("success",result.Status);
            var value=result.Outputs["outer:7"][0][0]!;Assert.Equal("widget-value",value["argument"]!.GetValue<string>());
            Assert.True(JsonNode.DeepEquals(prompt,value["original"]));Assert.Equal(label,value["png"]!["tag"]!.GetValue<string>());
            Assert.Equal("outer:7",value["id"]!.GetValue<string>());
        }
    }
    [Fact]
    public async Task Undeclared_metadata_is_absent_and_live_dynamic_prompt_is_not_replaced_by_JSON()
    {
        var registry=new NodeRegistry();registry.Register(new Capture(hidden:["UNIQUE_ID"]));
        var prompt=JsonNode.Parse("{\"1\":{\"class_type\":\"CaptureV3\",\"inputs\":{\"prompt\":\"visible\"}}}")!.AsObject();
        var result=await new EngineService(registry).ExecuteAsync(prompt,["1"],extraData:new JsonObject{["extra_pnginfo"]=new JsonObject{["secret"]="not-requested"}});
        Assert.Equal("success",result.Status);var value=result.Outputs["1"][0][0]!;Assert.Null(value["original"]);Assert.Null(value["png"]);Assert.Equal("1",value["id"]!.GetValue<string>());
        Assert.Throws<NotSupportedException>(()=>new NodeRegistry().Register(new Capture(hidden:["DYNPROMPT"])));
        Assert.Throws<NotSupportedException>(()=>new NodeRegistry().Register(new Capture(hidden:["PROMPT","PROMPT"])));
    }
}
