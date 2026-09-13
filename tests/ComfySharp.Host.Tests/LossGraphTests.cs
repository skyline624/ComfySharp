using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Host;
using ComfySharp.Media;
using ComfySharp.Nodes.Tensor;
using ComfySharp.Storage;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using static ComfySharp.Testing.PngFixtureReader;

namespace ComfySharp.Host.Tests;

public sealed class LossGraphTests
{
    [Theory] [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)] [InlineData(6)] [InlineData(7)]
    public void Layout_and_failure_cases_follow_actual_frozen_node(int index)
    {
        using var stream=GetType().Assembly.GetManifestResourceStream("ComfySharp.Host.Tests.Fixtures.loss-graph.reference.json")!;
        using var buffer=new MemoryStream();stream.CopyTo(buffer);byte[] bytes=buffer.ToArray();
        Assert.Equal("cae0f108105347b04375684a3f7f868495f909e7bedb0357517e30375d3b49e2",Convert.ToHexStringLower(SHA256.HashData(bytes)));
        using var json=JsonDocument.Parse(bytes);var row=json.RootElement.GetProperty("cases")[index];
        var losses=row.GetProperty("losses").EnumerateArray().Select(v=>v.GetDouble()).ToArray();
        if(row.TryGetProperty("error",out var error))
        {
            if(error.GetString()=="ValueError")Assert.Throws<ArgumentException>(()=>NativeLossGraphRenderer.Describe(losses));
            else Assert.Throws<ArithmeticException>(()=>NativeLossGraphRenderer.Describe(losses));
            return;
        }
        var layout=NativeLossGraphRenderer.Describe(losses);var calls=row.GetProperty("calls").EnumerateArray().ToArray();
        var blue=calls.Where(c=>c.GetProperty("kind").GetString()=="line"&&c.GetProperty("fill").GetString()=="blue").ToArray();
        Assert.Equal(layout.Points.Count-1,blue.Length);
        for(int i=0;i<blue.Length;i++)
        {
            Assert.Equal(2,blue[i].GetProperty("width").GetInt32());var points=blue[i].GetProperty("points");
            Assert.Equal(new[]{layout.Points[i].X,layout.Points[i].Y},points[0].EnumerateArray().Select(v=>v.GetInt32()));
            Assert.Equal(new[]{layout.Points[i+1].X,layout.Points[i+1].Y},points[1].EnumerateArray().Select(v=>v.GetInt32()));
        }
        var text=calls.Where(c=>c.GetProperty("kind").GetString()=="text").ToArray();Assert.Equal(layout.Labels.Count,text.Length);
        for(int i=0;i<text.Length;i++)
        {
            Assert.Equal(layout.Labels[i].Text,text[i].GetProperty("text").GetString());
            Assert.Equal(new[]{layout.Labels[i].X,layout.Labels[i].Y},text[i].GetProperty("point").EnumerateArray().Select(v=>v.GetInt32()));
        }
        var rendered=NativeLossGraphRenderer.Render(losses);Assert.Equal(840,rendered.Width);Assert.Equal(520,rendered.Height);Assert.Equal(1,rendered.Frames);
        Assert.Null(rendered.Alpha);Assert.Equal(840*520*3,rendered.Rgb.Length);
        Assert.All(rendered.Rgb,v=>Assert.InRange(v,0,1));
        Assert.Equal(new[]{0f,0f,0f},rendered.Rgb.AsSpan((200*840+40)*3,3).ToArray());
        Assert.Equal(new[]{1f,1f,1f},rendered.Rgb.AsSpan((519*840+839)*3,3).ToArray());
        Assert.Contains(Enumerable.Range(0,840*520),p=>rendered.Rgb[p*3]==0&&rendered.Rgb[p*3+1]==0&&rendered.Rgb[p*3+2]==1);
    }
    private sealed class MemoryStore : IImageFileStore
    {
        public List<byte[]> Pngs {get;}=[];
        public IReadOnlyList<string> PrepareDirectory(string type,string subfolder,CancellationToken cancellationToken=default)=>[];
        public ValueTask WriteAsync(ImageFileDescriptor file,ReadOnlyMemory<byte> png,CancellationToken cancellationToken=default)
        {cancellationToken.ThrowIfCancellationRequested();Assert.Equal("temp",file.Type);Assert.StartsWith("ComfyUI_temp_",file.Filename);Pngs.Add(png.ToArray());return ValueTask.CompletedTask;}
    }
    [Theory] [InlineData(false)] [InlineData(true)]
    public async Task Typed_history_survives_producer_and_preview_preserves_metadata_policy(bool disableMetadata)
    {
        using var consumer=new RuntimeNodeContext();RuntimeValue loss;
        using(var producer=new RuntimeNodeContext())loss=consumer.Retain(TrainingNodeValues.CaptureLosses(producer,new[]{1f,.5f,.2f}));
        var store=new MemoryStore();var node=new LossGraphNode(NativeLossGraphRenderer.Render,store,disableMetadata);
        using var outputContext=new RuntimeNodeContext(new(JsonNode.Parse("{\"n\":1}")!.AsObject(),JsonNode.Parse("{\"workflow\":{\"version\":1}}")));
        var inputs=new Dictionary<string,RuntimeValue>{{"loss",loss},{"filename_prefix",outputContext.Json(JsonValue.Create("../../unused"))},
            {"prompt",outputContext.Json(JsonNode.Parse("{\"n\":1}"))},{"extra_pnginfo",outputContext.Json(JsonNode.Parse("{\"workflow\":{\"version\":1}}"))}};
        var result=await node.ExecuteAsync(outputContext,inputs,default);Assert.Empty(result.Result);
        Assert.False(result.Ui!["animated"]![0]!.GetValue<bool>());Assert.Single(result.Ui["images"]!.AsArray());
        var png=Read(Assert.Single(store.Pngs));Assert.Equal(840,png.Width);Assert.Equal(520,png.Height);
        Assert.Equal(disableMetadata?Array.Empty<string>():new[]{"prompt","workflow"},png.Text.Select(p=>p.Keyword));
        await Assert.ThrowsAsync<OperationCanceledException>(async()=>await node.ExecuteAsync(outputContext,inputs,new(true)));Assert.Single(store.Pngs);
        Assert.Equal(3,loss.Properties["loss"].Items.Count);
        Assert.Throws<ArgumentException>(()=>NativeLossGraphRenderer.Render(new[]{double.NaN,1d}));
    }
    private sealed class LossSource : IRuntimeNode
    {
        public NodeSchema Schema {get;}=new("TestTrainingLoss","Test Training Loss","test",[],[new("LOSS_MAP")]);
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,IReadOnlyDictionary<string,RuntimeValue> inputs,CancellationToken token)
        {
            float[] losses=[1.1856256f,.9327829f,.83137566f,.7629695f];
            if(Environment.GetEnvironmentVariable("COMFYSHARP_LOSS_GRAPH_REPORT") is { } report)
            {
                using var json=JsonDocument.Parse(File.ReadAllBytes(report));
                losses=json.RootElement.GetProperty("steps").EnumerateArray().Select(s=>s.GetProperty("loss").GetSingle()).ToArray();
            }
            return ValueTask.FromResult(new NodeExecutionOutput([TrainingNodeValues.CaptureLosses(context,losses)]));
        }
    }
    [Fact]
    public async Task Host_submits_loss_graph_and_serves_preview_with_workflow_metadata()
    {
        string root=Path.Combine(Path.GetTempPath(),"comfysharp-loss-graph-"+Guid.NewGuid().ToString("N"));var store=new ImageFileStore(root);
        try
        {
            var registry=new NodeRegistry();registry.Register(new LossSource());registry.Register(new LossGraphNode(NativeLossGraphRenderer.Render,store));
            await using var factory=new WebApplicationFactory<Program>().WithWebHostBuilder(builder=>builder.ConfigureServices(services=>
            {services.AddSingleton(new EngineService(registry));services.AddSingleton(store);services.AddSingleton<IImageFileStore>(store);}));
            using var client=factory.CreateClient();using var timeout=new CancellationTokenSource(TimeSpan.FromSeconds(30));
            var body=JsonNode.Parse("""
                {"prompt":{"loss":{"class_type":"TestTrainingLoss","inputs":{}},"graph":{"class_type":"LossGraphNode","inputs":{"loss":["loss",0],"filename_prefix":"unused"}}},"extra_data":{"extra_pnginfo":{"workflow":{"version":1,"custom":"preserved"}}}}
                """)!;
            using var response=await client.PostAsJsonAsync("/prompt",body,timeout.Token);response.EnsureSuccessStatusCode();
            string id=(await response.Content.ReadFromJsonAsync<JsonObject>(timeout.Token))!["prompt_id"]!.GetValue<string>();JsonNode? history=null;
            while(history is null)
            {
                history=(await client.GetFromJsonAsync<JsonObject>("/history",timeout.Token))![id];
                if(history is null)await Task.Delay(10,timeout.Token);
            }
            Assert.True(history["status"]!["completed"]!.GetValue<bool>(),history.ToJsonString());
            var output=history["outputs"]!["graph"]!;Assert.False(output["animated"]![0]!.GetValue<bool>());
            var file=Assert.Single(output["images"]!.AsArray())!.Deserialize<ImageFileDescriptor>()!;Assert.Equal("temp",file.Type);Assert.Equal("",file.Subfolder);
            byte[] bytes=await client.GetByteArrayAsync("/view?type=temp&filename="+Uri.EscapeDataString(file.Filename),timeout.Token);var png=Read(bytes);
            Assert.Equal(840,png.Width);Assert.Equal(520,png.Height);Assert.Equal(new[]{"prompt","workflow"},png.Text.Select(p=>p.Keyword));
            Assert.True(JsonNode.DeepEquals(body["prompt"],JsonNode.Parse(png.Text[0].Text)));
            Assert.True(JsonNode.DeepEquals(body["extra_data"]!["extra_pnginfo"]!["workflow"],JsonNode.Parse(png.Text[1].Text)));
            Assert.Single(Directory.GetFiles(root,"*.png",SearchOption.AllDirectories));
            if(Environment.GetEnvironmentVariable("COMFYSHARP_LOSS_GRAPH_PREVIEW") is { } preview)
            {using var outputFile=new FileStream(preview,FileMode.CreateNew,FileAccess.Write);outputFile.Write(bytes);}
        }
        finally{if(Directory.Exists(root))Directory.Delete(root,true);}
    }
}
