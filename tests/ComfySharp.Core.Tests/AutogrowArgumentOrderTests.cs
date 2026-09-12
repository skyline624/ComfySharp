using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class AutogrowArgumentOrderTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task PromptArgumentOrderDoesNotChangeProducerResolutionOrDeclaredGroupOrder(bool names, bool inputIsList)
    {
        AutogrowTemplate template = names
            ? new AutogrowNamesTemplate(new("value", "*"), ["v0", "v1"], 0)
            : new AutogrowPrefixTemplate(new("value", "*"), "v", 0, 2);
        var registry = new NodeRegistry();
        var resolution = new List<string>();
        foreach (string id in new[] { "headSource", "firstSource", "secondSource", "tailSource" })
        {
            registry.Register(new Node(new(id, id, "test", [], [new("*", IsList: true)]), (context, _) =>
            {
                resolution.Add(id);
                return new([context.List([context.Json(id + "1"), context.Json(id + "2")])]);
            }));
        }
        var observedRoots = new List<string[]>();
        var observedMembers = new List<string[]>();
        registry.Register(new Node(new("Ordered", "Ordered", "test",
            [new("head", "*"), new("omitted", "*", Required: false),
             new("values", "COMFY_AUTOGROW_V3", Autogrow: template), new("tail", "*")],
            [new("*")], InputIsList: inputIsList), (context, inputs) =>
        {
            observedRoots.Add(inputs.Keys.ToArray());
            observedMembers.Add(inputs["values"].Properties.Keys.ToArray());
            return new([context.Map(inputs)]);
        }));
        var prompt = new JsonObject();
        foreach (string id in new[] { "tailSource", "secondSource", "headSource", "firstSource" })
            prompt[id] = new JsonObject { ["class_type"] = id, ["inputs"] = new JsonObject() };
        prompt["target"] = new JsonObject
        {
            ["class_type"] = "Ordered",
            ["inputs"] = new JsonObject
            {
                ["tail"] = new JsonArray("tailSource", 0),
                ["values.v1"] = new JsonArray("secondSource", 0),
                ["head"] = new JsonArray("headSource", 0),
                ["values.v0"] = new JsonArray("firstSource", 0)
            }
        };
        string before = prompt.ToJsonString();
        using var result = await new EngineService(registry).ExecuteValuesAsync(prompt, ["target"]);
        Assert.Equal("success", result.Status);
        Assert.Empty(result.Diagnostics);
        Assert.Equal(new[] { "headSource", "firstSource", "secondSource", "tailSource" }, resolution);
        Assert.Equal(inputIsList ? 1 : 2, observedRoots.Count);
        Assert.Equal(observedRoots.Count, observedMembers.Count);
        Assert.All(observedRoots, order => Assert.Equal(new[] { "tail", "head", "values" }, order));
        Assert.All(observedMembers, order => Assert.Equal(new[] { "v0", "v1" }, order));
        var outputs = result.Outputs["target"][0];
        Assert.Equal(observedRoots.Count, outputs.Count);
        for (int i = 0; i < outputs.Count; i++)
        {
            Assert.Equal(new[] { "tail", "head", "values" }, outputs[i].Properties.Keys);
            Assert.Equal(new[] { "v0", "v1" }, outputs[i].Properties["values"].Properties.Keys);
            if (inputIsList)
                Assert.Equal(new[] { "headSource1", "headSource2" },
                    outputs[i].Properties["head"].Items.Select(v => v.ToJson()!.GetValue<string>()));
            else Assert.Equal("headSource" + (i + 1), outputs[i].Properties["head"].ToJson()!.GetValue<string>());
        }
        Assert.Equal(before, prompt.ToJsonString());
    }

    private sealed class Node(NodeSchema schema,
        Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, NodeExecutionOutput> execute) : IRuntimeNode
    {
        public NodeSchema Schema => schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) =>
            ValueTask.FromResult(execute(context, inputs));
    }
}
