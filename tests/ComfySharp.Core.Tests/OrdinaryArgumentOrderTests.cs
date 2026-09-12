using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class OrdinaryArgumentOrderTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task PromptOrderReachesTheBodyWithoutReorderingDependenciesOrRequestingUnusedLazyInputs(bool v3, bool inputIsList)
    {
        var registry = new NodeRegistry(); var resolution = new List<string>();
        foreach (string id in new[] { "headSource", "tailSource", "lazySource", "unusedSource" })
            registry.Register(new Node(new(id, id, "test", [], [new("*", IsList: true)]), (context, _) =>
            {
                resolution.Add(id);
                return new([context.List([context.Json(id + "1"), context.Json(id + "2")])]);
            }));
        var bodyOrders = new List<string[]>();
        registry.Register(new Node(new("Ordered", "Ordered", "test",
            [new("head", "*"), new("omitted", "*", Required: false), new("lazy", "*", Lazy: true),
             new("unused", "*", Lazy: true), new("tail", "*")], [new("*")], InputIsList: inputIsList, V3ObjectInfo: v3),
            (context, inputs) =>
            {
                bodyOrders.Add(inputs.Keys.ToArray());
                return new([context.Map(inputs)]);
            }, resolved =>
            {
                Assert.Equal(new[] { "head", "tail" }, resolved.Keys);
                return ["lazy"];
            }));
        var prompt = new JsonObject();
        foreach (string id in new[] { "unusedSource", "lazySource", "tailSource", "headSource" })
            prompt[id] = new JsonObject { ["class_type"] = id, ["inputs"] = new JsonObject() };
        prompt["target"] = new JsonObject
        {
            ["class_type"] = "Ordered", ["inputs"] = new JsonObject
            {
                ["lazy"] = new JsonArray("lazySource", 0), ["tail"] = new JsonArray("tailSource", 0),
                ["head"] = new JsonArray("headSource", 0), ["unused"] = new JsonArray("unusedSource", 0)
            }
        };
        string before = prompt.ToJsonString();
        using var result = await new EngineService(registry).ExecuteValuesAsync(prompt, ["target"]);
        Assert.Equal("success", result.Status); Assert.Empty(result.Diagnostics);
        Assert.Equal(new[] { "headSource", "tailSource", "lazySource" }, resolution);
        Assert.Equal(inputIsList ? 1 : 2, bodyOrders.Count);
        Assert.All(bodyOrders, order => Assert.Equal(new[] { "lazy", "tail", "head" }, order));
        var outputs = result.Outputs["target"][0]; Assert.Equal(bodyOrders.Count, outputs.Count);
        for (int i = 0; i < outputs.Count; i++)
        {
            Assert.Equal(new[] { "lazy", "tail", "head" }, outputs[i].Properties.Keys);
            if (inputIsList)
                Assert.Equal(new[] { "lazySource1", "lazySource2" }, outputs[i].Properties["lazy"].Items.Select(v => v.ToJson()!.GetValue<string>()));
            else Assert.Equal("lazySource" + (i + 1), outputs[i].Properties["lazy"].ToJson()!.GetValue<string>());
        }
        Assert.Equal(before, prompt.ToJsonString());
    }

    private sealed class Node(NodeSchema schema,
        Func<RuntimeNodeContext, IReadOnlyDictionary<string, RuntimeValue>, NodeExecutionOutput> execute,
        Func<IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>>, IReadOnlyCollection<string>>? lazy = null) : IRuntimeNode
    {
        public NodeSchema Schema => schema;
        public IReadOnlyCollection<string> GetRequiredLazyInputs(IReadOnlyDictionary<string, IReadOnlyList<RuntimeValue>> resolvedInputs) => lazy?.Invoke(resolvedInputs) ?? [];
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken) => ValueTask.FromResult(execute(context, inputs));
    }
}
