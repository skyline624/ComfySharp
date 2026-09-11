using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Nodes.Tensor;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class PreviewAnyReferenceTests
{
    private const string CorpusHash = "d2febc42079e6ba57fa0ae105d7479d19d4ac2e63e8e0a7f354fb5fb7ac76954";
    private static byte[] ReadCorpus()
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var stream = assembly.GetManifestResourceStream(assembly.GetManifestResourceNames()
            .Single(n => n.EndsWith("preview-any.cpu.json", StringComparison.Ordinal)))!;
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        return buffer.ToArray();
    }

    public static IEnumerable<object[]> CaseIds()
    {
        using var document = JsonDocument.Parse(ReadCorpus());
        return document.RootElement.GetProperty("cases").EnumerateArray()
            .Select(c => new object[] { c.GetProperty("id").GetString()! }).ToArray();
    }

    [Fact]
    public void CorpusIsPinnedAndDoesNotClaimModelQualification()
    {
        var bytes = ReadCorpus();
        Assert.Equal(CorpusHash, Convert.ToHexStringLower(SHA256.HashData(bytes)));
        using var document = JsonDocument.Parse(bytes);
        var root = document.RootElement;
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", root.GetProperty("backendCommit").GetString());
        Assert.Equal("2.13.0+cu130", root.GetProperty("laboratory").GetProperty("torch").GetString());
        Assert.False(root.GetProperty("laboratory").GetProperty("modelWeightsUsed").GetBoolean());
        Assert.Equal(58, root.GetProperty("cases").GetArrayLength());
        Assert.Equal(44, root.GetProperty("cases").EnumerateArray().Count(c => c.GetProperty("kind").GetString() == "tensor"));
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public async Task PreviewTextAndOutputSlotExactlyMatchFrozenPython(string id)
    {
        using var document = JsonDocument.Parse(ReadCorpus());
        var reference = document.RootElement.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("id").GetString() == id);
        using var scope = NewDisposeScope();
        using var context = new RuntimeNodeContext();
        RuntimeValue source;
        if (reference.GetProperty("kind").GetString() == "json")
            source = context.Json(JsonNode.Parse(reference.GetProperty("input").GetRawText()));
        else
        {
            NativeRuntimeBootstrap.Initialize();
            byte[] bytes = Convert.FromBase64String(reference.GetProperty("dataBase64").GetString()!);
            Assert.Equal(reference.GetProperty("dataSha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(bytes)));
            Assert.True(BitConverter.IsLittleEndian);
            var shape = reference.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64()).ToArray();
            var dtype = reference.GetProperty("dtype").GetString() switch
            {
                "float32" => ScalarType.Float32, "float64" => ScalarType.Float64,
                "float16" => ScalarType.Float16, "bfloat16" => ScalarType.BFloat16,
                "int8" => ScalarType.Int8, "uint8" => ScalarType.Byte, "int16" => ScalarType.Int16,
                "int32" => ScalarType.Int32, "int64" => ScalarType.Int64, "bool" => ScalarType.Bool,
                _ => throw new InvalidDataException("Unknown reference dtype.")
            };
            var tensor = empty(shape, dtype: dtype, device: CPU);
            Assert.Equal(bytes.LongLength, tensor.numel() * tensor.element_size());
            bytes.CopyTo(tensor.bytes);
            source = context.Own(tensor);
            tensor.DetachFromDisposeScope();
        }
        Assert.True(TensorNodes.CreateRegistry().TryGet("PreviewAny", out var node));
        var result = await ((IRuntimeNode)node).ExecuteAsync(context,
            new Dictionary<string, RuntimeValue> { ["source"] = source }, CancellationToken.None);
        string expected = reference.GetProperty("expectedText").GetString()!;
        Assert.Equal(expected, Assert.Single(result.Result).ToJson()!.GetValue<string>());
        Assert.Equal(expected, Assert.Single(result.Ui!["text"]!.AsArray())!.GetValue<string>());
    }
}
