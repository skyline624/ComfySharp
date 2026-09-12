using System.Buffers.Binary;
using System.Text;
using System.Text.Json.Nodes;
using ComfySharp.Testing;
using ComfySharp.Workflow;
using Xunit;

namespace ComfySharp.Workflow.Tests;

public sealed class PngWorkflowImportTests
{
    private const string Workflow = """{"version":1,"nodes":[{"id":"vendor:1","type":"UnknownNode","extra":{"retain":["猫",null,2]}}],"unknown":true}""";

    [Fact]
    public async Task Pinned_frontend_png_fixture_imports_its_versionless_graph_as_legacy()
    {
        byte[] png = PngMetadataFixture.Upstream();
        Assert.Equal(PngMetadataFixture.UpstreamSha256, Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(png)).ToLowerInvariant());
        using var stream = new MemoryStream(png); var metadata = await PngWorkflowImport.ReadMetadataAsync(stream);
        Assert.Equal("{\"1\":{\"class_type\":\"KSampler\",\"inputs\":{}}}", metadata.Text["prompt"]);
        var document = PngWorkflowImport.ReadWorkflow(metadata); var sampler = Assert.Single(document.Nodes);
        Assert.Equal("KSampler", sampler.Type); Assert.Equal("1", sampler.Id.Value); Assert.Equal(100, sampler.X); Assert.Equal(100, sampler.Y);
        var expected = JsonNode.Parse(metadata.Text["workflow"])!.AsObject(); expected["version"] = 0.4;
        Assert.True(JsonNode.DeepEquals(expected, document.Snapshot()));
    }

    [Theory]
    [InlineData("tEXt", 0)]
    [InlineData("comf", 0)]
    [InlineData("iTXt", 0)]
    [InlineData("iTXt", 1)]
    public async Task Recognized_text_encodings_preserve_workflow_and_unknown_fields(string type, byte compression)
    {
        using var stream = new MemoryStream(PngMetadataFixture.Build((type, PngMetadataFixture.Text(type, "workflow", Workflow, compression))));
        var metadata = await PngWorkflowImport.ReadMetadataAsync(stream); Assert.Empty(metadata.Warnings);
        var document = PngWorkflowImport.ReadWorkflow(metadata);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(Workflow), document.Snapshot()));
        Assert.Equal("vendor:1", Assert.Single(document.Nodes).Id.Value);
        Assert.True(stream.CanRead); // caller owns input stream
    }

    [Fact]
    public async Task Last_keyword_wins_workflow_has_priority_and_keyword_case_is_preserved()
    {
        using var stream = new MemoryStream(PngMetadataFixture.Build(
            ("tEXt", PngMetadataFixture.Text("tEXt", "workflow", "invalid previous value")),
            ("comf", PngMetadataFixture.Text("comf", "prompt", "not a workflow")),
            ("tEXt", PngMetadataFixture.Text("tEXt", "parameters", "not a workflow either")),
            ("iTXt", PngMetadataFixture.Text("iTXt", "workflow", Workflow, 1)),
            ("tEXt", PngMetadataFixture.Text("tEXt", "Workflow", "uppercase remains separate"))));
        var metadata = await PngWorkflowImport.ReadMetadataAsync(stream);
        Assert.Equal("uppercase remains separate", metadata.Text["Workflow"]);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(Workflow), PngWorkflowImport.ReadWorkflow(metadata).Snapshot()));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Bad_compressed_text_is_reported_and_does_not_replace_an_earlier_value(bool badMethod)
    {
        byte[] text = PngMetadataFixture.Text("iTXt", "workflow", "replacement", 1, badMethod ? (byte)99 : (byte)0);
        if (!badMethod) text[^1] ^= 1; // valid PNG CRC, damaged zlib checksum
        using var stream = new MemoryStream(PngMetadataFixture.Build(("tEXt", PngMetadataFixture.Text("tEXt", "workflow", Workflow)), ("iTXt", text)));
        var metadata = await PngWorkflowImport.ReadMetadataAsync(stream); Assert.Single(metadata.Warnings);
        Assert.Equal(Workflow, metadata.Text["workflow"]);
    }

    [Fact]
    public async Task Utf8_textdecoder_behavior_is_preserved_in_legacy_chunks()
    {
        using var stream = new MemoryStream(PngMetadataFixture.Build(("tEXt", Encoding.UTF8.GetBytes("note\0\uFEFF猫").Concat(new byte[] { 0xff }).ToArray())));
        var metadata = await PngWorkflowImport.ReadMetadataAsync(stream); Assert.Equal("猫�", metadata.Text["note"]);
    }

    [Theory]
    [InlineData("prompt")]
    [InlineData("parameters")]
    public async Task Prompt_only_and_parameters_only_imports_are_explicitly_unavailable(string key)
    {
        using var stream = new MemoryStream(PngMetadataFixture.Build(("tEXt", PngMetadataFixture.Text("tEXt", key, "{}"))));
        var metadata = await PngWorkflowImport.ReadMetadataAsync(stream);
        Assert.Throws<NotSupportedException>(() => PngWorkflowImport.ReadWorkflow(metadata));
    }

    [Fact]
    public async Task Ztxt_is_ignored_as_in_the_frozen_frontend()
    {
        using var stream = new MemoryStream(PngMetadataFixture.Build(("zTXt", PngMetadataFixture.Text("zTXt", "workflow", Workflow))));
        var metadata = await PngWorkflowImport.ReadMetadataAsync(stream); Assert.Empty(metadata.Text);
        Assert.Throws<InvalidDataException>(() => PngWorkflowImport.ReadWorkflow(metadata));
    }

    [Theory]
    [InlineData("signature")]
    [InlineData("crc")]
    [InlineData("truncated")]
    [InlineData("length")]
    [InlineData("unterminated")]
    public async Task Corrupt_or_unbounded_chunks_cannot_open_a_document(string corruption)
    {
        byte[] png = PngMetadataFixture.Build(("tEXt", PngMetadataFixture.Text("tEXt", "workflow", Workflow)));
        switch (corruption)
        {
            case "signature": png[7] ^= 1; break;
            case "crc": png[^1] ^= 1; break;
            case "truncated": png = png[..^4]; break;
            case "length": BinaryPrimitives.WriteUInt32BigEndian(png.AsSpan(8), uint.MaxValue); break;
            default: png = PngMetadataFixture.Build(("tEXt", Encoding.UTF8.GetBytes("workflow without a separator"))); break;
        }
        using var stream = new MemoryStream(png);
        if (corruption == "truncated") await Assert.ThrowsAsync<EndOfStreamException>(() => PngWorkflowImport.ReadMetadataAsync(stream));
        else await Assert.ThrowsAsync<InvalidDataException>(() => PngWorkflowImport.ReadMetadataAsync(stream));
    }

    [Fact]
    public async Task Compressed_expansion_limit_is_not_swallowed_as_a_skippable_chunk()
    {
        using var stream = new MemoryStream(PngMetadataFixture.Build(("iTXt", PngMetadataFixture.Text("iTXt", "workflow", new string('a', PngWorkflowImport.MaximumTextBytes + 1), 1))));
        var error = await Assert.ThrowsAnyAsync<IOException>(() => PngWorkflowImport.ReadMetadataAsync(stream));
        Assert.Contains("text size limit", error.Message);
    }

    [Fact]
    public async Task Cancellation_does_not_close_the_callers_stream()
    {
        using var cancel = new CancellationTokenSource(); cancel.Cancel(); using var stream = new MemoryStream(PngMetadataFixture.Build());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => PngWorkflowImport.ReadMetadataAsync(stream, cancel.Token)); Assert.True(stream.CanRead);
    }
}
