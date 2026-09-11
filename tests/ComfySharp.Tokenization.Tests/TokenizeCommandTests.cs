using System.Text.Json;
using ComfySharp.Tokenize;
using Xunit;

namespace ComfySharp.Tokenization.Tests;

public sealed class TokenizeCommandTests
{
    [Fact]
    public async Task StdinProducesIndependentSdxlChannelsAndExactIds()
    {
        using var output = new StringWriter();
        using var error = new StringWriter();
        var exit = await TokenizeCommand.RunAsync(["--stdin", "--profile", "sdxl"], new StringReader("a"), output, error);
        Assert.Equal(0, exit);
        Assert.Equal("", error.ToString());
        using var report = JsonDocument.Parse(output.ToString());
        var root = report.RootElement;
        Assert.Equal("ok", root.GetProperty("status").GetString());
        Assert.Equal("not_assessed", root.GetProperty("modelCompatibility").GetString());
        Assert.Equal(new[] { 49406, 320, 49407 }, root.GetProperty("ids").EnumerateArray().Select(id => id.GetInt32()));
        var channels = root.GetProperty("outputs");
        Assert.Equal(new[] { "g", "l" }, channels.EnumerateObject().Select(p => p.Name));
        Assert.Equal(77, channels.GetProperty("g")[0].GetArrayLength());
        Assert.Equal(0, channels.GetProperty("g")[0][76].GetProperty("id").GetInt32());
        Assert.Equal(49407, channels.GetProperty("l")[0][76].GetProperty("id").GetInt32());
        Assert.Equal(1, channels.GetProperty("l")[0][1].GetProperty("wordId").GetInt32());
    }

    [Fact]
    public async Task NonfiniteWeightHasValidExplicitJsonRepresentation()
    {
        using var output = new StringWriter();
        using var error = new StringWriter();
        Assert.Equal(0, await TokenizeCommand.RunAsync(["--text", "(a:-inf)"], TextReader.Null, output, error));
        using var report = JsonDocument.Parse(output.ToString());
        var token = report.RootElement.GetProperty("outputs").GetProperty("l")[0][1];
        Assert.Equal("-Infinity", token.GetProperty("weight").GetString());
        Assert.Equal("fff0000000000000", token.GetProperty("weightBits").GetString());
    }

    public static IEnumerable<object[]> InvalidArguments =>
    [
        [Array.Empty<string>()], [new[] { "--text", "a", "--stdin" }],
        [new[] { "--text", "a", "--profile", "bad" }], [new[] { "--text", "a", "--text", "b" }],
        [new[] { "--profile" }], [new[] { "--wat" }]
    ];

    [Theory, MemberData(nameof(InvalidArguments))]
    public async Task InvalidInputCannotEmitSuccess(string[] args)
    {
        using var output = new StringWriter();
        using var error = new StringWriter();
        Assert.Equal(2, await TokenizeCommand.RunAsync(args, TextReader.Null, output, error));
        Assert.Equal("", output.ToString());
        Assert.NotEmpty(error.ToString());
    }

    [Fact]
    public async Task PreCancelledRequestProducesNoJson()
    {
        using var output = new StringWriter();
        using var error = new StringWriter();
        Assert.Equal(130, await TokenizeCommand.RunAsync(["--text", "a"], TextReader.Null, output, error, new CancellationToken(true)));
        Assert.Equal("", output.ToString());
    }

    [Theory]
    [InlineData("fffe00d8")]
    [InlineData("fffe6100")]
    [InlineData("feff0061")]
    [InlineData("c328")]
    public async Task FileInputRejectsNonUtf8BytesWithoutReplacement(string hex)
    {
        var path = Path.Combine(Path.GetTempPath(), "comfysharp-utf8-" + Guid.NewGuid().ToString("N") + ".txt");
        try
        {
            await File.WriteAllBytesAsync(path, Convert.FromHexString(hex));
            using var output = new StringWriter();
            using var error = new StringWriter();
            Assert.Equal(2, await TokenizeCommand.RunAsync(["--file", path], TextReader.Null, output, error));
            Assert.Equal("", output.ToString());
            Assert.NotEmpty(error.ToString());
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task Utf8BomIsAcceptedAsEncodingMarker()
    {
        var path = Path.Combine(Path.GetTempPath(), "comfysharp-utf8-" + Guid.NewGuid().ToString("N") + ".txt");
        try
        {
            await File.WriteAllBytesAsync(path, new byte[] { 0xef, 0xbb, 0xbf, 0x61 });
            using var output = new StringWriter();
            using var error = new StringWriter();
            Assert.Equal(0, await TokenizeCommand.RunAsync(["--file", path], TextReader.Null, output, error));
            using var json = JsonDocument.Parse(output.ToString());
            Assert.Equal(new[] { 49406, 320, 49407 }, json.RootElement.GetProperty("ids").EnumerateArray().Select(value => value.GetInt32()));
        }
        finally { File.Delete(path); }
    }
}
