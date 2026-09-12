using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>
/// Independent builtin CPython source digests versus real managed upper/capitalize/title.
/// No product table/extractor is used as an oracle. The five constructions do not exhaust all strings.
/// </summary>
public sealed class PythonUnicodeCaseSourceReferenceTests
{
    private const string ProtocolSha = "5ccc7810c4c99d427ee1dea167b26bcc638187bb4e0ae0baa2d6cda8291a58de";
    private const string ReferenceSha = "649e592edbb38b02220f9f7b92fda634cb6e8a3c1eea1bc5bde95e2165f98e7b";
    private static readonly string[] ModeNames = ["upper_single", "capitalize_single", "capitalize_a_cp_sigma", "title_a_cp_sigma", "title_a_sigma_cp_a"];
    private static readonly Lazy<Corpus> Source = new(ReadCorpus, LazyThreadSafetyMode.ExecutionAndPublication);

    public static IEnumerable<object[]> Plans()
    {
        for (int plane = 0; plane < 17; plane++)
            foreach (string mode in ModeNames) yield return [plane, mode];
    }

    [Theory]
    [MemberData(nameof(Plans))]
    public void RealCaseOperationMatchesThePublishedBuiltinDigestForOnePlaneAndRecipe(int plane, string mode)
    {
        var corpus = Source.Value;
        var plan = corpus.Protocol.GetProperty("helperCasing");
        var source = corpus.Reference.GetProperty("helperCasing");
        var planePlan = plan.GetProperty("planes")[plane];
        var modePlan = Assert.Single(plan.GetProperty("modes").EnumerateArray(), m => m.GetProperty("id").GetString() == mode);
        var expected = Assert.Single(source.GetProperty("records").EnumerateArray(), record =>
            record.GetProperty("plane").GetInt32() == plane && record.GetProperty("mode").GetString() == mode);
        string operation = modePlan.GetProperty("operation").GetString()!;
        string prefix = modePlan.GetProperty("prefix").GetString()!;
        string suffix = modePlan.GetProperty("suffix").GetString()!;
        int start = planePlan.GetProperty("startInclusive").GetInt32();
        int end = planePlan.GetProperty("endInclusive").GetInt32();
        Assert.Equal(plane << 16, start); Assert.Equal(((plane + 1) << 16) - 1, end);
        using var inputs = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var outputs = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var utf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);
        // The five fixed constructions and their full mappings fit within 24 UTF-8 bytes.
        // Reuse one frame buffer; no complete plane or output corpus is materialized.
        byte[] frame = new byte[32];
        int count = 0;
        long inputBytes = 0, outputBytes = 0;
        for (int cp = start; cp <= end; cp++)
        {
            if (cp is >= 0xd800 and <= 0xdfff) continue;
            string input = string.Concat(prefix, new Rune(cp).ToString(), suffix);
            string output = operation switch
            {
                "upper" => PythonUnicodeCase.Upper(input),
                "capitalize" => PythonUnicodeCase.Capitalize(input),
                "title" => PythonUnicodeCase.Title(input),
                _ => throw new InvalidDataException("Unexpected published casing operation.")
            };
            inputBytes = checked(inputBytes + Append(inputs, cp, input));
            outputBytes = checked(outputBytes + Append(outputs, cp, output));
            count++;
        }
        Assert.Equal(plane == 0 ? 63488 : 65536, count);
        Assert.Equal(planePlan.GetProperty("scalarCount").GetInt32(), count);
        Assert.Equal(expected.GetProperty("count").GetInt32(), count);
        Assert.Equal(expected.GetProperty("inputFramedBytes").GetInt64(), inputBytes);
        Assert.Equal(expected.GetProperty("outputFramedBytes").GetInt64(), outputBytes);
        Assert.Equal(expected.GetProperty("inputSha256").GetString(), Convert.ToHexStringLower(inputs.GetHashAndReset()));
        string actualHash = Convert.ToHexStringLower(outputs.GetHashAndReset());
        Assert.Equal(expected.GetProperty("sha256").GetString(), actualHash);
        var repeats = expected.GetProperty("repeatSha256").EnumerateArray().ToArray();
        Assert.Equal(3, repeats.Length); Assert.True(expected.GetProperty("repeatsIdentical").GetBoolean());
        Assert.All(repeats, repeat => Assert.Equal(actualHash, repeat.GetString()));

        int Append(IncrementalHash hash, int cp, string text)
        {
            int byteCount = utf8.GetByteCount(text);
            Assert.InRange(byteCount, 0, frame.Length - 8);
            BinaryPrimitives.WriteUInt32LittleEndian(frame.AsSpan(0, 4), checked((uint)cp));
            BinaryPrimitives.WriteUInt32LittleEndian(frame.AsSpan(4, 4), checked((uint)byteCount));
            int written = utf8.GetBytes(text.AsSpan(), frame.AsSpan(8));
            Assert.Equal(byteCount, written);
            hash.AppendData(frame.AsSpan(0, 8 + written));
            return 8 + written;
        }
    }

    private sealed record Corpus(JsonElement Protocol, JsonElement Reference);

    private static Corpus ReadCorpus()
    {
        var protocol = Read("case-converter.protocol.json", 44546, ProtocolSha);
        var reference = Read("case-converter.reference.json", 268611, ReferenceSha);
        Assert.Equal(1, protocol.GetProperty("schema").GetInt32());
        Assert.Equal(1, reference.GetProperty("schema").GetInt32());
        Assert.Equal("case-converter-python312-v1", protocol.GetProperty("id").GetString());
        Assert.Equal(protocol.GetProperty("id").GetString(), reference.GetProperty("protocolId").GetString());
        Assert.Equal(ProtocolSha, reference.GetProperty("protocolSha256").GetString());
        Assert.Equal("74e1f790f0c22ddd91d1230322ed3abc44b0143a", reference.GetProperty("collectorCommit").GetString());
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", reference.GetProperty("backendCommit").GetString());
        Assert.Equal(reference.GetProperty("backendCommit").GetString(), protocol.GetProperty("backendCommit").GetString());
        Assert.True(reference.GetProperty("executed").GetBoolean());
        Assert.Equal("CPython", reference.GetProperty("implementation").GetString());
        Assert.Equal("3.12.10", reference.GetProperty("python").GetString());
        Assert.Equal("3.12.10", protocol.GetProperty("python").GetString());
        Assert.Equal("15.0.0", reference.GetProperty("unicodeVersion").GetString());
        Assert.Equal("15.0.0", protocol.GetProperty("unicodeVersion").GetString());
        Assert.True(JsonElement.DeepEquals(protocol.GetProperty("sources"), reference.GetProperty("sourceFiles")));
        Assert.True(JsonElement.DeepEquals(protocol.GetProperty("helperFiles"), reference.GetProperty("helperFiles")));
        Assert.Equal(6, reference.GetProperty("helperFiles").GetArrayLength());
        Assert.True(JsonElement.DeepEquals(protocol.GetProperty("profile"), reference.GetProperty("profile")));
        Assert.Equal(10, reference.GetProperty("sourceFiles").GetArrayLength());
        Assert.Equal(74, reference.GetProperty("sourceFiles").EnumerateArray().Sum(file => file.GetProperty("declarations").GetArrayLength()));
        Assert.Equal(85, reference.GetProperty("casingDigestPlans").GetInt32());
        var plan = protocol.GetProperty("helperCasing");
        var casing = reference.GetProperty("helperCasing");
        Assert.Equal("cpython312-upper-capitalize-title-valid-scalars-v1", plan.GetProperty("id").GetString());
        Assert.Equal(plan.GetProperty("id").GetString(), casing.GetProperty("id").GetString());
        Assert.Equal(plan.GetProperty("framing").GetString(), casing.GetProperty("framing").GetString());
        Assert.Equal(3, plan.GetProperty("repeats").GetInt32()); Assert.Equal(3, casing.GetProperty("repeats").GetInt32());
        Assert.Equal(1112064, casing.GetProperty("totalValidScalars").GetInt32());
        Assert.Equal(1112064, plan.GetProperty("totalValidScalars").GetInt32());
        Assert.Equal(5560320, casing.GetProperty("inputApplicationsPerPass").GetInt32());
        Assert.Equal(5560320, plan.GetProperty("inputApplicationsPerPass").GetInt32());
        Assert.Equal(85, plan.GetProperty("digestCount").GetInt32());
        Assert.False(casing.GetProperty("nativeProductOrTableRead").GetBoolean());
        Assert.Equal(85, casing.GetProperty("records").GetArrayLength());
        Assert.Equal(17, plan.GetProperty("planes").GetArrayLength());
        Assert.Equal(5, plan.GetProperty("modes").GetArrayLength());
        string[] prefixes = ["", "", "A", "A", "AΣ"], suffixes = ["", "", "Σ", "Σ", "A"];
        string[] operations = ["upper", "capitalize", "capitalize", "title", "title"];
        for (int i = 0; i < ModeNames.Length; i++)
        {
            var mode = plan.GetProperty("modes")[i];
            Assert.Equal(ModeNames[i], mode.GetProperty("id").GetString());
            Assert.Equal(operations[i], mode.GetProperty("operation").GetString());
            Assert.Equal(prefixes[i], mode.GetProperty("prefix").GetString());
            Assert.Equal(suffixes[i], mode.GetProperty("suffix").GetString());
        }
        var records = casing.GetProperty("records").EnumerateArray().ToArray();
        Assert.Equal(5560320, records.Sum(record => record.GetProperty("count").GetInt32()));
        Assert.Equal(85, records.Select(record => (record.GetProperty("plane").GetInt32(), record.GetProperty("mode").GetString())).Distinct().Count());
        for (int i = 0; i < records.Length; i++)
        {
            Assert.Equal(i / 5, records[i].GetProperty("plane").GetInt32());
            Assert.Equal(ModeNames[i % 5], records[i].GetProperty("mode").GetString());
            Assert.Matches("^[0-9a-f]{64}$", records[i].GetProperty("sha256").GetString()!);
            Assert.Matches("^[0-9a-f]{64}$", records[i].GetProperty("inputSha256").GetString()!);
            var hashes = records[i].GetProperty("repeatSha256").EnumerateArray().ToArray();
            Assert.Equal(3, hashes.Length); Assert.True(records[i].GetProperty("repeatsIdentical").GetBoolean());
            Assert.All(hashes, hash => Assert.Equal(records[i].GetProperty("sha256").GetString(), hash.GetString()));
        }
        foreach (var (name, size, sha) in new[]
        {
            ("protocol.json", 44546, ProtocolSha),
            ("reference.py", 17248, "ceb3378998ea7491e75d5b04f6637f055befe9ca86eb9a1f2e1b998f3b080e9d"),
            ("README.md", 7772, "a4339a38ce77071bfb61733dd64270886f275c27d17f4e58f474f5c8024ca4df")
        })
        {
            var pin = reference.GetProperty("laboratoryFiles").GetProperty("labs/case-converter-source/" + name);
            Assert.Equal(size, pin.GetProperty("bytes").GetInt32());
            Assert.Equal(sha, pin.GetProperty("sha256").GetString());
        }
        return new(protocol, reference);
    }

    private static JsonElement Read(string name, int length, string sha)
    {
        using var stream = typeof(PythonUnicodeCaseSourceReferenceTests).Assembly.GetManifestResourceStream("ComfySharp.Core.Tests.Fixtures." + name);
        Assert.NotNull(stream); using var copy = new MemoryStream(); stream.CopyTo(copy); byte[] bytes = copy.ToArray();
        Assert.Equal(length, bytes.Length); Assert.Equal(sha, Convert.ToHexStringLower(SHA256.HashData(bytes)));
        using var document = JsonDocument.Parse(bytes);
        return document.RootElement.Clone(); // Immutable managed snapshot; stream/document are disposed.
    }
}
