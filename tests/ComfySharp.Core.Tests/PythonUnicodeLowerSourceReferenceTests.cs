using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

/// <summary>
/// Independent builtin CPython source digests versus the real managed Lower implementation.
/// No product table/extractor is used as an oracle. The four contexts do not exhaust all strings.
/// </summary>
public sealed class PythonUnicodeLowerSourceReferenceTests
{
    private const string ProtocolSha = "0d00fa0c7721b07b2fc12381f60ed364b286963553577fcdb6672c028e18aa92";
    private const string ReferenceSha = "12935c33d4632a79e499a7c24b052d88036aee8323e45ea92d923443b2b9e04f";
    private static readonly string[] ModeNames = ["single", "suffix_sigma", "prefix_a_suffix_sigma", "a_sigma_cp_a"];
    private static readonly Lazy<Corpus> Source = new(ReadCorpus, LazyThreadSafetyMode.ExecutionAndPublication);

    public static IEnumerable<object[]> Plans()
    {
        for (int plane = 0; plane < 17; plane++)
            foreach (string mode in ModeNames) yield return [plane, mode];
    }

    [Theory]
    [MemberData(nameof(Plans))]
    public void RealLowerMatchesThePublishedBuiltinDigestForOnePlaneAndContext(int plane, string mode)
    {
        var corpus = Source.Value;
        var plan = corpus.Protocol.GetProperty("helperLower");
        var source = corpus.Reference.GetProperty("helperLower");
        var planePlan = plan.GetProperty("planes")[plane];
        var modePlan = Assert.Single(plan.GetProperty("modes").EnumerateArray(), m => m.GetProperty("id").GetString() == mode);
        var expected = Assert.Single(source.GetProperty("records").EnumerateArray(), record =>
            record.GetProperty("plane").GetInt32() == plane && record.GetProperty("mode").GetString() == mode);
        string prefix = modePlan.GetProperty("prefix").GetString()!;
        string suffix = modePlan.GetProperty("suffix").GetString()!;
        int start = planePlan.GetProperty("startInclusive").GetInt32();
        int end = planePlan.GetProperty("endInclusive").GetInt32();
        Assert.Equal(plane << 16, start); Assert.Equal(((plane + 1) << 16) - 1, end);
        using var inputs = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var outputs = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var utf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);
        // The four exact constructions and their lowercase mappings fit well below 24 UTF-8 bytes.
        // Reuse one frame buffer; no complete plane or output corpus is materialized.
        byte[] frame = new byte[32];
        int count = 0;
        long inputBytes = 0, outputBytes = 0;
        for (int cp = start; cp <= end; cp++)
        {
            if (cp is >= 0xd800 and <= 0xdfff) continue;
            string input = string.Concat(prefix, new Rune(cp).ToString(), suffix);
            string output = PythonUnicodeLower.Lower(input);
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
        var protocol = Read("text-comparison.protocol.json", 41632, ProtocolSha);
        var reference = Read("text-comparison.reference.json", 280652, ReferenceSha);
        Assert.Equal(1, protocol.GetProperty("schema").GetInt32());
        Assert.Equal(1, reference.GetProperty("schema").GetInt32());
        Assert.Equal("text-comparison-python312-v1", protocol.GetProperty("id").GetString());
        Assert.Equal(protocol.GetProperty("id").GetString(), reference.GetProperty("protocolId").GetString());
        Assert.Equal(ProtocolSha, reference.GetProperty("protocolSha256").GetString());
        Assert.Equal("5466f97312d8dbdbe5c656a6c31061895afc9af3", reference.GetProperty("collectorCommit").GetString());
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
        Assert.Equal(10, reference.GetProperty("sourceFiles").GetArrayLength());
        Assert.Equal(75, reference.GetProperty("sourceFiles").EnumerateArray().Sum(file => file.GetProperty("declarations").GetArrayLength()));
        Assert.Equal(68, reference.GetProperty("lowerDigestPlans").GetInt32());
        var plan = protocol.GetProperty("helperLower");
        var lower = reference.GetProperty("helperLower");
        Assert.Equal("cpython312-lower-valid-scalars-v1", plan.GetProperty("id").GetString());
        Assert.Equal(plan.GetProperty("id").GetString(), lower.GetProperty("id").GetString());
        Assert.Equal(plan.GetProperty("framing").GetString(), lower.GetProperty("framing").GetString());
        Assert.Equal(3, plan.GetProperty("repeats").GetInt32()); Assert.Equal(3, lower.GetProperty("repeats").GetInt32());
        Assert.Equal(1112064, lower.GetProperty("totalValidScalars").GetInt32());
        Assert.Equal(1112064, plan.GetProperty("totalValidScalars").GetInt32());
        Assert.Equal(4448256, lower.GetProperty("inputApplicationsPerPass").GetInt32());
        Assert.False(lower.GetProperty("nativeProductOrTableRead").GetBoolean());
        Assert.Equal(68, lower.GetProperty("records").GetArrayLength());
        Assert.Equal(17, plan.GetProperty("planes").GetArrayLength());
        Assert.Equal(4, plan.GetProperty("modes").GetArrayLength());
        string[] prefixes = ["", "", "A", "AΣ"], suffixes = ["", "Σ", "Σ", "A"];
        for (int i = 0; i < ModeNames.Length; i++)
        {
            var mode = plan.GetProperty("modes")[i];
            Assert.Equal(ModeNames[i], mode.GetProperty("id").GetString());
            Assert.Equal(prefixes[i], mode.GetProperty("prefix").GetString());
            Assert.Equal(suffixes[i], mode.GetProperty("suffix").GetString());
        }
        var records = lower.GetProperty("records").EnumerateArray().ToArray();
        Assert.Equal(4448256, records.Sum(record => record.GetProperty("count").GetInt32()));
        Assert.Equal(68, records.Select(record => (record.GetProperty("plane").GetInt32(), record.GetProperty("mode").GetString())).Distinct().Count());
        for (int i = 0; i < records.Length; i++)
        {
            Assert.Equal(i / 4, records[i].GetProperty("plane").GetInt32());
            Assert.Equal(ModeNames[i % 4], records[i].GetProperty("mode").GetString());
            Assert.Matches("^[0-9a-f]{64}$", records[i].GetProperty("sha256").GetString()!);
            Assert.Matches("^[0-9a-f]{64}$", records[i].GetProperty("inputSha256").GetString()!);
            var hashes = records[i].GetProperty("repeatSha256").EnumerateArray().ToArray();
            Assert.Equal(3, hashes.Length); Assert.True(records[i].GetProperty("repeatsIdentical").GetBoolean());
            Assert.All(hashes, hash => Assert.Equal(records[i].GetProperty("sha256").GetString(), hash.GetString()));
        }
        return new(protocol, reference);
    }

    private static JsonElement Read(string name, int length, string sha)
    {
        using var stream = typeof(PythonUnicodeLowerSourceReferenceTests).Assembly.GetManifestResourceStream("ComfySharp.Core.Tests.Fixtures." + name);
        Assert.NotNull(stream); using var copy = new MemoryStream(); stream.CopyTo(copy); byte[] bytes = copy.ToArray();
        Assert.Equal(length, bytes.Length); Assert.Equal(sha, Convert.ToHexStringLower(SHA256.HashData(bytes)));
        using var document = JsonDocument.Parse(bytes);
        return document.RootElement.Clone(); // Immutable managed snapshot; stream/document are disposed.
    }
}
