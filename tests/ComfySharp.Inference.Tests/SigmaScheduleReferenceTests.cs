using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using TorchSharp;
using Xunit;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class SigmaScheduleReferenceTests(ITestOutputHelper output)
{
    private const string CorpusHash = "c78e03c13d3feed5aaf876a9efa5cc2f9f53822765be5cb6c97b0469f63412e1";
    private static byte[] ReadCorpus()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var name = assembly.GetManifestResourceNames().Single(n => n.EndsWith("sigma-schedules.cpu-f32.json", StringComparison.Ordinal));
        using var stream = assembly.GetManifestResourceStream(name)!;
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
    public void ReferenceIsPinnedAndRecordsTheIndependentLaboratory()
    {
        var bytes = ReadCorpus();
        Assert.Equal(CorpusHash, Convert.ToHexStringLower(SHA256.HashData(bytes)));
        using var document = JsonDocument.Parse(bytes);
        var root = document.RootElement;
        Assert.Equal("1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", root.GetProperty("backendCommit").GetString());
        Assert.Equal("2.13.0+cu130", root.GetProperty("laboratory").GetProperty("torch").GetString());
        Assert.False(root.GetProperty("laboratory").GetProperty("modelWeightsUsed").GetBoolean());
        Assert.Equal(46, root.GetProperty("cases").GetArrayLength());
        Assert.Equal(4, root.GetProperty("invalidCases").GetArrayLength());
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void NativeFloat32ScheduleMatchesFrozenPythonFunction(string id)
    {
        using var document = JsonDocument.Parse(ReadCorpus());
        var root = document.RootElement;
        var reference = root.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("id").GetString() == id);
        var expectedBits = reference.GetProperty("float32Bits").EnumerateArray().Select(b => b.GetUInt32()).ToArray();
        var expectedBytes = expectedBits.SelectMany(BitConverter.GetBytes).ToArray();
        Assert.True(BitConverter.IsLittleEndian, "The qualified x64/ARM64 targets are little-endian.");
        Assert.Equal(reference.GetProperty("bytesSha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(expectedBytes)));
        using var scope = NewDisposeScope();
        using var actual = Generate(reference);
        Assert.Equal(ScalarType.Float32, actual.dtype);
        Assert.Equal("cpu", actual.device.ToString());
        Assert.Equal(new long[] { expectedBits.Length }, actual.shape);
        var values = actual.data<float>().ToArray();
        var profile = root.GetProperty("comparison");
        double atol = profile.GetProperty("absoluteTolerance").GetDouble();
        double rtol = profile.GetProperty("relativeTolerance").GetDouble();
        double maxAbsolute = 0;
        long maxUlp = 0;
        for (int i = 0; i < values.Length; i++)
        {
            float expected = BitConverter.UInt32BitsToSingle(expectedBits[i]);
            float value = values[i];
            Assert.True(float.IsFinite(value), $"{id}[{i}] is not finite.");
            Assert.Equal(expected == 0, value == 0);
            if (expected == 0) Assert.Equal(expectedBits[i], BitConverter.SingleToUInt32Bits(value));
            double error = Math.Abs((double)value - expected);
            double tolerance = atol + rtol * Math.Abs(expected);
            Assert.True(error <= tolerance, $"{id}[{i}]: actual={value:R}, expected={expected:R}, error={error:R}, tolerance={tolerance:R}");
            maxAbsolute = Math.Max(maxAbsolute, error);
            // Every value in these nonnegative sigma fixtures has the same sign ordering in its float bits.
            maxUlp = Math.Max(maxUlp, Math.Abs((long)BitConverter.SingleToUInt32Bits(value) - expectedBits[i]));
        }
        output.WriteLine($"{id}: maxAbsolute={maxAbsolute:R}, maxULP={maxUlp}; reference torch2.13 CPU, product libtorch2.10 CPU.");
    }

    [Theory]
    [InlineData("karras/zero-rho", "rho")]
    [InlineData("exponential/zero-minimum", "sigmaMin")]
    [InlineData("polyexponential/zero-minimum", "sigmaMin")]
    [InlineData("vp/overflow", null)]
    public void UndefinedReferenceDomainsProduceAnExplicitPortError(string id, string? parameter)
    {
        using var document = JsonDocument.Parse(ReadCorpus());
        var reference = document.RootElement.GetProperty("invalidCases").EnumerateArray().Single(c => c.GetProperty("id").GetString() == id);
        using var scope = NewDisposeScope();
        if (parameter is null)
            Assert.Throws<ArithmeticException>(() => { using var result = Generate(reference); });
        else
            Assert.Equal(parameter, Assert.Throws<ArgumentOutOfRangeException>(() => { using var result = Generate(reference); }).ParamName);
    }

    private static Tensor Generate(JsonElement reference)
    {
        var p = reference.GetProperty("parameters");
        int n = p.GetProperty("n").GetInt32();
        double Value(string name) => p.GetProperty(name).GetDouble();
        return reference.GetProperty("generator").GetString() switch
        {
            "Karras" => SigmaSchedules.Karras(n, Value("sigma_min"), Value("sigma_max"), Value("rho")),
            "Exponential" => SigmaSchedules.Exponential(n, Value("sigma_min"), Value("sigma_max")),
            "Polyexponential" => SigmaSchedules.Polyexponential(n, Value("sigma_min"), Value("sigma_max"), Value("rho")),
            "Laplace" => SigmaSchedules.Laplace(n, Value("sigma_min"), Value("sigma_max"), Value("mu"), Value("beta")),
            "VP" => SigmaSchedules.VP(n, Value("beta_d"), Value("beta_min"), Value("eps_s")),
            _ => throw new InvalidDataException("Unknown generator in reference fixture.")
        };
    }
}
