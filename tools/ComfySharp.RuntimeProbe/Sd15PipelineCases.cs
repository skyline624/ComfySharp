using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ComfySharp.Tokenization;

namespace ComfySharp.RuntimeProbe;

// Managed protocol inputs only. No tensor, native bootstrap or output oracle is constructed here.
internal sealed class Sd15PipelineCase
{
    private readonly Sd15PipelineCases.TextRecord positive;
    private readonly Sd15PipelineCases.TextRecord negative;
    private readonly float[] noise;
    private readonly float[] sigmas;

    internal Sd15PipelineCase(string id, Sd15PipelineCases.TextRecord positive,
        Sd15PipelineCases.TextRecord negative, double scale, bool maximumDenoise,
        float[] noise, float[] sigmas, string noiseSha256, string sigmaSha256)
    {
        Id = id; this.positive = positive; this.negative = negative;
        GuidanceScale = scale; MaximumDenoise = maximumDenoise;
        this.noise = (float[])noise.Clone(); this.sigmas = (float[])sigmas.Clone();
        NoiseSha256 = noiseSha256; SigmaSha256 = sigmaSha256;
    }

    internal string Id { get; }
    internal double GuidanceScale { get; }
    internal bool MaximumDenoise { get; }
    internal int Steps => sigmas.Length - 1;
    internal int Repetitions => 3;
    internal string NoiseSha256 { get; }
    internal string SigmaSha256 { get; }
    internal string ProtocolSha256 => Sd15PipelineCases.ProtocolSha256;
    internal Sd15PipelineConditioning Conditioning => TokenizeAndVerify();
    internal float[] NoiseValues() => (float[])noise.Clone();
    internal float[] SigmaValues() => (float[])sigmas.Clone();

    internal Sd15PipelineConditioning TokenizeAndVerify(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var result = Sd15PipelineExecution.Tokenize(positive.Text, negative.Text,
            GuidanceScale, MaximumDenoise, cancellationToken);
        positive.Verify(result.Positive);
        negative.Verify(result.Negative);
        cancellationToken.ThrowIfCancellationRequested();
        return result;
    }
}

internal static class Sd15PipelineCases
{
    internal const string ResourceName = "ComfySharp.RuntimeProbe.Fixtures.sd15-pipeline.protocol.json";
    internal const string ProtocolSha256 = "f0e4f537414c6f8692c852832d040fe3fd5dc23fdc270bb351a16bbba6cd75e7";
    internal const string BackendCommit = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a";
    internal const string Profile = "sd15-pipeline-native210-cpu-f32-v1";
    internal const string ParameterRecipe = "sha256-name-lcg-high16-power2-v1";
    internal const int ParameterCount = 971;
    internal const long ParameterBytes = 58_137_836;
    private static readonly IReadOnlyList<Sd15PipelineCase> cases = Load();
    internal static IReadOnlyList<Sd15PipelineCase> All => cases;

    internal static Sd15PipelineCase Get(string id)
    {
        ArgumentNullException.ThrowIfNull(id);
        return cases.FirstOrDefault(c => string.Equals(c.Id, id, StringComparison.Ordinal))
            ?? throw new ArgumentException("Unknown fixed SD1.5 pipeline case.", nameof(id));
    }

    internal sealed class TextRecord(string text, ClipToken[][] chunks)
    {
        internal string Text { get; } = text;
        internal int ChunkCount => chunks.Length;

        internal void Verify(ClipTokenization actual)
        {
            Require(actual.Profile == ClipProfile.Sd1L && actual.Chunks.Count == chunks.Length,
                "Tokenization profile/chunk count differs from frozen source.");
            for (int row = 0; row < chunks.Length; row++)
            {
                Require(actual.Chunks[row].Count == chunks[row].Length, "Token chunk length differs.");
                for (int column = 0; column < chunks[row].Length; column++)
                {
                    var observed = actual.Chunks[row][column];
                    var expected = chunks[row][column];
                    Require(observed.Id == expected.Id && observed.WordId == expected.WordId &&
                        BitConverter.DoubleToInt64Bits(observed.Weight) == BitConverter.DoubleToInt64Bits(expected.Weight),
                        $"Token ID/weight bits/word ID differs at chunk {row}, token {column}.");
                }
            }
        }
    }

    private static IReadOnlyList<Sd15PipelineCase> Load()
    {
        using var stream = typeof(Sd15PipelineCases).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidDataException("Missing fixed SD1.5 pipeline protocol resource.");
        Require(stream.Length is > 0 and <= 262144, "Unexpected protocol resource size.");
        byte[] bytes = new byte[checked((int)stream.Length)];
        stream.ReadExactly(bytes);
        Require(Hash(bytes) == ProtocolSha256, "SD1.5 pipeline protocol bytes differ from the reviewed pin.");
        // Hash precedes parsing; no newline normalization or acceptance of a reserialized protocol.
        using var document = JsonDocument.Parse(bytes);
        var root = document.RootElement;
        Require(root.GetProperty("schema").GetInt32() == 1 && String(root, "profile") == Profile &&
            String(root, "backendCommit") == BackendCommit, "Protocol identity differs.");
        Require(root.GetProperty("syntheticWeights").GetBoolean() &&
            !root.GetProperty("pretrainedWeightsUsed").GetBoolean() &&
            !root.GetProperty("familyQualification").GetBoolean(), "Protocol scope differs.");
        ValidateConfiguration(root);
        Require(root.GetProperty("repetitions").GetInt32() == 3, "Repeat count differs.");
        Require(root.GetProperty("observerSequence").EnumerateArray().Select(x => x.GetString())
            .SequenceEqual(new[] { "off", "on", "off" }), "Observer sequence differs.");
        var inputs = root.GetProperty("inputRecords").EnumerateArray()
            .ToDictionary(e => String(e, "id"), StringComparer.Ordinal);
        Require(inputs.Count == 20, "Input record count differs.");
        var texts = ReadTexts(root, inputs);
        var result = new List<Sd15PipelineCase>(4);
        string[] ids = ["empty-one-step", "weighted-three-step", "two-chunks-separate", "maximum-start"];
        string[] positives = ["packing/empty", "packing/nested", "packing/length-76", "packing/implicit"];
        string[] negatives = ["packing/empty", "packing/implicit", "packing/empty", "packing/empty"];
        double[] scales = [1, 3.5, 3.5, 7];
        int[] steps = [1, 3, 2, 2];
        var definitions = root.GetProperty("cases");
        Require(definitions.GetArrayLength() == 4, "Case count differs.");
        for (int index = 0; index < 4; index++)
        {
            var definition = definitions[index];
            string id = String(definition, "id");
            Require(id == ids[index] && String(definition, "positiveText") == positives[index] &&
                String(definition, "negativeText") == negatives[index], "Case/text identity differs.");
            Require(String(definition, "predictionKind") == "epsilon" && String(definition, "policy") == "separate" &&
                !definition.GetProperty("negativeContextIsNull").GetBoolean() &&
                !definition.GetProperty("disableScaleOneOptimization").GetBoolean(), "EPS/CFG options differ.");
            double scale = definition.GetProperty("scale").GetDouble();
            Require(scale == scales[index] && Bits64(scale) == Parse64(String(definition, "scaleFloat64BitsHex")),
                "CFG scale bits differ.");
            bool maximum = definition.GetProperty("maximumDenoise").GetBoolean();
            Require(maximum == (index == 3) && definition.GetProperty("steps").GetInt32() == steps[index] &&
                definition.GetProperty("unetCallsPerRun").GetInt32() == steps[index] * (index == 0 ? 1 : 2),
                "Case step/maximum-denoise contract differs.");
            var positive = texts[positives[index]]; var negative = texts[negatives[index]];
            Shape(definition.GetProperty("positiveContextShape"), [1, positive.ChunkCount * 77, 16]);
            Shape(definition.GetProperty("negativeContextShape"), [1, negative.ChunkCount * 77, 16]);
            foreach (string key in new[] { "initialDiffusionLatentShape", "finalDiffusionLatentShape", "rawVaeLatentShape" })
                Shape(definition.GetProperty(key), [1, 4, 4, 5]);
            Shape(definition.GetProperty("imageShape"), [1, 32, 40, 3]);
            var bindings = definition.GetProperty("inputs");
            foreach (var (key, expected) in new[] { ("positiveIds", positives[index] + "/ids"),
                ("positiveWeights", positives[index] + "/weights"), ("negativeIds", negatives[index] + "/ids"),
                ("negativeWeights", negatives[index] + "/weights"), ("noise", id + "/noise"),
                ("emptyLatent", id + "/emptyLatent"), ("sigmas", id + "/sigmas") })
                Require(String(bindings, key) == expected, "Case input binding differs.");
            var noiseRecord = inputs[id + "/noise"];
            var sigmaRecord = inputs[id + "/sigmas"];
            var noise = ReadNoise(noiseRecord, id);
            var sigmas = ReadSigmas(sigmaRecord, steps[index], maximum);
            var empty = inputs[id + "/emptyLatent"];
            Require(String(empty, "recipe") == "all-zero F32", "Empty latent recipe differs.");
            VerifyPayload(empty, "float32", [1, 4, 4, 5], new byte[320]);
            result.Add(new(id, positive, negative, scale, maximum, noise, sigmas,
                String(noiseRecord, "sha256"), String(sigmaRecord, "sha256")));
        }
        return result.AsReadOnly();
    }

    private static Dictionary<string, TextRecord> ReadTexts(JsonElement root, Dictionary<string, JsonElement> inputs)
    {
        var result = new Dictionary<string, TextRecord>(StringComparer.Ordinal);
        foreach (var text in root.GetProperty("texts").EnumerateArray())
        {
            string id = String(text, "id");
            Require(String(text, "profile") == "sd1-l" && String(text, "inputIds") == id + "/ids" &&
                String(text, "inputWeights") == id + "/weights", "Source text bindings differ.");
            foreach (string key in new[] { "config", "data", "options", "call" })
                Require(text.GetProperty(key).EnumerateObject().Count() == 0, "Nondefault tokenization options.");
            var chunks = text.GetProperty("sourceChunks");
            Require(chunks.GetArrayLength() is 1 or 2, "Unsupported fixed token chunk count.");
            var tokens = new ClipToken[chunks.GetArrayLength()][];
            var idBytes = new byte[tokens.Length * 77 * 8];
            var weightBytes = new byte[idBytes.Length];
            for (int row = 0; row < tokens.Length; row++)
            {
                Require(chunks[row].GetArrayLength() == 77, "Source token chunk length differs.");
                tokens[row] = new ClipToken[77];
                for (int col = 0; col < 77; col++)
                {
                    var triple = chunks[row][col];
                    Require(triple.GetArrayLength() == 3, "Malformed source token triple.");
                    int tokenId = triple[0].GetInt32(), wordId = triple[2].GetInt32();
                    ulong bits = Parse64(triple[1].GetString() ?? "");
                    double weight = BitConverter.Int64BitsToDouble(unchecked((long)bits));
                    Require(tokenId is >= 0 and < 49408 && wordId >= 0 && double.IsFinite(weight), "Invalid source token.");
                    tokens[row][col] = new(tokenId, weight, wordId);
                    BinaryPrimitives.WriteInt64LittleEndian(idBytes.AsSpan((row * 77 + col) * 8), tokenId);
                    BinaryPrimitives.WriteUInt64LittleEndian(weightBytes.AsSpan((row * 77 + col) * 8), bits);
                }
            }
            VerifyPayload(inputs[id + "/ids"], "int64", [tokens.Length, 77], idBytes);
            VerifyPayload(inputs[id + "/weights"], "float64", [tokens.Length, 77], weightBytes);
            result.Add(id, new(String(text, "text"), tokens));
        }
        Require(result.Count == 4, "Source text count differs.");
        return result;
    }

    private static float[] ReadNoise(JsonElement record, string id)
    {
        string name = "sd15-pipeline/" + id + "/noise";
        Require(String(record, "recipe") == ParameterRecipe && String(record, "recipeName") == name &&
            !record.GetProperty("parameter").GetBoolean() &&
            String(record, "distribution") == "deterministic_test_input_not_Gaussian", "Noise recipe differs.");
        uint seed = BinaryPrimitives.ReadUInt32LittleEndian(SHA256.HashData(Encoding.UTF8.GetBytes(name)));
        Require(seed == record.GetProperty("seed32").GetUInt32(), "Noise seed differs.");
        var values = new float[80]; var bytes = new byte[320];
        for (int i = 0; i < values.Length; i++)
        {
            // The source recipe is independently indexed, not a recurrent LCG or Gaussian RNG.
            uint word = unchecked((uint)i * 1664525u + seed);
            values[i] = ((int)(word >> 16) - 32768) * (1f / 32768f);
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        }
        VerifyPayload(record, "float32", [1, 4, 4, 5], bytes);
        return values;
    }

    private static float[] ReadSigmas(JsonElement record, int steps, bool maximum)
    {
        var hex = record.GetProperty("float32BitsHex"); var decimals = record.GetProperty("values");
        Require(hex.GetArrayLength() == steps + 1 && decimals.GetArrayLength() == steps + 1, "Sigma count differs.");
        var values = new float[steps + 1]; var bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            string value = hex[i].GetString() ?? "";
            Require(value.Length == 8, "Malformed sigma bits.");
            uint bits = uint.Parse(value, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture);
            values[i] = BitConverter.Int32BitsToSingle(unchecked((int)bits));
            Require(float.IsFinite(values[i]) && (double)values[i] == decimals[i].GetDouble(), "Sigma decimal/bit mismatch.");
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(i * 4), bits);
            Require(i == steps ? bits == 0 : values[i] > 0, "Sigma sign/endpoint differs.");
            if (i > 0) Require(values[i] <= values[i - 1], "Sigma order differs.");
        }
        Require(BitConverter.SingleToInt32Bits(values[0]) == (maximum ? 0x4169d592 : 0x3fc00000), "Initial sigma differs.");
        VerifyPayload(record, "float32", [steps + 1], bytes);
        return values;
    }

    private static void ValidateConfiguration(JsonElement root)
    {
        Require(root.GetProperty("parameterCount").GetInt32() == ParameterCount &&
            root.GetProperty("parameterBytes").GetInt64() == ParameterBytes, "Parameter totals differ.");
        var recipe = root.GetProperty("parameterRecipe");
        Require(String(recipe, "version") == ParameterRecipe &&
            String(recipe, "seed") == "uint32 LE first four SHA256(UTF8 name) bytes" &&
            String(recipe, "sample") == "(((index*1664525 + seed) modulo 2^32) >> 16) - 32768; independent indexed expression, not a recurrent LCG",
            "Parameter recipe differs.");
        var configs = root.GetProperty("configs"); var clip = configs.GetProperty("clip");
        Require(clip.GetProperty("hidden_size").GetInt32() == 16 && clip.GetProperty("intermediate_size").GetInt32() == 32 &&
            clip.GetProperty("num_hidden_layers").GetInt32() == 2 && clip.GetProperty("num_attention_heads").GetInt32() == 4 &&
            clip.GetProperty("vocab_size").GetInt32() == 49408 && clip.GetProperty("max_position_embeddings").GetInt32() == 77 &&
            String(clip, "hidden_act") == "quick_gelu" && configs.GetProperty("clipProjectionPresent").GetBoolean() &&
            !configs.GetProperty("projectionUsedForConditioning").GetBoolean(), "CLIP configuration differs.");
        var unet = configs.GetProperty("unet"); var vae = configs.GetProperty("vae");
        Require(unet.GetProperty("model_channels").GetInt32() == 32 && unet.GetProperty("context_dim").GetInt32() == 16 &&
            unet.GetProperty("num_heads").GetInt32() == 4 && unet.GetProperty("num_head_channels").GetInt32() == -1 &&
            !unet.GetProperty("use_linear_in_transformer").GetBoolean() && vae.GetProperty("ch").GetInt32() == 32,
            "U-Net/VAE configuration differs.");
        foreach (var (name, count, elements, digest) in new[] {
            ("clip", 37, 796496L, "e5a0a5b5072818dea0753a7914b8e6bc3908ab0caf94bb52b9b351e9464f21b8"),
            ("unet", 686, 8485476L, "742b08be069f93c45db0239eb4c942a17f96f9be7ee120f54a14c028c3bc637c"),
            ("vae", 248, 5252487L, "4a45d278169c6a8f1b77f7c404e6ac4f4790327f7f6f386825fea9fdfc5eb4fb") })
        {
            var schema = root.GetProperty("parameterSchemas").GetProperty(name);
            Require(schema.GetProperty("tensorCount").GetInt32() == count && schema.GetProperty("elements").GetInt64() == elements &&
                schema.GetProperty("residentBytes").GetInt64() == elements * 4 && String(schema, "nameShapeSchemaSha256") == digest,
                "Parameter schema descriptor differs.");
        }
        var pipeline = root.GetProperty("pipeline");
        Require(String(pipeline, "clipProfile") == "sd1-l" && !pipeline.GetProperty("projectPooled").GetBoolean() &&
            String(pipeline, "prediction") == "EPS, sigma_data=1" &&
            Bits64(pipeline.GetProperty("latentScale").GetDouble()) == 0x3fc750b0f27bb2ffUL,
            "Pipeline options differ.");
        Require(String(root.GetProperty("sigmaMaximum"), "float32BitsHex") == "4169d592", "Maximum sigma pin differs.");
        var comparison = root.GetProperty("comparison");
        Require(comparison.GetProperty("absoluteTolerance").GetDouble() == 3e-5 &&
            comparison.GetProperty("relativeTolerance").GetDouble() == 3e-5, "Prospective comparison profile differs.");
        // The full resource SHA also pins every remaining config, source/helper/lock and recipe field.
        // This loader validates descriptors; it does not claim to validate a live model parameter bank.
    }

    private static void VerifyPayload(JsonElement record, string dtype, int[] shape, byte[] bytes)
    {
        Require(String(record, "dtype") == dtype && String(record, "encoding") == "little-endian row-major",
            "Input dtype/encoding differs.");
        Shape(record.GetProperty("shape"), shape);
        Require(record.GetProperty("bytes").GetInt32() == bytes.Length && String(record, "sha256") == Hash(bytes),
            "Input payload byte count/SHA differs.");
    }
    private static void Shape(JsonElement actual, int[] expected) =>
        Require(actual.EnumerateArray().Select(v => v.GetInt32()).SequenceEqual(expected), "Protocol shape differs.");
    private static ulong Bits64(double value) => unchecked((ulong)BitConverter.DoubleToInt64Bits(value));
    private static ulong Parse64(string value)
    {
        Require(value.Length == 16, "Malformed binary64 bits.");
        return ulong.Parse(value, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture);
    }
    private static string String(JsonElement parent, string name) => parent.GetProperty(name).GetString()
        ?? throw new InvalidDataException("Null protocol string.");
    private static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidDataException(message);
    }
}
