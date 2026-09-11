using System.Text;
using System.Text.Json;
using ComfySharp.Tokenization;

namespace ComfySharp.Tokenize;

public static class TokenizeCommand
{
    private const string Usage = "ComfySharp.Tokenize (--text <prompt> | --file <UTF-8-file> | --stdin) [--profile sd1-l|sdxl-l|sdxl-g|sdxl] [--disable-weights]";

    public static async Task<int> RunAsync(string[] args, TextReader input, TextWriter output, TextWriter error, CancellationToken cancellationToken = default)
    {
        if (args is ["--help"] or ["-h"])
        {
            await output.WriteLineAsync(Usage);
            await output.WriteLineAsync("Outputs CLIP token IDs, decoded text, weighted chunks and word IDs using embedded resources. Does not load a model or resolve textual inversions.");
            return 0;
        }
        try
        {
            string? text = null, file = null;
            var fromStdin = false;
            var profile = "sd1-l";
            var disableWeights = false;
            var seen = new HashSet<string>(StringComparer.Ordinal);
            for (var i = 0; i < args.Length; i++)
            {
                var flag = args[i];
                if (!seen.Add(flag)) throw new ArgumentException("Repeated argument.");
                switch (flag)
                {
                    case "--text": text = Value(args, ref i); break;
                    case "--file": file = Value(args, ref i); break;
                    case "--profile": profile = Value(args, ref i); break;
                    case "--stdin": fromStdin = true; break;
                    case "--disable-weights": disableWeights = true; break;
                    default: throw new ArgumentException("Unknown argument.");
                }
            }
            if ((text is not null ? 1 : 0) + (file is not null ? 1 : 0) + (fromStdin ? 1 : 0) != 1)
                throw new ArgumentException("Choose exactly one input: --text, --file or --stdin.");
            var singleProfile = profile switch
            {
                "sd1-l" => ClipProfile.Sd1L, "sdxl-l" => ClipProfile.SdXlL,
                "sdxl-g" or "sdxl" => ClipProfile.SdXlG,
                _ => throw new ArgumentException("Unknown CLIP profile.")
            };
            cancellationToken.ThrowIfCancellationRequested();
            if (file is not null)
            {
                // StreamReader's BOM auto-detection can substitute a permissive
                // UTF-16 decoder even when a strict UTF-8 Encoding is supplied.
                var bytes = await File.ReadAllBytesAsync(file, cancellationToken);
                var offset = bytes.AsSpan().StartsWith(new byte[] { 0xef, 0xbb, 0xbf }) ? 3 : 0;
                text = new UTF8Encoding(false, true).GetString(bytes, offset, bytes.Length - offset);
            }
            if (fromStdin) text = await input.ReadToEndAsync(cancellationToken);
            var clip = new ClipTokenizer();
            var rawIds = clip.Encode(text!, cancellationToken: cancellationToken);
            var call = new ClipTokenizeOptions { ReturnWordIds = true, DisableWeights = disableWeights };
            IReadOnlyDictionary<string, ClipTokenization> results = profile == "sdxl"
                ? new ComfySdxlTokenizer(clip).Tokenize(text!, call, cancellationToken)
                : new Dictionary<string, ClipTokenization>
                {
                    [singleProfile == ClipProfile.SdXlG ? "g" : "l"] = new ComfyClipTokenizer(clip, singleProfile).Tokenize(text!, call, cancellationToken)
                };
            var report = new
            {
                status = "ok", profile, dependencyProfile = "clip-transformers-5.14.1-tokenizers-0.22.2",
                modelCompatibility = "not_assessed", textualInversions = "not_resolved_text_only",
                ids = rawIds, decoded = clip.Decode(rawIds),
                outputs = results.ToDictionary(pair => pair.Key, pair => pair.Value.Chunks.Select(chunk => chunk.Select(token => new
                {
                    id = token.Id, weight = DisplayWeight(token.Weight),
                    weightBits = unchecked((ulong)BitConverter.DoubleToInt64Bits(token.Weight)).ToString("x16"), wordId = token.WordId
                }).ToArray()).ToArray())
            };
            cancellationToken.ThrowIfCancellationRequested();
            var serialized = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
            await output.WriteLineAsync(serialized.AsMemory(), cancellationToken);
            return 0;
        }
        catch (OperationCanceledException)
        {
            await error.WriteLineAsync("CLIP tokenization cancelled.");
            return 130;
        }
        catch (Exception exception) when (exception is ArgumentException or IOException or UnauthorizedAccessException or InvalidDataException or NotSupportedException)
        {
            await error.WriteLineAsync("CLIP tokenization failed: " + exception.Message);
            return 2;
        }
    }

    private static string Value(string[] args, ref int index)
    {
        if (++index == args.Length) throw new ArgumentException("Missing argument value.");
        return args[index];
    }

    private static object DisplayWeight(double value)
    {
        if (double.IsFinite(value)) return value;
        if (double.IsPositiveInfinity(value)) return "Infinity";
        if (double.IsNegativeInfinity(value)) return "-Infinity";
        return BitConverter.DoubleToInt64Bits(value) < 0 ? "-NaN" : "NaN";
    }
}
