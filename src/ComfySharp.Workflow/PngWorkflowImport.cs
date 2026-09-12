using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.Json.Nodes;

namespace ComfySharp.Workflow;

public sealed record PngMetadata(IReadOnlyDictionary<string, string> Text, IReadOnlyList<string> Warnings);

/// <summary>PNG metadata extraction independent of pixels, native codecs and the editor.</summary>
public static class PngWorkflowImport
{
    public const int MaximumTextBytes = 16 * 1024 * 1024;
    public const long MaximumFileBytes = 128L * 1024 * 1024;
    private static readonly uint[] CrcTable = Enumerable.Range(0, 256).Select(value =>
    {
        uint crc = (uint)value;
        for (int bit = 0; bit < 8; bit++) crc = (crc & 1) == 0 ? crc >> 1 : (crc >> 1) ^ 0xedb88320;
        return crc;
    }).ToArray();

    public static async Task<PngMetadata> ReadMetadataAsync(Stream stream, CancellationToken cancellationToken = default)
    {
        var text = new Dictionary<string, string>(StringComparer.Ordinal); var warnings = new List<string>();
        long total = 0; int retainedBytes = 0, textChunkBytes = 0; bool first = true;
        async Task Exact(Memory<byte> buffer)
        {
            if (total + buffer.Length > MaximumFileBytes) throw new InvalidDataException("PNG exceeds the import file size limit.");
            await stream.ReadExactlyAsync(buffer, cancellationToken).ConfigureAwait(false); total += buffer.Length;
        }
        var header = new byte[8]; await Exact(header).ConfigureAwait(false);
        if (!header.AsSpan().SequenceEqual(new byte[] { 137,80,78,71,13,10,26,10 })) throw new InvalidDataException("Invalid PNG signature.");
        var buffer = new byte[64 * 1024]; var checksum = new byte[4];
        while (true)
        {
            await Exact(header).ConfigureAwait(false);
            uint length = BinaryPrimitives.ReadUInt32BigEndian(header); string type = Encoding.ASCII.GetString(header, 4, 4);
            if (first && (type != "IHDR" || length != 13)) throw new InvalidDataException("PNG must begin with IHDR.");
            first = false;
            if (total + length + 4 > MaximumFileBytes) throw new InvalidDataException("PNG chunk exceeds the import file size limit.");
            bool metadata = type is "tEXt" or "comf" or "iTXt";
            if (metadata && length > MaximumTextBytes - textChunkBytes) throw new InvalidDataException("PNG metadata exceeds the text size limit.");
            if (metadata) textChunkBytes += (int)length;
            using var chunk = metadata ? new MemoryStream((int)length) : null;
            uint crc = uint.MaxValue; foreach (byte value in header.AsSpan(4).ToArray()) crc = CrcTable[(crc ^ value) & 255] ^ (crc >> 8);
            for (long remaining = length; remaining > 0;)
            {
                int count = (int)Math.Min(buffer.Length, remaining); await Exact(buffer.AsMemory(0, count)).ConfigureAwait(false);
                for (int i = 0; i < count; i++) crc = CrcTable[(crc ^ buffer[i]) & 255] ^ (crc >> 8);
                chunk?.Write(buffer, 0, count); remaining -= count;
            }
            await Exact(checksum).ConfigureAwait(false);
            if (~crc != BinaryPrimitives.ReadUInt32BigEndian(checksum)) throw new InvalidDataException($"Invalid PNG checksum for {type}.");
            if (chunk is not null)
            {
                var decoded = Decode(type, chunk.ToArray(), warnings, MaximumTextBytes - retainedBytes, cancellationToken);
                if (decoded is { } item) { retainedBytes += item.Bytes; text[item.Key] = item.Value; }
            }
            if (type == "IEND")
            {
                if (length != 0) throw new InvalidDataException("Invalid PNG IEND length.");
                return new(text, warnings);
            }
        }
    }

    private static (string Key, string Value, int Bytes)? Decode(string type, byte[] data, List<string> warnings, int budget, CancellationToken token)
    {
        int offset = 0;
        void Warn(string message)
        {
            if (warnings.Count < 100) warnings.Add(message);
            else if (warnings.Count == 100) warnings.Add("Further PNG metadata warnings omitted.");
        }
        byte[] Field()
        {
            int end = Array.IndexOf(data, (byte)0, offset);
            if (end < 0) throw new InvalidDataException("Unterminated PNG text field.");
            var field = data[offset..end]; offset = end + 1; return field;
        }
        byte[] keyword = Field();
        if (keyword.Length is < 1 or > 79) throw new InvalidDataException("Invalid PNG text keyword length.");
        string key = Encoding.Latin1.GetString(keyword); bool compressed = false;
        if (type == "iTXt")
        {
            if (offset + 2 > data.Length) throw new InvalidDataException("Truncated PNG international text header.");
            byte flag = data[offset++], method = data[offset++]; compressed = flag == 1;
            if (flag > 1) throw new InvalidDataException("Invalid PNG text compression flag.");
            Field(); Field(); // Language and translated keyword are not JSON metadata keys.
            if (compressed && method != 0) { Warn($"Skipped {key}: unsupported iTXt compression method {method}."); return null; }
        }
        byte[] content;
        if (compressed)
        {
            using var source = new MemoryStream(data, offset, data.Length - offset, writable: false);
            using var zlib = new ZLibStream(source, CompressionMode.Decompress); using var output = new MemoryStream();
            var buffer = new byte[8192];
            try
            {
                if (data.Length - offset < 6) throw new InvalidDataException("Truncated zlib text.");
                while (true)
                {
                    token.ThrowIfCancellationRequested(); int count = zlib.Read(buffer); if (count == 0) break;
                    if (output.Length + count > budget) throw new TextLimitException(); output.Write(buffer, 0, count);
                }
                content = output.ToArray();
                uint a = 1, b = 0;
                foreach (byte element in content) { a = (a + element) % 65521; b = (b + a) % 65521; }
                if ((b << 16 | a) != BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(data.Length - 4)))
                    throw new InvalidDataException("Invalid zlib text checksum.");
            }
            catch (InvalidDataException) { Warn($"Skipped {key}: invalid compressed iTXt text."); return null; }
        }
        else content = data[offset..];
        if (content.Length > budget) throw new TextLimitException();
        // The pinned frontend uses UTF-8 TextDecoder for every recognized type, including legacy tEXt/comf.
        string value = Encoding.UTF8.GetString(content);
        if (value.StartsWith('\uFEFF')) value = value[1..];
        return (key, value, content.Length);
    }

    private sealed class TextLimitException() : IOException("PNG metadata exceeds the aggregate text size limit.");

    public static WorkflowDocument ReadWorkflow(PngMetadata metadata, Func<string, JsonObject>? templateFactory = null, Action<string>? reportWarning = null)
    {
        Exception? failure = null;
        if (metadata.Text.TryGetValue("workflow", out string? workflow) && workflow.Length != 0)
        {
            try
            {
                var document = ImportJson.Parse(workflow, reportWarning) as JsonObject ?? throw new FormatException("PNG workflow must be a JSON object.");
                // Older PNG graphs (including the pinned frontend fixture) omit version.
                if (!document.ContainsKey("version")) document["version"] = 0.4;
                return WorkflowDocument.Parse(document.ToJsonString());
            }
            catch (Exception error) when (error is System.Text.Json.JsonException or FormatException or InvalidOperationException)
            {
                reportWarning?.Invoke("PNG workflow could not be imported; trying API prompt. " + error.Message); failure = error;
            }
        }
        if (metadata.Text.TryGetValue("prompt", out string? prompt) && prompt.Length != 0)
        {
            try { return ApiPromptImport.Parse(prompt, templateFactory, reportWarning); }
            catch (Exception error) when (error is System.Text.Json.JsonException or FormatException)
            {
                reportWarning?.Invoke("PNG API prompt could not be imported. " + error.Message); failure = error;
            }
        }
        if (metadata.Text.ContainsKey("parameters"))
            throw new NotSupportedException("This PNG has no workflow or API prompt. A1111 reconstruction is not yet available.");
        if (failure is not null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failure).Throw();
        throw new InvalidDataException("This PNG contains no graphical workflow metadata.");
    }
}
