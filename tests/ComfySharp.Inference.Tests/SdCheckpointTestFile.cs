using System.Buffers.Binary;
using System.Text.Json;
using ComfySharp.Inference;

namespace ComfySharp.Inference.Tests;

/// <summary>Small real combined safetensors file; metadata and payloads are written without Torch.
/// Reduced dimensions test assembly contracts, never stock loading or model compatibility.</summary>
internal sealed class SdCheckpointTestFile : IDisposable
{
    internal sealed record Entry(string Name, long[] Shape, string DType = "F32", float Offset = 0);
    internal static SdCheckpointAssemblyProfile ReducedProfile { get; } = new(
        new(16, 32, 2, 4, ClipActivation.QuickGelu),
        new(32, 16, SdAttentionHeadMode.FixedCount, 4, false), new(32));
    internal string Path { get; }
    internal IReadOnlyList<Entry> Entries { get; }

    private SdCheckpointTestFile(string path, IReadOnlyList<Entry> entries) { Path = path; Entries = entries; }

    internal static float ExpectedValue(Entry entry, long index) => index < 16 ? (index - 7) * 0.125f + entry.Offset : 0;

    internal static SdCheckpointTestFile Create(bool includeProjection = false, bool mixedDTypes = false,
        Action<List<Entry>>? edit = null)
    {
        var entries = ClipWeightSchema.Describe(ReducedProfile.Clip)
            .Where(p => includeProjection || p.Key != ClipWeightSchema.Projection)
            .Select(p => new Entry("cond_stage_model.transformer." + p.Key, p.Value.ToArray(),
                mixedDTypes ? "F16" : "F32", 0.25f)).ToList();
        entries.AddRange(UnetWeightSchema.Describe(ReducedProfile.Unet).Select(p =>
            new Entry("model.diffusion_model." + p.Key, p.Value.ToArray(), mixedDTypes ? "BF16" : "F32", 0.5f)));
        entries.AddRange(ClassicalVaeWeightSchema.Describe(ReducedProfile.Vae).Select(p =>
            new Entry("first_stage_model." + p.Key, p.Value.ToArray(), "F32", -0.25f)));
        edit?.Invoke(entries);
        string path = System.IO.Path.GetTempFileName();
        try
        {
            var header = new Dictionary<string, object>(StringComparer.Ordinal);
            var starts = new List<(Entry Entry, long Start, long Elements, int Width)>();
            long offset = 0;
            foreach (var entry in entries)
            {
                int width = entry.DType switch { "F16" or "BF16" => 2, "F64" or "I64" => 8, _ => 4 };
                long elements = entry.Shape.Aggregate(1L, (n, d) => checked(n * d));
                long end = checked(offset + elements * width);
                header.Add(entry.Name, new { dtype = entry.DType, shape = entry.Shape, data_offsets = new[] { offset, end } });
                starts.Add((entry, offset, elements, width));
                offset = end;
            }
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
            using var stream = File.Create(path);
            Span<byte> prefix = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)json.Length);
            stream.Write(prefix); stream.Write(json);
            stream.SetLength(checked(8 + json.Length + offset));
            foreach (var item in starts)
            {
                byte[] bytes = new byte[checked((int)Math.Min(item.Elements, 16) * item.Width)];
                for (int i = 0; i < bytes.Length / item.Width; i++)
                {
                    float value = ExpectedValue(item.Entry, i);
                    var target = bytes.AsSpan(i * item.Width, item.Width);
                    switch (item.Entry.DType)
                    {
                        case "F32": BinaryPrimitives.WriteSingleLittleEndian(target, value); break;
                        case "F16": BinaryPrimitives.WriteUInt16LittleEndian(target, BitConverter.HalfToUInt16Bits((Half)value)); break;
                        case "BF16": BinaryPrimitives.WriteUInt16LittleEndian(target, (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16)); break;
                    }
                }
                stream.Position = checked(8 + json.Length + item.Start);
                stream.Write(bytes);
            }
            return new(path, entries.AsReadOnly());
        }
        catch { File.Delete(path); throw; }
    }

    public void Dispose() => File.Delete(Path);
}
