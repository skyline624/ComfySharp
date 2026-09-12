using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

namespace ComfySharp.Testing;

internal static class PngMetadataFixture
{
    // GPL-3.0-only: ComfyUI_frontend e7d1c7fc6823e330fdab524610b0000394cb1dbc,
    // src/scripts/metadata/__fixtures__/with_metadata.png (exact 223-byte fixture).
    public const string UpstreamSha256 = "393da603c10b778089e4c48cae80c12469f03da18fbc87aa76879256e88ae1a9";
    public static byte[] Upstream() => Convert.FromBase64String("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAUHRFWHR3b3JrZmxvdwB7Im5vZGVzIjpbeyJpZCI6MSwidHlwZSI6IktTYW1wbGVyIiwicG9zIjpbMTAwLDEwMF0sInNpemUiOlsyMDAsMjAwXX1dfTWSGyoAAAAydEVYdHByb21wdAB7IjEiOnsiY2xhc3NfdHlwZSI6IktTYW1wbGVyIiwiaW5wdXRzIjp7fX19FlamhwAAAAxJREFUeJxj+M/AAAADAQEAyf6S7wAAAABJRU5ErkJggg==");
    public static byte[] Build(params (string Type, byte[] Data)[] chunks)
    {
        using var output = new MemoryStream(); output.Write(new byte[] { 137,80,78,71,13,10,26,10 });
        var header = new byte[13]; header[3] = header[7] = 1; header[8] = 8; header[9] = 2;
        Chunk(output, "IHDR", header); Chunk(output, "IDAT", Compress(new byte[4]));
        foreach (var (type, data) in chunks) Chunk(output, type, data);
        Chunk(output, "IEND", []); return output.ToArray();
    }
    public static byte[] Text(string type, string key, string value, byte flag = 0, byte method = 0)
    {
        using var output = new MemoryStream(); output.Write(Encoding.Latin1.GetBytes(key)); output.WriteByte(0);
        if (type == "iTXt") { output.WriteByte(flag); output.WriteByte(method); output.Write("fr\0Workflow\0"u8); }
        byte[] body = Encoding.UTF8.GetBytes(value); output.Write(flag == 1 ? Compress(body) : body); return output.ToArray();
    }
    public static byte[] Compress(byte[] bytes)
    {
        using var output = new MemoryStream(); using (var zlib = new ZLibStream(output, CompressionLevel.Optimal, true)) zlib.Write(bytes);
        return output.ToArray();
    }
    private static void Chunk(Stream output, string type, byte[] data)
    {
        Span<byte> word = stackalloc byte[4]; BinaryPrimitives.WriteInt32BigEndian(word, data.Length); output.Write(word);
        byte[] kind = Encoding.ASCII.GetBytes(type); output.Write(kind); output.Write(data);
        uint crc = uint.MaxValue;
        foreach (byte value in kind.Concat(data)) { crc ^= value; for (int bit = 0; bit < 8; bit++) crc = (crc & 1) == 0 ? crc >> 1 : crc >> 1 ^ 0xedb88320; }
        BinaryPrimitives.WriteUInt32BigEndian(word, ~crc); output.Write(word);
    }
}
