using System.Buffers.Binary;
using System.Text;
using System.Text.Json.Nodes;

namespace ComfySharp.Host;

public static class PreviewCodec
{
    public static byte[] Image(ReadOnlySpan<byte> encodedImage, bool png)
    {
        var buffer = new byte[checked(8 + encodedImage.Length)];
        BinaryPrimitives.WriteUInt32BigEndian(buffer, 1);
        BinaryPrimitives.WriteUInt32BigEndian(buffer.AsSpan(4), png ? 2u : 1u);
        encodedImage.CopyTo(buffer.AsSpan(8));
        return buffer;
    }

    public static byte[] ImageWithMetadata(ReadOnlySpan<byte> encodedImage, JsonObject metadata)
    {
        var json = Encoding.UTF8.GetBytes(metadata.ToJsonString());
        var buffer = new byte[checked(8 + json.Length + encodedImage.Length)];
        BinaryPrimitives.WriteUInt32BigEndian(buffer, 4);
        BinaryPrimitives.WriteUInt32BigEndian(buffer.AsSpan(4), (uint)json.Length);
        json.CopyTo(buffer.AsSpan(8));
        encodedImage.CopyTo(buffer.AsSpan(8 + json.Length));
        return buffer;
    }
}
