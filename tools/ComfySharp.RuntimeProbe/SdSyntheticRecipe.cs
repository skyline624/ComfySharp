using System.Buffers.Binary;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

namespace ComfySharp.RuntimeProbe;

/// <summary>Input recipe only. No model outputs or random generator are involved.</summary>
internal static class SdSyntheticRecipe
{
    internal const string Version = "sha256-name-lcg-high16-power2-v1";
    internal readonly record struct Descriptor(uint Seed, float Offset, int Exponent, long Elements);

    internal static Descriptor Describe(string name, IReadOnlyList<long> shape, bool parameter)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Count == 0) throw new ArgumentException("A synthetic tensor requires a positive rank.", nameof(shape));
        long elements = 1, fan = 1;
        for (int i = 0; i < shape.Count; i++)
        {
            if (shape[i] <= 0) throw new ArgumentOutOfRangeException(nameof(shape));
            elements = checked(elements * shape[i]);
            if (i > 0) fan = checked(fan * shape[i]);
        }
        uint seed = BinaryPrimitives.ReadUInt32LittleEndian(SHA256.HashData(Encoding.UTF8.GetBytes(name)));
        float offset = 0;
        int exponent = -15;
        if (parameter && shape.Count == 1)
        {
            bool weight = name.EndsWith(".weight", StringComparison.Ordinal);
            offset = weight ? 1 : 0;
            exponent = weight ? -18 : -20;
        }
        else if (parameter)
        {
            int ceilLog2 = fan == 1 ? 0 : BitOperations.Log2((ulong)(fan - 1)) + 1;
            exponent -= (ceilLog2 + 1) / 2;
        }
        return new(seed, offset, exponent, elements);
    }

    internal static void Fill(Span<float> destination, long globalStart, Descriptor descriptor)
    {
        if (globalStart < 0 || globalStart > descriptor.Elements || destination.Length > descriptor.Elements - globalStart)
            throw new ArgumentOutOfRangeException(nameof(globalStart));
        uint word = unchecked((uint)globalStart * 1664525U + descriptor.Seed);
        float scale = MathF.ScaleB(1, descriptor.Exponent);
        for (int i = 0; i < destination.Length; i++)
        {
            int sample = (int)(word >> 16) - 32768;
            destination[i] = descriptor.Offset + sample * scale;
            word = unchecked(word + 1664525U);
        }
    }
}
