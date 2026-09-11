using System.Buffers.Binary;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using ComfySharp.Inference;
using TorchSharp;

namespace ComfySharp.Inference.Tests;

/// <summary>Version 1 laboratory input recipe only; never computes expected model outputs.</summary>
internal static class SdSyntheticInputs
{
    internal static float[] Values(string name, IReadOnlyList<long> shape, bool parameter = false)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(shape);
        long length = 1;
        foreach (long dimension in shape)
        {
            if (dimension < 0) throw new ArgumentOutOfRangeException(nameof(shape));
            length = checked(length * dimension);
        }
        var result = new float[checked((int)length)];
        uint seed = BinaryPrimitives.ReadUInt32LittleEndian(SHA256.HashData(Encoding.UTF8.GetBytes(name)));
        float offset = 0;
        int exponent = -15;
        if (parameter && shape.Count == 1)
        {
            bool normWeight = name.EndsWith(".weight", StringComparison.Ordinal);
            offset = normWeight ? 1 : 0;
            exponent = normWeight ? -18 : -20;
        }
        else if (parameter)
        {
            long fan = 1;
            for (int i = 1; i < shape.Count; i++) fan = checked(fan * shape[i]);
            if (fan <= 0) throw new ArgumentOutOfRangeException(nameof(shape));
            int ceilLog2 = fan == 1 ? 0 : BitOperations.Log2((ulong)(fan - 1)) + 1;
            exponent = -15 - (ceilLog2 + 1) / 2;
        }
        float scale = MathF.ScaleB(1, exponent);
        for (int i = 0; i < result.Length; i++)
        {
            int sample = (int)(unchecked((uint)i * 1664525U + seed) >> 16) - 32768;
            result[i] = offset + sample * scale;
        }
        return result;
    }

    internal static UnetWeightSet CreateUnet(SdUnetConfig config)
        => Create(UnetWeightSchema.Describe(config), weights => UnetWeightSet.FromOwnedTensors(config, weights));

    internal static ClassicalVaeWeightSet CreateVae(ClassicalVaeConfig config)
        => Create(ClassicalVaeWeightSchema.Describe(config), weights => ClassicalVaeWeightSet.FromOwnedTensors(config, weights));

    private static T Create<T>(IReadOnlyDictionary<string, IReadOnlyList<long>> schema,
        Func<IReadOnlyDictionary<string, torch.Tensor>, T> transfer)
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = torch.NewDisposeScope();
        var weights = new Dictionary<string, torch.Tensor>(StringComparer.Ordinal);
        foreach (var (name, shape) in schema)
            weights.Add(name, torch.tensor(Values(name, shape, parameter: true), shape.ToArray(), dtype: torch.ScalarType.Float32));
        return transfer(weights);
    }
}
