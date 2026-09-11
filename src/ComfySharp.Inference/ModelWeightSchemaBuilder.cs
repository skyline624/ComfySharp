using System.Collections.ObjectModel;

namespace ComfySharp.Inference;

internal sealed class ModelWeightSchemaBuilder
{
    internal readonly Dictionary<string, IReadOnlyList<long>> Shapes = new(StringComparer.Ordinal);
    internal readonly Dictionary<string, (string Canonical, bool Reshape)> Diffusers = new(StringComparer.Ordinal);
    internal IReadOnlyDictionary<string, IReadOnlyList<long>> ReadOnlyShapes => new ReadOnlyDictionary<string, IReadOnlyList<long>>(Shapes);

    internal void Add(string name, string diffusers, bool reshape, params long[] shape)
    {
        Shapes.Add(name, Array.AsReadOnly(shape));
        Diffusers.Add(diffusers, (name, reshape));
    }

    internal void Affine(string name, string diffusers, long output, long? input = null, int kernel = 0, bool bias = true)
    {
        long[] shape = input is null ? [output] : kernel == 0 ? [output, input.Value] : [output, input.Value, kernel, kernel];
        Add(name + ".weight", diffusers + ".weight", false, shape);
        if (bias) Add(name + ".bias", diffusers + ".bias", false, output);
    }

    internal static void CheckShape(string name, IReadOnlyList<long> actual, IReadOnlyList<long> expected)
    {
        if (!actual.SequenceEqual(expected))
            throw new InvalidDataException($"Weight '{name}' has shape [{string.Join(',', actual)}]; expected [{string.Join(',', expected)}].");
    }
}
