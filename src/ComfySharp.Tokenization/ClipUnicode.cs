using System.Buffers;
using System.Text;
using System.Text.Json;

namespace ComfySharp.Tokenization;

/// <summary>Scalar Unicode behavior captured from the pinned tokenizer dependency profile.</summary>
internal sealed class ClipUnicode
{
    internal const string ResourceHash = "ff294f10809b7012d1392387bd516859a2aa0d79d33a139ad6eb1d5d8002b842";
    private readonly int[][] letters, numbers, whitespace;
    private readonly Dictionary<int, int[]> lower, decomposition;
    private readonly Dictionary<int, int> combining;
    private readonly Dictionary<(int, int), int> composition;

    internal ClipUnicode(byte[] resource)
    {
        using var document = JsonDocument.Parse(resource);
        var root = document.RootElement;
        letters = Ranges("letters"); numbers = Ranges("numbers"); whitespace = Ranges("whitespace");
        lower = Map("lower"); decomposition = Map("decomposition");
        combining = root.GetProperty("combining").EnumerateObject().ToDictionary(p => int.Parse(p.Name), p => p.Value.GetInt32());
        composition = root.GetProperty("composition").EnumerateObject().ToDictionary(p =>
        {
            var pair = p.Name.Split(','); return (int.Parse(pair[0]), int.Parse(pair[1]));
        }, p => p.Value.GetInt32());
        int[][] Ranges(string key) => root.GetProperty(key).EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        Dictionary<int, int[]> Map(string key) => root.GetProperty(key).EnumerateObject().ToDictionary(p => int.Parse(p.Name), p => p.Value.EnumerateArray().Select(v => v.GetInt32()).ToArray());
    }

    internal bool IsLetter(int scalar) => Contains(letters, scalar);
    internal bool IsNumber(int scalar) => Contains(numbers, scalar);
    internal bool IsWhitespace(int scalar) => Contains(whitespace, scalar);
    private int Class(int scalar) => combining.GetValueOrDefault(scalar);

    private static bool Contains(int[][] ranges, int scalar)
    {
        var left = 0; var right = ranges.Length - 1;
        while (left <= right)
        {
            var mid = (left + right) / 2;
            if (scalar < ranges[mid][0]) right = mid - 1;
            else if (scalar > ranges[mid][1]) left = mid + 1;
            else return true;
        }
        return false;
    }

    internal string Normalize(string text, CancellationToken cancellationToken)
    {
        var decomposed = new List<int>(text.Length);
        for (int i = 0, scalarCount = 0; i < text.Length; scalarCount++)
        {
            if ((scalarCount & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            if (Rune.DecodeFromUtf16(text.AsSpan(i), out var rune, out var consumed) != OperationStatus.Done)
                throw new ArgumentException("CLIP input must contain valid Unicode scalar values; unpaired UTF-16 surrogates are unsupported.", nameof(text));
            i += consumed;
            if (decomposition.TryGetValue(rune.Value, out var expansion)) decomposed.AddRange(expansion);
            else decomposed.Add(rune.Value);
        }
        // Canonical ordering is stable within each run of non-starters. Sorting
        // avoids quadratic insertion behavior on adversarial combining strings.
        for (int i = 0; i < decomposed.Count;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (Class(decomposed[i]) == 0) { i++; continue; }
            var start = i;
            while (i < decomposed.Count && Class(decomposed[i]) != 0)
            { if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested(); i++; }
            if (i - start > 1)
            {
                var counts = new int[256];
                for (var j = start; j < i; j++) { if ((j & 1023) == 0) cancellationToken.ThrowIfCancellationRequested(); counts[Class(decomposed[j])]++; }
                var position = 0;
                for (var j = 0; j < counts.Length; j++) { var count = counts[j]; counts[j] = position; position += count; }
                var ordered = new int[i - start];
                for (var j = start; j < i; j++) { if ((j & 1023) == 0) cancellationToken.ThrowIfCancellationRequested(); ordered[counts[Class(decomposed[j])]++] = decomposed[j]; }
                for (var j = 0; j < ordered.Length; j++) decomposed[start + j] = ordered[j];
            }
        }
        var composed = new List<int>(decomposed.Count);
        int starter = -1, lastClass = 0;
        foreach (var scalar in decomposed)
        {
            if ((composed.Count & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            var currentClass = Class(scalar);
            if (starter >= 0 && (lastClass == 0 || lastClass < currentClass) && composition.TryGetValue((composed[starter], scalar), out var composite))
                composed[starter] = composite;
            else
            {
                if (currentClass == 0) starter = composed.Count;
                composed.Add(scalar); lastClass = currentClass;
            }
        }
        var result = new StringBuilder(text.Length);
        bool previousWhitespace = false;
        for (var i = 0; i < composed.Count; i++)
        {
            if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            var scalar = composed[i];
            if (IsWhitespace(scalar))
            { if (!previousWhitespace) result.Append(' '); previousWhitespace = true; continue; }
            previousWhitespace = false;
            if (lower.TryGetValue(scalar, out var expansion))
                foreach (var lowered in expansion) result.Append(new Rune(lowered));
            else result.Append(new Rune(scalar));
        }
        cancellationToken.ThrowIfCancellationRequested();
        return result.ToString();
    }
}
