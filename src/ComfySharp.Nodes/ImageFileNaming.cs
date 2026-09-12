using System.Globalization;
using System.Numerics;
using System.Text;
using ComfySharp.Contracts;

namespace ComfySharp.Nodes;

/// <summary>Local SaveImage filename rules from folder_paths.get_save_image_path and nodes.SaveImage.
/// Uses native path separators and the pinned Python lowercase tables on Windows.</summary>
public static class ImageFileNaming
{
    public sealed record Plan(string Filename, string Subfolder, string ExpandedPrefix, BigInteger Counter)
    {
        public ImageFileDescriptor Frame(string type, long batchIndex) => new(
            Filename.Replace("%batch_num%", batchIndex.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal) +
            "_" + FormatCounter(Counter + batchIndex) + "_.png", Subfolder, type);
    }

    public static Plan Prepare(ILocalFileStore store, string type, string prefix, int width, int height,
        DateTimeOffset localTime, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(prefix);
        PythonUnicodeLower.ValidateUnicode(prefix, cancellationToken);
        string expanded = Expand(prefix, width, height, localTime);
        string normalized = Normalize(expanded);
        string filename = Path.GetFileName(normalized);
        string subfolder = Path.GetDirectoryName(normalized) ?? "";
        // Source uses basename of the un-normalized expanded prefix for the scan's slice length.
        int length = Path.GetFileName(expanded).EnumerateRunes().Count();
        var entries = store.PrepareDirectory(type, subfolder, cancellationToken);
        string match = NormCase(filename, cancellationToken);
        BigInteger? maximum = null;
        foreach (string entry in entries)
        {
            cancellationToken.ThrowIfCancellationRequested();
            PythonUnicodeLower.ValidateUnicode(entry, cancellationToken);
            int end = RuneOffset(entry, length + 1);
            string candidate = entry[..end];
            if (candidate.Length == 0 || candidate[^1] != '_' || NormCase(candidate[..^1], cancellationToken) != match) continue;
            string rest = entry[end..].Split('.')[0].Split('_')[0];
            BigInteger counter = ParseCounter(rest);
            if (maximum is null || counter > maximum) maximum = counter;
        }
        return new(filename, subfolder, expanded, maximum is null ? BigInteger.One : maximum.Value + 1);
    }

    private static string Expand(string prefix, int width, int height, DateTimeOffset time)
    {
        string result = prefix.Replace("%width%", width.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal)
            .Replace("%height%", height.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal);
        foreach (var (key, value, digits) in new[] { ("year", time.Year, 1), ("month", time.Month, 2), ("day", time.Day, 2),
                     ("hour", time.Hour, 2), ("minute", time.Minute, 2), ("second", time.Second, 2) })
            result = result.Replace("%" + key + "%", value.ToString("D" + digits, CultureInfo.InvariantCulture), StringComparison.Ordinal);
        return result;
    }

    // normpath is lexical: unlike GetFullPath, a relative prefix stays relative and an empty one becomes '.'.
    private static string Normalize(string path)
    {
        if (OperatingSystem.IsWindows()) path = path.Replace('/', '\\');
        char separator = Path.DirectorySeparatorChar;
        string root = Path.GetPathRoot(path) ?? "";
        if (!OperatingSystem.IsWindows() && path.StartsWith("//", StringComparison.Ordinal) && !path.StartsWith("///", StringComparison.Ordinal)) root = "//";
        bool rooted = root.EndsWith(separator);
        var parts = new List<string>();
        foreach (string part in path[root.Length..].Split(separator))
        {
            if (part is "" or ".") continue;
            if (part == ".." && parts.Count > 0 && parts[^1] != "..") parts.RemoveAt(parts.Count - 1);
            else if (part != ".." || !rooted) parts.Add(part);
        }
        string normalized = root + string.Join(separator, parts);
        return normalized.Length == 0 ? "." : normalized;
    }

    private static string NormCase(string text, CancellationToken token) => OperatingSystem.IsWindows()
        ? PythonUnicodeLower.Lower(text.Replace('/', '\\'), token) : text;

    private static int RuneOffset(string text, int count)
    {
        int offset = 0;
        foreach (Rune rune in text.EnumerateRunes())
        {
            if (count-- == 0) break;
            offset += rune.Utf16SequenceLength;
        }
        return offset;
    }

    public static string FormatCounter(BigInteger counter)
    {
        string magnitude = BigInteger.Abs(counter).ToString(CultureInfo.InvariantCulture);
        return counter.Sign < 0 ? "-" + magnitude.PadLeft(4, '0') : magnitude.PadLeft(5, '0');
    }

    private static BigInteger ParseCounter(string text)
    {
        // The source splits on '_' before int(), so integer underscores never reach this parser.
        ReadOnlySpan<char> value = text;
        while (!value.IsEmpty && IntegerWhitespace(value[0])) value = value[1..];
        while (!value.IsEmpty && IntegerWhitespace(value[^1])) value = value[..^1];
        if (value.IsEmpty) return BigInteger.Zero;
        bool negative = value[0] == '-';
        if (value[0] is '+' or '-') value = value[1..];
        if (value.IsEmpty) return BigInteger.Zero;
        BigInteger number = BigInteger.Zero;
        foreach (Rune rune in value.EnumerateRunes())
        {
            int digit = DecimalDigit(rune.Value);
            if (digit < 0) return BigInteger.Zero;
            number = number * 10 + digit;
        }
        return negative ? -number : number;
    }

    private static bool IntegerWhitespace(char value) => value is >= '\u0009' and <= '\u000d'
        or '\u0020' or '\u0085' or '\u00a0' or '\u1680' or >= '\u2000' and <= '\u200a'
        or '\u2028' or '\u2029' or '\u202f' or '\u205f' or '\u3000';

    // Unicode 15 decimal blocks, matching the existing pinned CPython 3.12 text profile.
    private static int DecimalDigit(int codePoint)
    {
        ReadOnlySpan<int> zeroes = [0x30,0x660,0x6f0,0x7c0,0x966,0x9e6,0xa66,0xae6,0xb66,0xbe6,0xc66,0xce6,
            0xd66,0xde6,0xe50,0xed0,0xf20,0x1040,0x1090,0x17e0,0x1810,0x1946,0x19d0,0x1a80,0x1a90,0x1b50,
            0x1bb0,0x1c40,0x1c50,0xa620,0xa8d0,0xa900,0xa9d0,0xa9f0,0xaa50,0xabf0,0xff10,0x104a0,0x10d30,
            0x11066,0x110f0,0x11136,0x111d0,0x112f0,0x11450,0x114d0,0x11650,0x116c0,0x11730,0x118e0,
            0x11950,0x11c50,0x11d50,0x11da0,0x11f50,0x16a60,0x16ac0,0x16b50,0x1d7ce,0x1d7d8,0x1d7e2,
            0x1d7ec,0x1d7f6,0x1e140,0x1e2f0,0x1e4f0,0x1e950,0x1fbf0];
        foreach (int zero in zeroes)
        {
            if (codePoint < zero) break;
            if (codePoint <= zero + 9) return codePoint - zero;
        }
        return -1;
    }
}
