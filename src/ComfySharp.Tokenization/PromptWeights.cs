using System.Globalization;
using System.Text;

namespace ComfySharp.Tokenization;

/// <summary>ComfyUI sd1_clip.py prompt parsing, including its unbalanced-parenthesis and escape-sentinel behavior.</summary>
public static class PromptWeights
{
    public static IReadOnlyList<WeightedPromptSegment> Parse(string text, bool disableWeights = false,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        cancellationToken.ThrowIfCancellationRequested();
        var escaped = Escape(text);
        if (disableWeights)
            return Array.AsReadOnly(new[] { new WeightedPromptSegment(Unescape(escaped), 1.0) });

        var output = new List<WeightedPromptSegment>();
        var work = new Stack<WeightedPromptSegment>();
        PushSegments(work, escaped, 1.0, cancellationToken);
        while (work.TryPop(out var segment))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var value = segment.Text;
            if (value.Length >= 2 && value[0] == '(' && value[^1] == ')')
            {
                value = value[1..^1];
                var weight = segment.Weight * 1.1;
                var colon = value.LastIndexOf(':');
                if (colon > 0 && TryPythonFloat(value.AsSpan(colon + 1), cancellationToken, out var explicitWeight))
                {
                    weight = explicitWeight;
                    value = value[..colon];
                }
                PushSegments(work, value, weight, cancellationToken);
            }
            else
                output.Add(new WeightedPromptSegment(Unescape(value), segment.Weight));
        }
        return output.AsReadOnly();
    }

    private static void PushSegments(Stack<WeightedPromptSegment> work, string text, double weight, CancellationToken cancellationToken)
    {
        var segments = new List<string>();
        var start = 0;
        var nesting = 0;
        for (var i = 0; i < text.Length; i++)
        {
            if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            if (text[i] == '(')
            {
                if (nesting == 0 && i > start)
                {
                    segments.Add(text[start..i]);
                    start = i;
                }
                nesting++;
            }
            else if (text[i] == ')')
            {
                nesting--;
                if (nesting == 0)
                {
                    segments.Add(text[start..(i + 1)]);
                    start = i + 1;
                }
            }
        }
        if (start < text.Length) segments.Add(text[start..]);
        for (var i = segments.Count - 1; i >= 0; i--)
            work.Push(new WeightedPromptSegment(segments[i], weight));
    }

    private static string Escape(string text) => text.Replace("\\)", "\0\u0001", StringComparison.Ordinal)
        .Replace("\\(", "\0\u0002", StringComparison.Ordinal);
    private static string Unescape(string text) => text.Replace("\0\u0001", ")", StringComparison.Ordinal)
        .Replace("\0\u0002", "(", StringComparison.Ordinal);

    // Python 3.12 / Unicode 15.0.0, independently pinned from tokenizers' Unicode data.
    // Generated from the reference lab's unicodedata.category/decimal and str.isspace.
    private static ReadOnlySpan<int> PythonDecimalZeroes =>
    [
        0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6,
        0xd66, 0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0,
        0x1a80, 0x1a90, 0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0,
        0xa9f0, 0xaa50, 0xabf0, 0xff10, 0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0,
        0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50,
        0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2,
        0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0,
    ];

    internal static bool IsPythonWhitespace(char value) => IsFloatWhitespace(value) || value is >= '\u001c' and <= '\u001f';
    private static bool IsFloatWhitespace(char value) => value is >= '\u0009' and <= '\u000d'
        or '\u0020' or '\u0085' or '\u00a0' or '\u1680' or >= '\u2000' and <= '\u200a'
        or '\u2028' or '\u2029' or '\u202f' or '\u205f' or '\u3000';

    private static int PythonDecimalDigit(int codePoint)
    {
        foreach (var zero in PythonDecimalZeroes)
        {
            if (codePoint < zero) break;
            if (codePoint <= zero + 9) return codePoint - zero;
        }
        return -1;
    }

    private static bool TryPythonFloat(ReadOnlySpan<char> value, CancellationToken cancellationToken, out double result)
    {
        result = default;
        // Python float accepts Unicode decimal digits, but not the extra U+001C..U+001F
        // separators accepted by str.isspace()/re \s. Trim only float's whitespace.
        while (!value.IsEmpty && IsFloatWhitespace(value[0])) value = value[1..];
        while (!value.IsEmpty && IsFloatWhitespace(value[^1])) value = value[..^1];
        if (value.IsEmpty) return false;
        var normalized = new StringBuilder(value.Length);
        foreach (var rune in value.EnumerateRunes())
        {
            if ((normalized.Length & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            var digit = PythonDecimalDigit(rune.Value);
            if (digit >= 0)
                normalized.Append((char)('0' + digit));
            else if (rune.IsAscii) normalized.Append((char)rune.Value);
            else return false;
        }
        var number = normalized.ToString();
        var unsigned = number.AsSpan();
        var negative = unsigned[0] == '-';
        if (unsigned[0] is '+' or '-') unsigned = unsigned[1..];
        if (unsigned.Equals("nan", StringComparison.OrdinalIgnoreCase))
        {
            result = BitConverter.Int64BitsToDouble(negative ? unchecked((long)0xfff8000000000000UL) : 0x7ff8000000000000L);
            return true;
        }
        if (unsigned.Equals("inf", StringComparison.OrdinalIgnoreCase) || unsigned.Equals("infinity", StringComparison.OrdinalIgnoreCase))
        {
            result = negative ? double.NegativeInfinity : double.PositiveInfinity;
            return true;
        }
        for (var i = 0; i < number.Length; i++)
            if (number[i] == '_' && (i == 0 || i + 1 == number.Length || !IsDigit(number[i - 1]) || !IsDigit(number[i + 1])))
                return false;
        number = number.Replace("_", "", StringComparison.Ordinal);
        var position = number.Length > 0 && number[0] is '+' or '-' ? 1 : 0;
        var digits = ConsumeDigits(number, ref position);
        if (position < number.Length && number[position] == '.')
        {
            position++;
            digits += ConsumeDigits(number, ref position);
        }
        if (digits == 0) return false;
        if (position < number.Length && number[position] is 'e' or 'E')
        {
            position++;
            if (position < number.Length && number[position] is '+' or '-') position++;
            if (ConsumeDigits(number, ref position) == 0) return false;
        }
        return position == number.Length && double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out result);
    }

    private static bool IsDigit(char value) => value is >= '0' and <= '9';
    private static int ConsumeDigits(string text, ref int position)
    {
        var start = position;
        while (position < text.Length && IsDigit(text[position])) position++;
        return position - start;
    }
}
