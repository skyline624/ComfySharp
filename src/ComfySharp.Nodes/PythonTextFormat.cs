using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;

namespace ComfySharp.Nodes;

/// <summary>The partial python-format-text-v1 profile, not a general Python formatter.</summary>
internal static class PythonTextFormat
{
    internal sealed record Limits(int FormatCodePoints = 65536, int WidthOrPrecision = 1048576,
        int OutputCodePoints = 1048576);

    internal sealed class Error(string category, int position, string detail)
        : FormatException($"{category} at UTF-16 position {position}: {detail}")
    {
        public string Category { get; } = category;
        public int Position { get; } = position;
    }

    private readonly record struct Field(string Name, char? Conversion, string Spec, int End);
    private readonly record struct TextSpec(string Fill, char Align, int Width, int? Precision);

    public static string Format(string format, IReadOnlyDictionary<string, RuntimeValue> values,
        CancellationToken cancellationToken = default, Limits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(format);
        ArgumentNullException.ThrowIfNull(values);
        limits ??= new();
        if (limits.FormatCodePoints < 0 || limits.WidthOrPrecision < 0 || limits.OutputCodePoints < 0)
            throw new ArgumentOutOfRangeException(nameof(limits));
        cancellationToken.ThrowIfCancellationRequested();
        Count(format, limits.FormatCodePoints, cancellationToken, 0);
        var output = new StringBuilder();
        int outputPoints = 0;
        for (int i = 0; i < format.Length;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int position = i;
            if (format[i] is '{' or '}')
            {
                char brace = format[i];
                if (i + 1 < format.Length && format[i + 1] == brace)
                {
                    Reserve(1, 1, position); output.Append(brace); i += 2; continue;
                }
                if (brace == '}') throw Fail("format_syntax", position, "Single closing brace.");
                var field = ReadField(format, i);
                i = field.End;
                // Finish this field's structural parse before lookup, but never scan later fields first.
                var value = Lookup(values, field.Name, position);
                if (field.Conversion is 'r' or 'a')
                    throw Fail("unsupported_format_feature", position, "Conversion is outside the text profile.");
                if (field.Conversion is not null and not 's')
                    throw Fail("format_syntax", position, "Unknown conversion.");
                if (field.Spec.Contains('{') || field.Spec.Contains('}'))
                    throw Fail("unsupported_format_feature", position, "Nested specifications are not supported.");
                var (text, isString) = Scalar(value, position);
                if (!isString && field.Conversion != 's' && field.Spec.Length != 0)
                    throw Fail("unsupported_format_feature", position, "Scalar specifications require explicit !s.");
                var spec = ReadSpec(field.Spec, limits, position);
                int length = 0, end = 0;
                // Validate the entire used string, including text excluded by precision.
                for (int t = 0; t < text.Length;)
                {
                    if ((length & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
                    int consumed = ReadRune(text, t, position).Utf16SequenceLength;
                    t += consumed;
                    if (spec.Precision is null || length < spec.Precision) end = t;
                    length++;
                }
                int kept = Math.Min(length, spec.Precision ?? int.MaxValue);
                int padding = Math.Max(0, spec.Width - kept);
                int left = spec.Align == '>' ? padding : spec.Align == '^' ? padding / 2 : 0;
                long utf16 = (long)end + (long)padding * spec.Fill.Length;
                Reserve((long)kept + padding, utf16, position);
                Pad(left, spec.Fill);
                output.Append(text.AsSpan(0, end));
                Pad(padding - left, spec.Fill);
            }
            else
            {
                int length = ReadRune(format, i, position).Utf16SequenceLength;
                Reserve(1, length, position); output.Append(format.AsSpan(i, length)); i += length;
            }
        }
        cancellationToken.ThrowIfCancellationRequested();
        return output.ToString();

        void Reserve(long points, long units, int position)
        {
            if (points > limits.OutputCodePoints - (long)outputPoints || units > int.MaxValue - (long)output.Length)
                throw Fail("format_limit", position, "Output exceeds the configured budget.");
            outputPoints = checked(outputPoints + (int)points);
        }
        void Pad(int count, string fill)
        {
            for (int p = 0; p < count; p++)
            {
                if ((p & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
                output.Append(fill);
            }
        }
    }

    private static Field ReadField(string format, int start)
    {
        int i = start + 1, nameStart = i;
        while (i < format.Length && format[i] is not '!' and not ':' and not '}')
        {
            if (format[i] == '{') throw Fail("format_syntax", start, "Opening brace in field name.");
            i++;
        }
        if (i == format.Length) throw Fail("format_syntax", start, "Unclosed field.");
        string name = format[nameStart..i];
        char? conversion = null;
        if (format[i] == '!')
        {
            i++;
            if (i == format.Length) throw Fail("format_syntax", start, "Missing conversion.");
            conversion = format[i++];
            if (i == format.Length || format[i] is not ':' and not '}')
                throw Fail("format_syntax", start, "Conversion must contain one character.");
        }
        if (format[i] == '}') return new(name, conversion, "", i + 1);
        int specStart = ++i, depth = 1;
        while (i < format.Length)
        {
            if (format[i] == '{') depth++;
            else if (format[i] == '}' && --depth == 0)
                return new(name, conversion, format[specStart..i], i + 1);
            i++;
        }
        throw Fail("format_syntax", start, "Unclosed field specification.");
    }

    private static RuntimeValue Lookup(IReadOnlyDictionary<string, RuntimeValue> values, string name, int position)
    {
        if (name.Length == 0 || name.All(char.IsAsciiDigit))
            throw Fail("format_missing_field", position, "No positional arguments are supplied.");
        int suffix = name.IndexOfAny(['.', '[', ']']);
        string root = suffix < 0 ? name : name[..suffix];
        if (root.Length == 0 || root.All(char.IsAsciiDigit))
            throw Fail("format_missing_field", position, "No positional arguments are supplied.");
        // Ordinal lookup even when a caller supplies a dictionary with a different comparer.
        RuntimeValue? value = null;
        foreach (var entry in values)
            if (string.Equals(entry.Key, root, StringComparison.Ordinal)) { value = entry.Value; break; }
        if (value is null) throw Fail("format_missing_field", position, "Named argument is absent.");
        if (suffix >= 0 || !root.All(c => char.IsAsciiLetterOrDigit(c) || c == '_'))
            throw Fail("unsupported_format_feature", position, "Only simple named fields are supported.");
        return value;
    }

    private static (string Text, bool IsString) Scalar(RuntimeValue value, int position)
    {
        if (value.Kind != RuntimeValueKind.Json)
            throw Fail("unsupported_format_value", position, "Only JSON text, null, Boolean and Int64 scalars are supported.");
        var node = value.ToJson();
        if (node is null) return ("None", false);
        if (node is not JsonValue json)
            throw Fail("unsupported_format_value", position, "Containers are outside the text profile.");
        if (json.TryGetValue<string>(out var text)) return (text, true);
        if (json.TryGetValue<bool>(out var flag)) return (flag ? "True" : "False", false);
        if (Integer(json, out var integer)) return (integer.ToString(CultureInfo.InvariantCulture), false);
        throw Fail("unsupported_format_value", position, "Numeric values must be exact Int64 integers, not floating point.");
    }

    private static bool Integer(JsonValue value, out long result)
    {
        // Parsed JSON retains its number lexeme: 1.0/1e0 are not integer tokens.
        if (value.TryGetValue<JsonElement>(out var element))
        {
            result = 0;
            return element.ValueKind == JsonValueKind.Number && element.TryGetInt64(out result);
        }
        if (value.TryGetValue<sbyte>(out var i8)) { result = i8; return true; }
        if (value.TryGetValue<byte>(out var u8)) { result = u8; return true; }
        if (value.TryGetValue<short>(out var i16)) { result = i16; return true; }
        if (value.TryGetValue<ushort>(out var u16)) { result = u16; return true; }
        if (value.TryGetValue<int>(out var i32)) { result = i32; return true; }
        if (value.TryGetValue<uint>(out var u32)) { result = u32; return true; }
        if (value.TryGetValue<long>(out result)) return true;
        if (value.TryGetValue<ulong>(out var u64) && u64 <= long.MaxValue) { result = (long)u64; return true; }
        result = 0; return false;
    }

    private static TextSpec ReadSpec(string spec, Limits limits, int position)
    {
        int i = 0;
        string fill = " "; char align = '<';
        if (spec.Length != 0)
        {
            var first = ReadRune(spec, 0, position);
            int second = first.Utf16SequenceLength;
            if (second < spec.Length && spec[second] is '<' or '>' or '^')
            {
                fill = first.ToString(); align = spec[second]; i = second + 1;
            }
            else if (spec[0] is '<' or '>' or '^') { align = spec[0]; i = 1; }
        }
        int widthStart = i;
        int width = Digits(spec, ref i, limits, position);
        if (i - widthStart > 1 && spec[widthStart] == '0')
            throw Fail("unsupported_format_feature", position, "Leading-zero widths are outside the profile; explicit 0> is supported.");
        int? precision = null;
        if (i < spec.Length && spec[i] == '.')
        {
            i++; int begin = i;
            precision = Digits(spec, ref i, limits, position);
            if (i == begin) throw Fail("format_syntax", position, "Precision requires digits.");
        }
        if (i < spec.Length && spec[i] == 's') i++;
        if (i != spec.Length)
            throw Fail("unsupported_format_feature", position, "Only text fill, alignment, width, precision and s are supported.");
        return new(fill, align, width, precision);
    }

    private static int Digits(string spec, ref int i, Limits limits, int position)
    {
        int result = 0;
        while (i < spec.Length && char.IsAsciiDigit(spec[i]))
        {
            int digit = spec[i++] - '0';
            if ((long)result * 10 + digit > limits.WidthOrPrecision)
                throw Fail("format_limit", position, "Width or precision exceeds the configured budget.");
            result = result * 10 + digit;
        }
        return result;
    }

    private static Rune ReadRune(string text, int position, int errorPosition)
    {
        if (Rune.DecodeFromUtf16(text.AsSpan(position), out var rune, out _) != OperationStatus.Done)
            throw Fail("unsupported_format_value", errorPosition, "Invalid UTF-16 is outside the Unicode scalar profile.");
        return rune;
    }

    private static void Count(string text, int limit, CancellationToken token, int position)
    {
        int count = 0;
        for (int i = 0; i < text.Length;)
        {
            if ((count & 1023) == 0) token.ThrowIfCancellationRequested();
            if (count++ >= limit) throw Fail("format_limit", position, "Format exceeds the configured budget.");
            i += ReadRune(text, i, position).Utf16SequenceLength;
        }
    }

    private static Error Fail(string category, int position, string detail) => new(category, position, detail);
}
