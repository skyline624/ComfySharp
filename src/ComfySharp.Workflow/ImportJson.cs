using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace ComfySharp.Workflow;

/// <summary>Import-only compatibility for Python JSON. Ordinary document and API serialization stays strict.</summary>
public static class ImportJson
{
    public const string NonFiniteWarning = "JSON contained non-finite numeric tokens (NaN/Infinity); they were replaced with null.";

    public static JsonNode? Parse(string text, Action<string>? reportWarning = null)
    {
        try { return JsonNode.Parse(text); }
        catch (JsonException)
        {
            var result = new StringBuilder(text.Length); bool changed = false;
            for (int i = 0; i < text.Length;)
            {
                if (text[i] == '"')
                {
                    // Skip quoted strings, including escaped quotes and backslashes, in one pass.
                    int start = i++;
                    while (i < text.Length)
                    {
                        char current = text[i++];
                        if (current == '\\' && i < text.Length) i++;
                        else if (current == '"') break;
                    }
                    result.Append(text, start, i - start); continue;
                }
                string? token = text.AsSpan(i).StartsWith("-Infinity", StringComparison.Ordinal) ? "-Infinity" :
                    text.AsSpan(i).StartsWith("Infinity", StringComparison.Ordinal) ? "Infinity" :
                    text.AsSpan(i).StartsWith("NaN", StringComparison.Ordinal) ? "NaN" : null;
                if (token is not null && (i == 0 || !LeftBoundaryBlocked(text[i - 1])) &&
                    (i + token.Length == text.Length || !RightBoundaryBlocked(text[i + token.Length])))
                {
                    result.Append("null"); i += token.Length; changed = true;
                }
                else result.Append(text[i++]);
            }
            // Warn once per fallback, including when unrelated malformed JSON subsequently fails.
            if (changed) reportWarning?.Invoke(NonFiniteWarning);
            return JsonNode.Parse(result.ToString());
        }
    }

    // JavaScript's non-Unicode regexp \w is ASCII. Unrelated invalid JSON remains invalid.
    private static bool RightBoundaryBlocked(char c) => char.IsAsciiLetterOrDigit(c) || c is '_' or '.';
    private static bool LeftBoundaryBlocked(char c) => RightBoundaryBlocked(c) || c == '-';
}
