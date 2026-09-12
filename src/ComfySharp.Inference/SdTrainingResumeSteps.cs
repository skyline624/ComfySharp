using System.Globalization;
using System.Numerics;
using System.Text;

namespace ComfySharp.Inference;

/// <summary>The frozen _load_existing_lora filename counter. Resolving/loading the selected file belongs to the caller.</summary>
public static class SdTrainingResumeSteps
{
    public static BigInteger Parse(string selectedFile)
    {
        ArgumentNullException.ThrowIfNull(selectedFile);
        if (selectedFile == "[None]") return BigInteger.Zero;
        int marker = selectedFile.IndexOf("_steps_",StringComparison.Ordinal);
        string prefix = marker < 0 ? selectedFile : selectedFile[..marker];
        string value = prefix[(prefix.LastIndexOf('_')+1)..].Trim();
        var normalized = new StringBuilder(); int digits = 0;
        if (value.StartsWith('+') || value.StartsWith('-')) { normalized.Append(value[0]); value = value[1..]; }
        foreach (var rune in value.EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(rune) != UnicodeCategory.DecimalDigitNumber) throw new FormatException("Existing adapter filename has no valid source step counter.");
            normalized.Append((char)('0'+(int)Rune.GetNumericValue(rune))); digits++;
        }
        // Frozen laboratory uses Python's default maximum decimal conversion length.
        if (digits == 0 || digits > 4300) throw new FormatException("Existing adapter step counter has an invalid decimal length.");
        return BigInteger.Parse(normalized.ToString(),NumberStyles.AllowLeadingSign,CultureInfo.InvariantCulture);
    }
}
