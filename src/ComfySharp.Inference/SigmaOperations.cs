using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Ports of the six model-independent SIGMAS operations. Returned tensors belong to the caller.</summary>
/// <remarks>Source: ComfyUI 1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a, comfy_extras/nodes_custom_sampler.py.</remarks>
public static partial class SigmaOperations
{
    public static (Tensor High, Tensor Low) Split(Tensor sigmas, int step, CancellationToken cancellationToken = default)
    {
        Validate(sigmas, cancellationToken);
        if (step < 0) throw new ArgumentOutOfRangeException(nameof(step));
        using var scope = NewDisposeScope();
        long length = sigmas.shape[0];
        var high = sigmas.slice(0, 0, Math.Min((long)step + 1, length), 1);
        var low = sigmas.slice(0, Math.Min(step, length), length, 1);
        cancellationToken.ThrowIfCancellationRequested();
        return (high.MoveToOuterDisposeScope(), low.MoveToOuterDisposeScope());
    }

    public static (Tensor High, Tensor Low) SplitDenoise(Tensor sigmas, double denoise, CancellationToken cancellationToken = default)
    {
        Validate(sigmas, cancellationToken);
        if (!double.IsFinite(denoise) || denoise is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(denoise));
        using var scope = NewDisposeScope();
        long length = sigmas.shape[0];
        long total = checked((long)Math.Round(Math.Max(length - 1, 0) * denoise, MidpointRounding.ToEven));
        // Python [:-(0)] is [:0], not the whole vector.
        var high = sigmas.slice(0, 0, total == 0 ? 0 : Math.Max(0, length - total), 1);
        var low = sigmas.slice(0, Math.Max(0, length - total - 1), length, 1);
        cancellationToken.ThrowIfCancellationRequested();
        return (high.MoveToOuterDisposeScope(), low.MoveToOuterDisposeScope());
    }

    public static Tensor Flip(Tensor sigmas, CancellationToken cancellationToken = default)
    {
        Validate(sigmas, cancellationToken);
        using var scope = NewDisposeScope();
        // A separate wrapper shares empty storage; the node adapter retains the original lease instead.
        var result = sigmas.shape[0] == 0 ? sigmas.alias() : sigmas.flip(0);
        if (result.shape[0] != 0)
        {
            using var first = result[0];
            using var number = first.to_type(ScalarType.Float64);
            if (number.item<double>() == 0) first.fill_(0.0001);
        }
        cancellationToken.ThrowIfCancellationRequested();
        return result.MoveToOuterDisposeScope();
    }

    public static Tensor SetFirst(Tensor sigmas, double sigma, CancellationToken cancellationToken = default)
    {
        Validate(sigmas, cancellationToken);
        if (sigmas.shape[0] == 0) throw new ArgumentException("Cannot set index 0 of empty SIGMAS.", nameof(sigmas));
        using var scope = NewDisposeScope();
        var result = sigmas.clone();
        using (var first = result[0]) first.fill_(sigma);
        cancellationToken.ThrowIfCancellationRequested();
        return result.MoveToOuterDisposeScope();
    }

    public static Tensor Extend(Tensor sigmas, int steps, double startAtSigma, double endAtSigma, string spacing,
        CancellationToken cancellationToken = default)
    {
        Validate(sigmas, cancellationToken);
        if (steps is < 1 or > 100) throw new ArgumentOutOfRangeException(nameof(steps));
        if (spacing is not ("linear" or "cosine" or "sine")) throw new ArgumentException("Unknown SIGMAS spacing.", nameof(spacing));
        using var scope = NewDisposeScope();
        if (startAtSigma < 0) startAtSigma = double.PositiveInfinity;
        var x = linspace(0, 1, steps + 1, dtype: ScalarType.Float32, device: sigmas.device).slice(0, 1, steps, 1);
        var computed = spacing switch { "cosine" => (x * Math.PI / 2).sin(), "sine" => 1 - (x * Math.PI / 2).cos(), _ => x };
        var values = new List<float>();
        for (long i = 0; i + 1 < sigmas.shape[0]; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var iteration = NewDisposeScope();
            var current = sigmas[i];
            var next = sigmas[i + 1];
            double number = current.to_type(ScalarType.Float64).item<double>();
            values.Add((float)number);
            if (endAtSigma <= number && number <= startAtSigma)
            {
                var interpolated = computed * (next - current) + current;
                values.AddRange(interpolated.to_type(ScalarType.Float32).cpu().data<float>().ToArray());
            }
        }
        if (sigmas.shape[0] > 0)
        {
            using var last = sigmas[sigmas.shape[0] - 1];
            using var converted = last.to_type(ScalarType.Float64);
            values.Add((float)converted.item<double>());
        }
        cancellationToken.ThrowIfCancellationRequested();
        var result = tensor(values.ToArray(), dtype: ScalarType.Float32, device: CPU);
        cancellationToken.ThrowIfCancellationRequested();
        return result.MoveToOuterDisposeScope();
    }

    public static Tensor Manual(string text, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(text);
        // Normalize before matching because .NET Regex operates on UTF-16 code units, whereas Python
        // \d includes supplementary-plane Unicode decimal digits (for example mathematical bold digits).
        var ascii = new StringBuilder(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            cancellationToken.ThrowIfCancellationRequested();
            ascii.Append(Rune.GetUnicodeCategory(rune) == UnicodeCategory.DecimalDigitNumber
                ? ((int)Rune.GetNumericValue(rune)).ToString(CultureInfo.InvariantCulture) : rune.ToString());
        }
        var values = new List<float>();
        foreach (Match match in SigmaPattern().Matches(ascii.ToString()))
        {
            cancellationToken.ThrowIfCancellationRequested();
            values.Add((float)double.Parse(match.Value, NumberStyles.Float, CultureInfo.InvariantCulture));
        }
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        var result = tensor(values.ToArray(), dtype: ScalarType.Float32, device: CPU);
        cancellationToken.ThrowIfCancellationRequested();
        return result.MoveToOuterDisposeScope();
    }

    private static void Validate(Tensor sigmas, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(sigmas);
        if (sigmas.IsInvalid) throw new ObjectDisposedException(nameof(sigmas));
        if (sigmas.dim() != 1) throw new ArgumentException("SIGMAS requires a rank-one tensor, not an execution list or matrix.", nameof(sigmas));
    }

    [GeneratedRegex(@"[-+]?(?:\d*\.*\d+)", RegexOptions.CultureInvariant | RegexOptions.NonBacktracking)]
    private static partial Regex SigmaPattern();
}
