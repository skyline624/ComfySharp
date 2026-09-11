using System.Globalization;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>PyTorch default dense tensor text with PreviewAny's edgeitems=6, precision=4 and linewidth=80.</summary>
/// <remarks>Ports torch/_tensor_str.py's dense real-number formatter. Sparse, complex, quantized and non-leaf
/// autograd tensors require additional adapters; they produce an explicit diagnostic instead of a misleading representation.</remarks>
public static class TensorPreviewFormatter
{
    private const int Edges = 6;
    public static string Format(Tensor input, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(input);
        if (input.IsInvalid) throw new ObjectDisposedException(nameof(input));
        string dtype = DtypeName(input.dtype);
        if (input.is_sparse || (input.requires_grad && !input.is_leaf))
            throw new NotSupportedException("PreviewAny currently formats dense real tensors and leaf gradients; sparse and non-leaf autograd representations are not ported.");
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        long[] shape = input.shape;
        bool summarize = input.numel() > 1000;
        // Select only displayed edges before a possible CPU transfer. Never copy a whole large GPU tensor for text.
        var selected = input.detach();
        if (summarize)
            for (int dimension = 0; dimension < shape.Length; dimension++)
                if (shape[dimension] > 2 * Edges)
                    selected = cat(new[] { selected.slice(dimension, 0, Edges, 1), selected.slice(dimension, shape[dimension] - Edges, shape[dimension], 1) }, dimension);
        bool floating = input.dtype is ScalarType.Float16 or ScalarType.BFloat16 or ScalarType.Float32 or ScalarType.Float64;
        bool boolean = input.dtype == ScalarType.Bool;
        // Some devices, notably MPS, cannot represent Float64. Transfer only the selected display data
        // before promoting it for managed formatting; this does not expand a summarized GPU tensor.
        var flat = selected.cpu().to_type(floating ? ScalarType.Float64 : ScalarType.Int64).contiguous().flatten();
        var doubles = floating ? flat.data<double>().ToArray() : [];
        var integers = floating ? [] : flat.data<long>().ToArray();
        var formatter = new NumberFormatter(doubles, integers, floating, boolean);
        int offset = 0;
        string Render(int dimension, int indent)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (dimension == shape.Length) return formatter.Value(offset++);
            int count = checked((int)Math.Min(shape[dimension], summarize ? 2 * Edges : int.MaxValue));
            bool skipped = summarize && shape[dimension] > 2 * Edges;
            var parts = new List<string>(count + 1);
            for (int i = 0; i < count; i++)
            {
                if (skipped && i == Edges) parts.Add(dimension + 1 == shape.Length ? " ..." : "...");
                parts.Add(Render(dimension + 1, indent + 1));
            }
            if (dimension + 1 == shape.Length)
            {
                int perLine = Math.Max(1, (80 - indent) / (formatter.Width + 2));
                return "[" + string.Join(",\n" + new string(' ', indent + 1), parts.Chunk(perLine).Select(line => string.Join(", ", line))) + "]";
            }
            return "[" + string.Join("," + new string('\n', shape.Length - dimension - 1) + new string(' ', indent + 1), parts) + "]";
        }
        string contents = input.numel() == 0 ? "[]" : Render(0, 7);
        var suffixes = new List<string>();
        if (input.device.type != DeviceType.CPU) suffixes.Add("device='" + input.device + "'");
        if (input.numel() == 0 && shape.Length != 1)
            suffixes.Add("size=(" + string.Join(", ", shape) + (shape.Length == 1 ? "," : "") + ")");
        if (input.dtype != ScalarType.Float32 && (input.numel() == 0 || input.dtype is not (ScalarType.Int64 or ScalarType.Bool)))
            suffixes.Add("dtype=torch." + dtype);
        if (input.requires_grad) suffixes.Add("requires_grad=True");
        var result = new StringBuilder("tensor(" + contents);
        int lastLine = result.Length - result.ToString().LastIndexOf('\n') + 1;
        foreach (string suffix in suffixes)
        {
            if (lastLine + suffix.Length + 2 > 80) { result.Append(",\n       "); lastLine = 7 + suffix.Length; }
            else { result.Append(", "); lastLine += suffix.Length + 2; }
            result.Append(suffix);
        }
        cancellationToken.ThrowIfCancellationRequested();
        return result.Append(')').ToString();
    }

    private static string DtypeName(ScalarType dtype) => dtype switch
    {
        ScalarType.Float32 => "float32", ScalarType.Float64 => "float64", ScalarType.Float16 => "float16", ScalarType.BFloat16 => "bfloat16",
        ScalarType.Byte => "uint8", ScalarType.Int8 => "int8", ScalarType.Int16 => "int16", ScalarType.Int32 => "int32", ScalarType.Int64 => "int64", ScalarType.Bool => "bool",
        _ => throw new NotSupportedException($"PreviewAny tensor dtype {dtype} is not yet ported.")
    };

    private sealed class NumberFormatter
    {
        private readonly double[] numbers;
        private readonly long[] integers;
        private readonly bool floating, boolean, integerMode = true, scientific;
        public int Width { get; private set; } = 1;
        public NumberFormatter(double[] numbers, long[] integers, bool floating, bool boolean)
        {
            this.numbers = numbers; this.integers = integers; this.floating = floating; this.boolean = boolean;
            if (!floating)
            {
                foreach (var value in integers) Width = Math.Max(Width, Integer(value).Length);
                return;
            }
            var finite = numbers.Where(v => double.IsFinite(v) && v != 0).ToArray();
            if (finite.Length == 0) return;
            integerMode = finite.All(v => v == Math.Ceiling(v));
            double min = finite.Min(v => Math.Abs(v)), max = finite.Max(v => Math.Abs(v));
            scientific = max / min > 1000 || max > 1e8 || min < 1e-4;
            foreach (double value in finite) Width = Math.Max(Width, Float(value).Length);
        }
        public string Value(int index) => (floating ? Float(numbers[index]) : Integer(integers[index])).PadLeft(Width);
        private string Integer(long value) => boolean ? (value == 0 ? "False" : "True") : value.ToString(CultureInfo.InvariantCulture);
        private string Float(double value)
        {
            if (double.IsNaN(value)) return "nan";
            if (double.IsPositiveInfinity(value)) return "inf";
            if (double.IsNegativeInfinity(value)) return "-inf";
            return scientific ? Scientific(value)
                : integerMode ? value.ToString("F0", CultureInfo.InvariantCulture) + "."
                : value.ToString("F4", CultureInfo.InvariantCulture);
        }
        private static string Scientific(double value)
        {
            // Standard formatting preserves binary-double rounding; custom decimal patterns can round
            // differently. Python pads exponents to two digits, whereas .NET's standard e format uses three.
            string[] parts = value.ToString("e4", CultureInfo.InvariantCulture).Split('e');
            int exponent = int.Parse(parts[1], CultureInfo.InvariantCulture);
            return parts[0] + "e" + (exponent < 0 ? "-" : "+") + Math.Abs(exponent).ToString("D2", CultureInfo.InvariantCulture);
        }
    }
}
