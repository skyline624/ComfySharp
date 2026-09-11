using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Inference;
using static TorchSharp.torch;

namespace ComfySharp.Nodes.Tensor;

/// <summary>Actual CPU sigma primitives and PreviewAny from the frozen ComfyUI backend; no model inference claim.</summary>
public static class TensorNodes
{
    private const string Module = "comfy_extras.nodes_custom_sampler";
    public static NodeRegistry CreateRegistry()
    {
        var registry = BuiltInNodes.CreateRegistry();
        Register(registry);
        return registry;
    }
    public static void Register(NodeRegistry registry)
    {
        foreach (var id in new[] { "KarrasScheduler", "ExponentialScheduler", "PolyexponentialScheduler", "LaplaceScheduler", "VPScheduler" })
        {
            var inputs = new List<InputSchema> { Int("steps", 20, 1, 10000) };
            if (id == "VPScheduler") inputs.AddRange([Float("beta_d", 19.9, 0, 5000, .01, true), Float("beta_min", .1, 0, 5000, .01, true), Float("eps_s", .001, 0, 1, .0001, true)]);
            else
            {
                inputs.AddRange([Float("sigma_max", 14.614642, 0, 5000, .01, true), Float("sigma_min", .0291675, 0, 5000, .01, true)]);
                if (id is "KarrasScheduler" or "PolyexponentialScheduler") inputs.Add(Float("rho", id == "KarrasScheduler" ? 7 : 1, 0, 100, .01, true));
                if (id == "LaplaceScheduler") inputs.AddRange([Float("mu", 0, -10, 10, .1, true), Float("beta", .5, 0, 10, .1, true)]);
            }
            registry.Register(new TensorNode(new(id, id, "model/sampling/schedulers", inputs, [new("SIGMAS")], PythonModule: Module)));
        }
        foreach (var id in new[] { "SplitSigmas", "SplitSigmasDenoise", "FlipSigmas", "SetFirstSigma", "ExtendIntermediateSigmas", "ManualSigmas" })
        {
            var inputs = new List<InputSchema>();
            if (id == "ManualSigmas") inputs.Add(new("sigmas", "STRING", Options: new() { ["default"] = "1, 0.5", ["multiline"] = false }));
            else inputs.Add(new("sigmas", "SIGMAS"));
            if (id == "SplitSigmas") inputs.Add(Int("step", 0, 0, 10000));
            if (id == "SplitSigmasDenoise") inputs.Add(Float("denoise", 1, 0, 1, .01, round: null));
            if (id == "SetFirstSigma") inputs.Add(Float("sigma", 136, 0, 20000, .001));
            if (id == "ExtendIntermediateSigmas") inputs.AddRange([Int("steps", 2, 1, 100), Float("start_at_sigma", -1, -1, 20000, .01), Float("end_at_sigma", 12, 0, 20000, .01),
                new("spacing", "COMBO", Options: new() { ["options"] = new JsonArray("linear", "cosine", "sine") })]);
            OutputSchema[] outputs = id is "SplitSigmas" or "SplitSigmasDenoise" ? [new("SIGMAS", "high_sigmas"), new("SIGMAS", "low_sigmas")] : [new("SIGMAS")];
            registry.Register(new TensorNode(new(id, id, "model/sampling/sigmas", inputs, outputs, Experimental: id == "ManualSigmas",
                SearchAliases: id == "ManualSigmas" ? ["custom noise schedule", "define sigmas"] : id == "ExtendIntermediateSigmas" ? ["interpolate sigmas"] : null, PythonModule: Module)));
        }
        registry.Register(new PreviewNode());
    }
    private static InputSchema Int(string name, int value, int min, int max) => new(name, "INT", Options: new() { ["default"] = value, ["min"] = min, ["max"] = max });
    private static InputSchema Float(string name, double value, double min, double max, double step, bool advanced = false, bool? round = false)
    {
        var options = new JsonObject { ["default"] = value, ["min"] = min, ["max"] = max, ["step"] = step };
        if (advanced) options["advanced"] = true;
        if (round.HasValue) options["round"] = round.Value;
        return new(name, "FLOAT", Options: options);
    }

    private sealed class TensorNode(NodeSchema schema) : IRuntimeNode
    {
        public NodeSchema Schema { get; } = schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            double D(string key) => PythonValues.Float(inputs[key].ToJson());
            int I(string key) => checked((int)PythonValues.Integer(inputs[key].ToJson()));
            string S(string key) => inputs[key].ToJson()!.GetValue<string>();
            RuntimeValue Own(TorchSharp.torch.Tensor tensor)
            {
                // A failed transfer must leave the tensor in this scope; detach only once context owns it.
                var value = context.Own(tensor);
                tensor.DetachFromDisposeScope();
                return value;
            }
            TorchSharp.torch.Tensor Sigmas()
            {
                if (inputs["sigmas"].Kind != RuntimeValueKind.Native)
                    throw new ArgumentException("SIGMAS requires a native rank-one tensor.");
                var value = inputs["sigmas"].GetNative<TorchSharp.torch.Tensor>();
                if (value.dim() != 1) throw new ArgumentException("SIGMAS requires a rank-one tensor.");
                return value;
            }
            using var scope = NewDisposeScope();
            IReadOnlyList<RuntimeValue> result;
            if (Schema.ClassType is "SplitSigmas" or "SplitSigmasDenoise")
            {
                var split = Schema.ClassType == "SplitSigmas" ? SigmaOperations.Split(Sigmas(), I("step"), cancellationToken)
                    : SigmaOperations.SplitDenoise(Sigmas(), D("denoise"), cancellationToken);
                result = [Own(split.High), Own(split.Low)];
            }
            else if (Schema.ClassType == "FlipSigmas" && Sigmas().shape[0] == 0)
                result = [context.Retain(inputs["sigmas"])];
            else
            {
                var tensor = Schema.ClassType switch
                {
                    "KarrasScheduler" => SigmaSchedules.Karras(I("steps"), D("sigma_min"), D("sigma_max"), D("rho"), cancellationToken),
                    "ExponentialScheduler" => SigmaSchedules.Exponential(I("steps"), D("sigma_min"), D("sigma_max"), cancellationToken),
                    "PolyexponentialScheduler" => SigmaSchedules.Polyexponential(I("steps"), D("sigma_min"), D("sigma_max"), D("rho"), cancellationToken),
                    "LaplaceScheduler" => SigmaSchedules.Laplace(I("steps"), D("sigma_min"), D("sigma_max"), D("mu"), D("beta"), cancellationToken),
                    "VPScheduler" => SigmaSchedules.VP(I("steps"), D("beta_d"), D("beta_min"), D("eps_s"), cancellationToken),
                    "FlipSigmas" => SigmaOperations.Flip(Sigmas(), cancellationToken),
                    "SetFirstSigma" => SigmaOperations.SetFirst(Sigmas(), D("sigma"), cancellationToken),
                    "ExtendIntermediateSigmas" => SigmaOperations.Extend(Sigmas(), I("steps"), D("start_at_sigma"), D("end_at_sigma"), S("spacing"), cancellationToken),
                    "ManualSigmas" => SigmaOperations.Manual(S("sigmas"), cancellationToken),
                    _ => throw new InvalidOperationException("Unregistered sigma operation.")
                };
                cancellationToken.ThrowIfCancellationRequested();
                result = [Own(tensor)];
            }
            return ValueTask.FromResult(new NodeExecutionOutput(result));
        }
    }

    private sealed class PreviewNode : IRuntimeNode
    {
        public NodeSchema Schema { get; } = new("PreviewAny", "Preview as Text", "utilities", [new("source", "*")], [new("STRING")], OutputNode: true,
            Description: "Preview any input value as text.", SearchAliases: ["preview", "preview text", "show output", "inspect", "debug", "print value", "show text"], PythonModule: "comfy_extras.nodes_preview_any");
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context, IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string value = PreviewValue(inputs["source"], cancellationToken);
            return ValueTask.FromResult(new NodeExecutionOutput([context.Json(JsonValue.Create(value))], new JsonObject { ["text"] = new JsonArray(value) }));
        }
    }

    private static string PreviewValue(RuntimeValue value, CancellationToken cancellationToken)
    {
        if (TryJson(value, out var json)) return json is JsonArray or JsonObject ? PrettyJson(json, 0) : Scalar(json);
        return Repr(value, cancellationToken);
    }
    private static bool TryJson(RuntimeValue value, out JsonNode? json)
    {
        try { json = value.ToJson(); return true; }
        catch (RuntimeValueProjectionException) { json = null; return false; }
    }
    private static string Repr(RuntimeValue value, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return value.Kind switch
        {
            RuntimeValueKind.Json => JsonRepr(value.ToJson()),
            RuntimeValueKind.Native => TensorPreviewFormatter.Format(value.GetNative<TorchSharp.torch.Tensor>(), cancellationToken),
            RuntimeValueKind.List => "[" + string.Join(", ", value.Items.Select(item => ItemRepr(item, cancellationToken))) + "]",
            RuntimeValueKind.Map => "{" + string.Join(", ", value.Properties.Select(p => StringRepr(p.Key) + ": " + ItemRepr(p.Value, cancellationToken))) + "}",
            _ => throw new NotSupportedException("PreviewAny runtime value is not supported.")
        };
    }
    private static string ItemRepr(RuntimeValue value, CancellationToken cancellationToken) => value.Kind == RuntimeValueKind.Json && value.ToJson() is JsonValue v && v.TryGetValue<string>(out var text)
        ? StringRepr(text) : Repr(value, cancellationToken);
    private static string StringRepr(string text)
    {
        char quote = text.Contains('\'') && !text.Contains('"') ? '"' : '\'';
        var result = new StringBuilder().Append(quote);
        foreach (var rune in text.EnumerateRunes())
        {
            if (rune.Value == quote || rune.Value == '\\') result.Append('\\').Append(rune.ToString());
            else if (rune.Value is '\n' or '\r' or '\t') result.Append(rune.Value switch { '\n' => "\\n", '\r' => "\\r", _ => "\\t" });
            else if (rune.Value != ' ' && Rune.GetUnicodeCategory(rune) is UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.Surrogate or UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned or UnicodeCategory.SpaceSeparator or UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator)
                result.Append(rune.Value <= 255 ? "\\x" + rune.Value.ToString("x2", CultureInfo.InvariantCulture) : rune.Value <= 65535 ? "\\u" + rune.Value.ToString("x4", CultureInfo.InvariantCulture) : "\\U" + rune.Value.ToString("x8", CultureInfo.InvariantCulture));
            else result.Append(rune.ToString());
        }
        return result.Append(quote).ToString();
    }
    private static string JsonRepr(JsonNode? node) => node switch
    {
        JsonValue v when v.TryGetValue<string>(out var s) => StringRepr(s),
        JsonArray a => "[" + string.Join(", ", a.Select(JsonRepr)) + "]",
        JsonObject o => "{" + string.Join(", ", o.Select(p => StringRepr(p.Key) + ": " + JsonRepr(p.Value))) + "}",
        _ => Scalar(node)
    };
    private static string Scalar(JsonNode? json)
    {
        if (json is null) return "None";
        if (json is JsonValue v)
        {
            if (v.TryGetValue<string>(out var text)) return text;
            if (v.TryGetValue<bool>(out var boolean)) return boolean ? "True" : "False";
            string raw = v.ToJsonString();
            if (v.TryGetValue<JsonElement>(out var element) ? !raw.Contains('.') && !raw.Contains('e', StringComparison.OrdinalIgnoreCase)
                : !v.TryGetValue<double>(out _) && !v.TryGetValue<float>(out _) && !v.TryGetValue<decimal>(out _)) return raw;
            double number = PythonValues.Float(v);
            string shortest = number.ToString("R", CultureInfo.InvariantCulture).ToLowerInvariant();
            string sign = shortest.StartsWith('-') ? "-" : "";
            string[] parts = shortest.TrimStart('-').Split('e');
            int explicitExponent = parts.Length == 2 ? int.Parse(parts[1], CultureInfo.InvariantCulture) : 0;
            int point = parts[0].IndexOf('.');
            if (point < 0) point = parts[0].Length;
            string digits = parts[0].Replace(".", "", StringComparison.Ordinal);
            int leading = digits.TakeWhile(c => c == '0').Count();
            if (leading == digits.Length) return sign + "0.0";
            int exponent = point - leading - 1 + explicitExponent;
            digits = digits[leading..].TrimEnd('0');
            if (exponent is < -4 or >= 16)
                return sign + digits[0] + (digits.Length > 1 ? "." + digits[1..] : "") + "e" + (exponent < 0 ? "-" : "+") + Math.Abs(exponent).ToString("D2", CultureInfo.InvariantCulture);
            if (exponent < 0) return sign + "0." + new string('0', -exponent - 1) + digits;
            if (exponent + 1 >= digits.Length) return sign + digits + new string('0', exponent + 1 - digits.Length) + ".0";
            return sign + digits.Insert(exponent + 1, ".");
        }
        throw new ArgumentException("Expected a JSON scalar.");
    }
    private static string JsonString(string value)
    {
        var result = new StringBuilder().Append('"');
        foreach (char c in value)
            result.Append(c switch
            {
                '"' => "\\\"", '\\' => "\\\\", '\b' => "\\b", '\f' => "\\f", '\n' => "\\n", '\r' => "\\r", '\t' => "\\t",
                < ' ' => "\\u" + ((int)c).ToString("x4", CultureInfo.InvariantCulture), _ => c.ToString()
            });
        return result.Append('"').ToString();
    }
    private static string PrettyJson(JsonNode? json, int indent)
    {
        string pad = new(' ', indent), nested = new(' ', indent + 4);
        return json switch
        {
            null => "null",
            JsonArray a when a.Count == 0 => "[]",
            JsonObject o when o.Count == 0 => "{}",
            JsonArray a => "[\n" + string.Join(",\n", a.Select(v => nested + PrettyJson(v, indent + 4))) + "\n" + pad + "]",
            JsonObject o => "{\n" + string.Join(",\n", o.Select(p => nested + JsonString(p.Key) + ": " + PrettyJson(p.Value, indent + 4))) + "\n" + pad + "}",
            JsonValue v when v.TryGetValue<string>(out var s) => JsonString(s),
            JsonValue v when v.TryGetValue<bool>(out var b) => b ? "true" : "false",
            _ => Scalar(json)
        };
    }
}
