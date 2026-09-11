using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using ComfySharp.Inference;

var jsonOptions = new JsonSerializerOptions
{
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
};
const string usage = "ComfySharp.ModelInspect <explicit-file-path> [--sha256] [--tensors <1..1000>]";
if (args is ["--help"] or ["-h"])
{
    Console.WriteLine(usage);
    Console.WriteLine("Validates safetensors headers and byte ranges without initializing libtorch or loading weights.");
    Console.WriteLine("By default emits aggregate counts only; --tensors explicitly includes bounded tensor names and shapes.");
    Console.WriteLine("--sha256 explicitly reads the entire file for hashing. Metadata validation does not establish model compatibility.");
    return 0;
}

string? path = null;
bool hashRequested = false;
int tensorLimit = 0;
try
{
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--sha256":
                if (hashRequested) throw new ArgumentException("Repeated --sha256 argument.");
                hashRequested = true;
                break;
            case "--tensors":
                if (tensorLimit != 0 || i + 1 >= args.Length ||
                    !int.TryParse(args[++i], NumberStyles.None, CultureInfo.InvariantCulture, out int limit) || limit is < 1 or > 1000)
                    throw new ArgumentException("--tensors requires one limit between 1 and 1000.");
                tensorLimit = limit;
                break;
            default:
                if (args[i].StartsWith('-') || path is not null) throw new ArgumentException("Invalid or repeated argument.");
                path = args[i];
                break;
        }
    }
    if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("One explicit file path is required.");
    path = Path.GetFullPath(path);
    if (Directory.Exists(path)) throw new ArgumentException("A file path is required; directories are not scanned.");

    using var cancellation = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
    // No tensor is materialized. The per-tensor byte ceiling can cover any representable file range;
    // parsing remains bounded by the standard 16 MiB header, 100,000 tensors and rank limits.
    using var file = new SafeTensorFile(path, new(MaxTensorBytes: long.MaxValue));
    cancellation.Token.ThrowIfCancellationRequested();
    var dtypeCounts = file.Tensors.Values.GroupBy(t => t.DType).OrderBy(g => g.Key, StringComparer.Ordinal)
        .ToDictionary(g => g.Key, g => g.Count());
    var rankCounts = file.Tensors.Values.GroupBy(t => t.Shape.Count).OrderBy(g => g.Key)
        .ToDictionary(g => g.Key, g => g.Count());
    long sizeBytes = file.FileSizeBytes;
    long tensorBytes = file.Tensors.Values.Sum(t => t.End - t.Start);
    string? sha256 = hashRequested ? file.ComputeSha256(cancellation.Token) : null;
    cancellation.Token.ThrowIfCancellationRequested();
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        status = "ok",
        format = "safetensors",
        metadataValidation = "passed",
        weightCompatibility = "not_assessed",
        nativeInitialization = "not_requested",
        sizeBytes,
        tensorBytes,
        tensorCount = file.Tensors.Count,
        metadataEntryCount = file.Metadata.Count,
        dtypeCounts,
        rankCounts,
        readerUnsupportedDTypes = dtypeCounts.Keys.Where(d => !SafeTensorFile.SupportsTensorDType(d)).ToArray(),
        sha256,
        tensors = tensorLimit == 0 ? null : file.Tensors.OrderBy(t => t.Key, StringComparer.Ordinal).Take(tensorLimit)
            .Select(t => new { name = t.Key, dtype = t.Value.DType, shape = t.Value.Shape, sizeBytes = t.Value.End - t.Value.Start }).ToArray(),
        tensorDetailsTruncated = tensorLimit == 0 ? (bool?)null : file.Tensors.Count > tensorLimit
    }, jsonOptions));
    return 0;
}
catch (Exception error) when (error is ArgumentException or InvalidDataException or IOException or UnauthorizedAccessException or OperationCanceledException)
{
    string code = error switch
    {
        OperationCanceledException => "cancelled",
        ArgumentException => "invalid_arguments",
        InvalidDataException => "invalid_safetensors",
        UnauthorizedAccessException => "access_denied",
        _ => "io_error"
    };
    Console.Error.WriteLine(JsonSerializer.Serialize(new { status = "error", code, message = error.Message,
        usage = code == "invalid_arguments" ? usage : null }, jsonOptions));
    return code == "cancelled" ? 130 : code == "invalid_arguments" ? 2 : 1;
}
