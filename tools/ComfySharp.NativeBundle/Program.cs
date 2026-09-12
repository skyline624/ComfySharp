using System.Text.Json;
using ComfySharp.NativeBundle;

using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
try
{
    if (args.Length is not (7 or 9) || args[0] is not ("prepare" or "compose")) throw new ArgumentException("Usage: prepare --recipe FILE --archive FILE --output NEW_DIR | compose --recipe FILE --bundle DIR --application DIR --output NEW_DIR");
    var options = new Dictionary<string, string>(StringComparer.Ordinal);
    for (int i = 1; i < args.Length; i += 2) if (!options.TryAdd(args[i], args[i + 1])) throw new ArgumentException("Duplicate option.");
    string[] allowed = args[0] == "prepare" ? ["--recipe", "--archive", "--output"] : ["--recipe", "--bundle", "--application", "--output"];
    if (!options.Keys.ToHashSet(StringComparer.Ordinal).SetEquals(allowed)) throw new ArgumentException("Unexpected or missing option.");
    if (args[0] == "prepare") BundleOperations.Prepare(options["--recipe"], options["--archive"], options["--output"], cancellation.Token);
    else BundleOperations.Compose(options["--recipe"], options["--bundle"], options["--application"], options["--output"], cancellation.Token);
    Console.WriteLine(JsonSerializer.Serialize(new { success = true, operation = args[0], qualification = "File integrity and composition only; numerical and hardware qualification remain separate." })); return 0;
}
catch (Exception error)
{
    Console.WriteLine(JsonSerializer.Serialize(new { success = false, error = error.GetType().Name, message = error.Message })); return error is OperationCanceledException ? 130 : 1;
}
