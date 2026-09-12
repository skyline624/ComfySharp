using Avalonia;

namespace ComfySharp.Desktop;
public static class Program
{
    public static bool SmokeTest { get; private set; }
    public static string? ModelsDirectory { get; private set; }
    public static string? DataDirectory { get; private set; }
    public static string? InitialWorkflow { get; private set; }
    public static string? Sd15SmokeReport { get; private set; }
    public static string InferenceDevice { get; private set; } = "cpu";
    public static bool IsDiagnostic => SmokeTest || Sd15SmokeReport is not null;
    internal static int DiagnosticExitCode { get; set; } = 1;
    [STAThread]
    public static int Main(string[] args)
    {
        SmokeTest = args.Contains("--smoke-test");
        ModelsDirectory = Option(args, "--models-dir"); DataDirectory = Option(args, "--data-dir");
        InitialWorkflow = Option(args, "--workflow"); Sd15SmokeReport = Option(args, "--sd15-smoke-report");
        int deviceIndex = Array.IndexOf(args, "--inference-device");
        if (deviceIndex >= 0)
        {
            if (deviceIndex + 1 >= args.Length || args[deviceIndex + 1] is not ("cpu" or "cuda:0"))
                throw new ArgumentException("Select --inference-device cpu or cuda:0.");
            InferenceDevice = args[deviceIndex + 1];
        }
        if (Sd15SmokeReport is not null && (SmokeTest || InitialWorkflow is null || DataDirectory is null || File.Exists(Sd15SmokeReport)))
            throw new ArgumentException("SD1.5 smoke requires --workflow, --data-dir and a new report path, without --smoke-test.");
        int exitCode = BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        return IsDiagnostic ? DiagnosticExitCode : exitCode;
    }
    private static string? Option(string[] args, string name)
    {
        int index = Array.IndexOf(args, name);
        if (index < 0) return null;
        if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
            throw new ArgumentException($"{name} requires a path.");
        return Path.GetFullPath(args[index + 1]);
    }
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>().UsePlatformDetect().LogToTrace();
}
