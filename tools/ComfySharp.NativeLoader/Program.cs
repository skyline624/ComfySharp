using System.Runtime.InteropServices;
using System.Text.Json;

if (args.Length == 0)
{
    Console.Error.WriteLine("Usage: ComfySharp.NativeLoader <absolute-native-library-path> [additional-library-path ...]");
    return 2;
}
var handles = new Stack<nint>();
try
{
    foreach (var path in args)
    {
        handles.Push(NativeLibrary.Load(Path.GetFullPath(path)));
        Console.WriteLine(JsonSerializer.Serialize(new { loaded = true, library = Path.GetFileName(path), architecture = RuntimeInformation.ProcessArchitecture.ToString(), os = RuntimeInformation.OSDescription }));
    }
    return 0;
}
catch (Exception error) when (error is DllNotFoundException or BadImageFormatException or FileLoadException)
{
    Console.Error.WriteLine(JsonSerializer.Serialize(new { loaded = false, architecture = RuntimeInformation.ProcessArchitecture.ToString(), os = RuntimeInformation.OSDescription, error = error.ToString() }));
    return 1;
}
finally
{
    while (handles.TryPop(out var handle)) NativeLibrary.Free(handle);
}
