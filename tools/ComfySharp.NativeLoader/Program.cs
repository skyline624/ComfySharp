using System.Runtime.InteropServices;
using System.Text.Json;

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: ComfySharp.NativeLoader <absolute-native-library-path>");
    return 2;
}
try
{
    var handle = NativeLibrary.Load(Path.GetFullPath(args[0]));
    try { Console.WriteLine(JsonSerializer.Serialize(new { loaded = true, architecture = RuntimeInformation.ProcessArchitecture.ToString(), os = RuntimeInformation.OSDescription })); }
    finally { NativeLibrary.Free(handle); }
    return 0;
}
catch (Exception error) when (error is DllNotFoundException or BadImageFormatException or FileLoadException)
{
    Console.Error.WriteLine(JsonSerializer.Serialize(new { loaded = false, architecture = RuntimeInformation.ProcessArchitecture.ToString(), os = RuntimeInformation.OSDescription, error = error.ToString() }));
    return 1;
}
