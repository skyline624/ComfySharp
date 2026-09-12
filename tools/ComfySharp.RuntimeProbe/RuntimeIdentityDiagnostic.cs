using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using ComfySharp.Inference;
using static TorchSharp.torch;

namespace ComfySharp.RuntimeProbe;

internal static class RuntimeIdentityDiagnostic
{
    internal static int Run(string[] arguments,TextWriter writer)
    {
        try
        {
            if(arguments.Length!=0)throw new ArgumentException("Usage: runtime-info");
            var info=NativeRuntimeBuildInfo.Read();
            using var process=Process.GetCurrentProcess();
            string managed=Path.GetFullPath(typeof(Tensor).Assembly.Location);
            var paths=process.Modules.Cast<ProcessModule>().Select(m=>m.FileName).Distinct(StringComparer.Ordinal)
                .Where(p=>!string.Equals(Path.GetFullPath(p),managed,OperatingSystem.IsWindows()?StringComparison.OrdinalIgnoreCase:StringComparison.Ordinal))
                .Where(p=>new[]{"torch","c10","gomp","iomp","libomp","ComfySharp.Native","python"}.Any(s=>Path.GetFileName(p).Contains(s,StringComparison.OrdinalIgnoreCase)))
                .OrderBy(Path.GetFileName,StringComparer.Ordinal);
            var libraries=paths.Select(path=>
            {
                using var stream=File.OpenRead(path);
                return new{name=Path.GetFileName(path),bytes=stream.Length,sha256=Convert.ToHexStringLower(SHA256.HashData(stream))};
            }).ToArray();
            using var managedStream=File.OpenRead(managed);
            var identity=new
            {
                schema="comfysharp-native-runtime-identity-v1",os=OperatingSystem.IsWindows()?"windows":OperatingSystem.IsLinux()?"linux":OperatingSystem.IsMacOS()?"macos":"unknown",
                architecture=RuntimeInformation.ProcessArchitecture.ToString(),dotnet=RuntimeInformation.FrameworkDescription,
                info.CpuCapability,info.BuildConfiguration,threads=get_num_threads(),interopThreads=get_num_interop_threads(),
                torchSharpAssemblySha256=Convert.ToHexStringLower(SHA256.HashData(managedStream)),
                controls=new[]{"ATEN_CPU_CAPABILITY","OMP_NUM_THREADS","MKL_NUM_THREADS","MKL_CBWR","ONEDNN_MAX_CPU_ISA","DNNL_MAX_CPU_ISA"}
                    .ToDictionary(n=>n,Environment.GetEnvironmentVariable,StringComparer.Ordinal),libraries
            };
            var options=new JsonSerializerOptions{PropertyNamingPolicy=JsonNamingPolicy.CamelCase};
            string fingerprint=Convert.ToHexStringLower(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(identity,options)));
            writer.WriteLine(JsonSerializer.Serialize(new{success=true,identitySha256=fingerprint,identity,
                qualification="Observed runtime identity only; no source profile or model qualification is implied."},options));
            return 0;
        }
        catch(Exception error)
        {
            writer.WriteLine(JsonSerializer.Serialize(new{success=false,error=error.GetType().Name,message=error.Message}));
            return 1;
        }
    }
}
