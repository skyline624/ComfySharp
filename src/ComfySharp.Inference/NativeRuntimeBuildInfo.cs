using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Configuration reported by the loaded libtorch image, not inferred from CPU hardware flags.</summary>
public sealed record NativeRuntimeBuildInfo(string CpuCapability, string BuildConfiguration)
{
    [DllImport("ComfySharp.Native", CallingConvention=CallingConvention.Cdecl)]
    private static extern int CSRuntime_AbiVersion();
    [DllImport("ComfySharp.Native", CallingConvention=CallingConvention.Cdecl)]
    private static extern nint CSRuntime_ReadBuildInfo(out nint capability, out nint configuration);

    /// <summary>Requires the runtime identity bridge built against libtorch 2.10.0. Does not alter threads or CPU dispatch.</summary>
    public static NativeRuntimeBuildInfo Read()
    {
        NativeRuntimeBootstrap.Initialize();
        InitializeDeviceType(DeviceType.CPU);
        try{return ReadCore(CSRuntime_AbiVersion,CSRuntime_ReadBuildInfo);}
        catch(Exception error) when(error is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            throw new NotSupportedException("Runtime identity requires ComfySharp.Native with the libtorch 2.10.0 identity exports. CPU hardware flags are not a substitute for the loaded runtime's dispatch.",error);
        }
    }

    internal delegate nint ReadNative(out nint capability, out nint configuration);
    internal static NativeRuntimeBuildInfo ReadCore(Func<int> abi,ReadNative read)
    {
        if(abi()!=210000)throw new NotSupportedException("The runtime identity bridge must target libtorch 2.10.0.");
        nint error=read(out nint capability,out nint configuration);
        if(error!=0)throw new InvalidOperationException("Native runtime identity failed: "+Marshal.PtrToStringUTF8(error));
        // Copy both thread-local native buffers synchronously; never expose borrowed pointers.
        string? cpu=Marshal.PtrToStringUTF8(capability),build=Marshal.PtrToStringUTF8(configuration);
        if(string.IsNullOrWhiteSpace(cpu)||string.IsNullOrWhiteSpace(build))throw new InvalidDataException("The native runtime returned an incomplete build identity.");
        return new(cpu,build);
    }
}
