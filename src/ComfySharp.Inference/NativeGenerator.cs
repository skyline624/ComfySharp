using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>Creates an owned private generator; supplements TorchSharp's CPU-only native constructor.</summary>
internal static class NativeGenerator
{
    [DllImport("ComfySharp.Native", CallingConvention = CallingConvention.Cdecl)]
    private static extern nint CSGenerator_Initialize(nint handle, int deviceType, int deviceIndex, ulong seed);
    [DllImport("ComfySharp.Native", CallingConvention = CallingConvention.Cdecl)]
    private static extern int CSGenerator_AbiVersion();

    internal static Generator Create(ulong seed, Device device)
    {
        device = InferenceDevice.Validate(device); var generator = new Generator(seed, device);
        if (device.type == DeviceType.CPU) return generator;
        try
        {
            if (CSGenerator_AbiVersion() != 210000) throw new NotSupportedException("The native generator bridge must target libtorch 2.10.0.");
            nint error = CSGenerator_Initialize(generator.Handle, (int)device.type, device.index < 0 ? 0 : device.index, seed);
            if (error != 0) throw new InvalidOperationException("Native generator initialization failed: " + Marshal.PtrToStringUTF8(error));
            return generator;
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            generator.Dispose(); throw new NotSupportedException("CUDA adapter initialization requires the ComfySharp.Native generator bridge built against libtorch 2.10.0. No CPU random fallback was selected.", error);
        }
        catch { generator.Dispose(); throw; }
    }
}
