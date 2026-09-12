using TorchSharp;

namespace ComfySharp.Inference;

/// <summary>Explicit Float32 inference devices. No automatic device fallback.</summary>
public static class InferenceDevice
{
    public static bool IsSupported(DeviceType type) => type is DeviceType.CPU or DeviceType.CUDA;

    /// <summary>Sets the process-wide CUDA Float32 policy before starting model operations.
    /// Host currently executes one prompt at a time. Callers must not change these flags during execution.</summary>
    public static void ConfigureFloat32(torch.Device device)
    {
        device = Validate(device);
        if (device.type != DeviceType.CUDA) return;
        // Float32 tensors alone do not prevent cuDNN from using reduced-mantissa TF32 kernels.
        torch.backends.cuda.matmul.allow_tf32 = false;
        torch.backends.cudnn.allow_tf32 = false;
    }

    public static torch.Device Validate(torch.Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (!IsSupported(device.type)) throw new NotSupportedException("This inference path supports CPU and CUDA devices only.");
        return device.type == DeviceType.CUDA && device.index < 0 ? new torch.Device(DeviceType.CUDA, 0) : device;
    }

    public static bool Same(torch.Device first, torch.Device second) => first.type == second.type &&
        (first.type == DeviceType.CPU || first.index == second.index);

    internal static void RequireSame(torch.Device expected, torch.Tensor tensor, string name)
    {
        if (!Same(expected, tensor.device)) throw new ArgumentException($"{name} must be on {expected}; explicit transfer is required.", name);
    }
}
