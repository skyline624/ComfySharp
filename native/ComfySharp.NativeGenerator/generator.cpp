#include <ATen/Context.h>
#include <ATen/core/Generator.h>
#include <torch/version.h>
#include <cstdint>
#include <mutex>
#include <string>

#if TORCH_VERSION_MAJOR != 2 || TORCH_VERSION_MINOR != 10 || TORCH_VERSION_PATCH != 0
#error ComfySharp native generator bridge requires libtorch 2.10.0 headers.
#endif
#if defined(_WIN32)
#define CS_EXPORT extern "C" __declspec(dllexport)
#else
#define CS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// The supplied handle is the at::Generator object owned by TorchSharp 0.107.0.
// Replace its implementation only after successful creation. TorchSharp remains its sole owner.
CS_EXPORT const char* CSGenerator_Initialize(void* handle, int deviceType, int deviceIndex, std::uint64_t seed) noexcept
{
    static thread_local std::string error;
    try {
        if (!handle) throw std::invalid_argument("Generator handle is null.");
        if (deviceType != static_cast<int>(at::kCUDA)) throw std::invalid_argument("This bridge currently admits CUDA generators only.");
        if (deviceIndex < 0 || deviceIndex > 127) throw std::invalid_argument("CUDA device index is invalid.");
        auto original = at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(deviceIndex)));
        auto cloned = [&]() {
            std::lock_guard<std::mutex> lock(original.mutex());
            return original.clone();
        }();
        cloned.set_current_seed(seed);
        *static_cast<at::Generator*>(handle) = std::move(cloned);
        return nullptr;
    } catch (const std::exception& failure) {
        error = failure.what(); return error.c_str();
    } catch (...) {
        error = "Unknown native generator initialization failure."; return error.c_str();
    }
}
CS_EXPORT int CSGenerator_AbiVersion() noexcept { return 210000; }
