# CUDA generator bridge

The same library now exports `CSRuntime_AbiVersion` and `CSRuntime_ReadBuildInfo`
for the loaded libtorch CPU dispatch/build configuration. The new exports are
optional for existing generation; the original generator ABI remains unchanged.
The C# runtime identity reader requires them and copies thread-local UTF-8 strings
synchronously. See [runtime identity](../../docs/NATIVE_RUNTIME_IDENTITY.md).

TorchSharp 0.107.0's native generator constructor returns a CPU generator even
when CUDA is requested; this was reproduced with the pinned Windows CUDA bundle.
The official [THSTorch.cpp](https://github.com/dotnet/TorchSharp/blob/main/src/Native/LibTorchSharp/THSTorch.cpp)
also identifies GPU generator construction as unfinished. Public managed device
metadata alone therefore does not establish the native generator's backend.

This small C ABI bridge supplements that missing operation. It clones libtorch's
selected CUDA default generator under its mutex and seeds the private clone.
Only after success does it replace the implementation in the `at::Generator`
object owned by TorchSharp. Its public handle and normal destructor continue to
own that object. The default generator is not reseeded or consumed. Native
exceptions become an error string at the C boundary; the managed wrapper disposes
its generator on failure. No private .NET reflection or global RNG fallback is used.

The ABI is pinned to TorchSharp 0.107.0 and libtorch 2.10.0. Compile with matching
standalone libtorch headers/import libraries and the same C++ runtime ABI. The
bridge links `torch_cpu`/`c10`, obtains the CUDA implementation through libtorch's
registered accelerator hooks, and uses the already loaded CUDA runtime. No CUDA
kernel compiler or Python interpreter is required to build or distribute it.

On Windows x64, with Visual Studio C++ Build Tools:

```powershell
pwsh -File tools/build-native-generator.ps1 -TorchSdkRoot <libtorch-sdk> -OutputDirectory <bridge-output>
dotnet build tools/ComfySharp.RuntimeProbe -c Release -p:NativeBackend=cuda -p:NativeRuntime=win-x64 -p:NativeGeneratorBridge=<absolute-path-to-ComfySharp.Native.dll>
```

`NativeGeneratorBridge` copies the supplied library beside the executable for
build and publish. CPU builds and existing inference paths do not require it.
CUDA adapter initialization without the bridge reports an explicit unsupported
operation. The CMake project is supplied for other toolchains; Linux/CUDA and
macOS accelerator qualification remain pending.

The Windows qualification used the already installed matching SDK headers/import
libraries; no SDK or model download was needed. Native libraries and build outputs
are not committed. Binary releases must include this bridge and the libtorch
runtime/notices alongside the application when this capability is advertised.
