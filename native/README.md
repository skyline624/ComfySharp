# Native inference foundations

`ComfySharp.Inference` uses TorchSharp 0.107.0. Executables that perform tensor work must supply a matching libtorch 2.10.0 native package. The runtime probe and inference tests select native packages with `NativeRuntime`, defaulting by build OS to `win-x64`, `linux-x64`, or `osx-arm64`. These are the intended architectures, not a claim that all were executed. Windows x64 CPU and an opt-in CUDA 12.8 probe on RTX 3090 have execution evidence. macOS Intel is unsupported. Separate `packages.<NativeRuntime>.<NativeBackend>.lock.json` files cover all three CPU runtimes and the optional probe Windows CUDA runtime. All seven configurations passed locked restore on Windows; Linux/macOS execution still requires their own hardware runners. Setting NativeRuntime selects the package and lock, not a .NET cross-publish RID.

Run from the repository root, choosing an artifacts directory on a drive with sufficient free space:

```powershell
rtk proxy dotnet test tests/ComfySharp.Inference.Tests --artifacts-path <artifacts-directory>
rtk proxy dotnet run --project tools/ComfySharp.RuntimeProbe --artifacts-path <artifacts-directory> -- --device cpu --repeat 25
```

The probe emits JSON and returns a nonzero exit code for unavailable or unimplemented backends. `--device cuda` never falls back to CPU. `--device mps` currently reports an explicit unsupported error. CUDA 12.8 packages are deliberately absent from the default build. Build the optional Windows CUDA probe with `-p:NativeBackend=cuda -p:NativeRuntime=win-x64` and a **separate** artifacts directory; its native packages are captured in `packages.win-x64.cuda.lock.json`. Do not mix CPU and CUDA native binaries. Run the resulting probe executable with `--device cuda --repeat 25`.

The probe checks real autograd and SGD, matrix multiplication, tensor view/storage lifetime, convolution, scaled dot product attention, and per-call CPU generator reproducibility. CUDA mode asserts GPU tensor placement and exact CPU-noise transfer roundtrip. These operations passed 25 times on RTX 3090 with driver 610.88; this is native primitive evidence, not model qualification. Private process memory is recorded after each dispose scope; this includes allocator caches and unrelated process activity and does not prove absence of leaks. CPU RNG repeatability is only a same-runtime contract, not bitwise parity across backends or with ComfyUI draw ordering.

The strict safetensors reader checks bounded headers, unique keys, supported dtype widths, checked shape products, exact ranges, overlaps, holes and trailing unindexed bytes. It validates metadata for common integer and floating types, but materializes **F32 only**, copying into independent CPU tensor storage. The default single-tensor bound is 512 MiB; materialization also temporarily allocates byte and float buffers. This is an initial bounded reader, not mmap/offload support. File handles and intermediate tensors are deterministically disposed; returned tensors belong to the caller or their enclosing TorchSharp dispose scope. Cancellation is checked before and after native calls and between file-read chunks; it cannot preempt a native kernel already executing.

`NativeMath.EulerStep` implements only the no-churn single-step equation with borrowed F32/F64 tensors and descending nonnegative sigmas. It has no denoiser, scheduler, model, conditioning, CFG, or image pipeline. SD1.5, training workflows, and model-family compatibility remain unqualified.

Measured Windows CPU package size: about 76 MiB compressed and 332 MiB in the NuGet cache including its archive. Probe plus test build outputs measured about 616 MiB. Existing CUDA 12.8 split-package cache measured about 6.73 GiB including archives; the isolated CUDA probe artifacts occupied another 4.12 GiB. No model weights are included or downloaded.
