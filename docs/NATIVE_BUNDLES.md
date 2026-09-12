# Offline native bundles

Portable publishing now accepts this bundle explicitly through the paired
`NativeBundleDirectory` / `NativeBundleRecipe` options. See the
[packaging procedure and extracted-application qualification](packaging.md#native-bundle-selection-and-process-isolation).
The default NuGet selection remains available and unchanged.

`ComfySharp.NativeBundle` prepares explicitly pinned native libraries and their
notices, then composes an independent application directory. Both operations use
C#/.NET only, work offline, and never execute archive contents. The original
application and prepared bundle remain unchanged. Model weights are not involved.

The initial [Linux CPU recipe](../native/bundles/linux-x64-cpu210-source.json)
selects libtorch 2.10 binaries previously compared with same-host ComfyUI source.
It does not change the default NuGet runtime or establish general model parity.
CUDA, macOS and other builds need separate recipes and qualification. The tool
does not select CPU dispatch or thread counts.

## Prepare

```sh
dotnet restore tools/ComfySharp.NativeBundle --locked-mode
dotnet build tools/ComfySharp.NativeBundle -c Release --no-restore
dotnet tools/ComfySharp.NativeBundle/bin/Release/net10.0/ComfySharp.NativeBundle.dll prepare \
  --recipe native/bundles/linux-x64-cpu210-source.json \
  --archive /path/to/torch-cpu210.whl --output /path/to/new-bundle
```

Obtain the archive separately from the [official PyTorch CPU distribution](https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl).
Its SHA-256 is `ee40b8a4b4b2cf0670c6fd4f35a7ef23871af956fecb238fbf5da15a72650b1d`.
The tool has no downloader. The wheel is used as an archive: no installation,
Python interpreter, Python binding or Python source is needed for preparation.

The recipe selects four native images (`libc10.so`, `libtorch_cpu.so`,
`libtorch.so`, `libgomp.so.1`) and complete `licenses/LICENSE` and
`licenses/NOTICE` entries from `torch-2.10.0+cpu.dist-info`. Every selected entry
has an exact length and hash. Preparation verifies the whole archive, extracts
only that allowlist, bounds metadata and expanded payload, rejects unsafe paths,
duplicate entries and symlinks, and publishes a new directory atomically. Classic
single-disk ZIP is supported; ZIP64 and archives above 2 GiB are rejected.
The receipt includes the canonical recipe hash and selected file hashes.

## Compose

```sh
dotnet tools/ComfySharp.NativeBundle/bin/Release/net10.0/ComfySharp.NativeBundle.dll compose \
  --recipe native/bundles/linux-x64-cpu210-source.json \
  --bundle /path/to/new-bundle --application /path/to/original-build \
  --output /path/to/new-application
```

Composition accepts flat native publish layouts or `runtimes/<rid>/native`.
It verifies the receipt and payload against the independently supplied recipe,
requires known hashes for every replaced original, and preserves the qualified
`libLibTorchSharp.so` binding. It copies other application files and Unix
executable permissions, adds declared native aliases, carries notices to
`third-party/<recipe-id>/`, and writes `comfysharp-native-bundle.json`.
Inputs are rechecked before atomic publication. Existing output directories,
overlapping trees, input-tree symlinks and conflicting payloads are rejected.
Use stable build inputs; this offline build tool does not sandbox concurrent
hostile filesystem changes.

The process returns JSON and exit code 0 on success, 1 on error and 130 on
cancellation. Success means verified file composition. Numerical profiles,
hardware support, full distribution notices and the SBOM remain separate gates.

## Qualification

The [Linux training campaign](../.github/workflows/training-trace.yml) prepares the
real archive with .NET before creating the separate Python source laboratory.
It composes the candidate with this tool, independently checks its files against
laboratory evidence, verifies loaded module origins and CPU build identity, and
compares training on the same host. The original build and an independent NuGet
copy remain controls. Cross-host fixture failures stay visible in TRX and are
not converted to passing tests by composition. No binary payload is uploaded by
this campaign. Application-contract CI also tests this tool on all three OSes.

The [qualification record](qualification/native-bundle.json) for commit
`1f8db165e353d6b3c39ddf49f42a247f3f5e6822` records 24 passing tool tests on
Windows, Linux and macOS. The [real Linux bundle campaign](https://github.com/skyline624/ComfySharp/actions/runs/34718270857)
completed successfully: all 282 selected training captures match the same-host
source exactly, the binding and other application files are preserved, and
the source/build/bundle input inventories remain unchanged.

This runner used **AVX512**, whereas the earlier identity campaign used AVX2.
The ordinary candidate suite passed **851/870**, retaining 19 failures against
historical references; the original NuGet suite passed **850/870**. Those counts
are not a model compatibility score. Numerical profile admission remains open.
The [normal CI](https://github.com/skyline624/ComfySharp/actions/runs/34718270875)
passed on Windows and failed on Linux training comparisons and macOS stock CLIP.
