# Local portable packaging validation

`tools/publish.ps1 -Runtime win-x64 -NativeBackend cpu` (or `linux-x64`, `osx-arm64` on
the corresponding OS) builds the self-contained Desktop and Host with locked
restore. Use a fresh `-OutputDirectory` on each run. The script writes per-file
SHA-256 checksums in `<OutputDirectory>/<Runtime>-<NativeBackend>` and a sibling
`.tar.gz` archive. Archive validation rejects
missing apphosts and, on Unix targets, missing owner execute permission for
`ComfySharp.Desktop` at the package root or `host/ComfySharp.Host` in the separate
Host folder (with `.exe` on Windows). The tar archive preserves Unix modes; uploading the raw publish
directory through GitHub artifact storage does not.

**These outputs are for local validation only. Binary redistribution is blocked.**
The manually dispatched portable workflow builds and validates archives on the
three target operating systems but deliberately has no binary upload step. The
workflow also extracts each archive to a fresh directory and starts its actual
Desktop apphost with the supervised, packaged Host, checking a real prompt result.
The normal CI workflow separately builds and tests the product. Source publication
does not depend on completion of binary packaging.

## Native bundle selection and process isolation

The optional [offline C# bundle tool](NATIVE_BUNDLES.md) prepares the pinned
Linux CPU candidate. Packaging can explicitly compose that bundle into the Host:

```powershell
./tools/publish.ps1 -Runtime linux-x64 -NativeBackend cpu -NativeBundleDirectory /path/to/prepared-bundle -NativeBundleRecipe native/bundles/linux-x64-cpu210-source.json -OutputDirectory /path/to/fresh-output
```

Both native bundle parameters are required together. This option currently
supports Linux CPU only and requires a matching recipe runtime. It publishes an
original Host under the private build root, then uses C# composition to produce
the package's independent `host/` directory. The original Host remains available
for inspection; it is not included in the archive. Notices and the composition
receipt are inside `host/` and covered by the package checksums. An existing
archive, conflicting staging directory or overlap with the prepared bundle is
rejected. Without these parameters, the NuGet selection below remains the default.
Desktop dependencies are published separately in both modes.

`tools/qualify-linux-portable.ps1` verifies extraction/checksums, executes the
actual self-contained Host with empty PATH and absent SDK roots, submits a native
sigma graph, and inspects its loaded image hashes through Linux `/proc`. It then
starts the extracted Desktop under Xvfb with the same application restrictions
and exercises its supervised Host. The dedicated CI runs both NuGet and composed
variants and uploads only reports/logs. These checks do not qualify model families
or remove the binary redistribution gate.

The [7723234 qualification](qualification/portable-native-bundle.json) passed
for both variants in [run 34719085344](https://github.com/skyline624/ComfySharp/actions/runs/34719085344).
Both extracted applications loaded the expected native images and their own
.NET runtime, reproduced both sigma previews, and passed the full current
Desktop/supervised-Host smoke. TorchSharp binding and .NET runtime hashes were
identical between variants. The runner still had developer software installed;
these checks prove the stated child-environment isolation, not a completely
clean machine or pretrained-model qualification.

The Desktop dependency closure stays at the archive root; the Host and **all**
its managed/native dependencies stay in `host/`. In particular, Avalonia currently
resolves SkiaSharp 3.119.4 while TorchSharp resolves SkiaSharp 2.88.6. Publishing
both applications into one directory could overwrite files needed by the other
process. Do not flatten or merge these folders.

The supervisor honors `COMFYSHARP_HOST_PATH` first, then searches `host/` before
the legacy flat layout and development build folders. A configured missing path
is reported as a launch error; the supervisor does not silently choose another
engine. Development auto-discovery selects the current platform's CPU build.
For a CUDA development build, set the override to that exact Host apphost or DLL.

`build/ComfySharp.Native.props` configures only Host, Host.Tests, Inference.Tests
and RuntimeProbe. Host.Tests inherits its payload from Host. Inference,
Nodes.Tensor, Catalog, ModelInspect and Desktop never select a libtorch payload;
schema registration and metadata inspection must remain possible without one.

`NativeRuntime` is the native RID and `NativeBackend` selects one distribution
bundle. An explicitly supplied `RuntimeIdentifier` selects that same RID;
contradictory properties fail restoration/build. The default is CPU on the
supported current OS/architecture. Exactly one libtorch bundle is selected:

| NativeRuntime | NativeBackend | Exact package version |
| --- | --- | --- |
| win-x64 | cpu | libtorch-cpu-win-x64 2.10.0 |
| linux-x64 | cpu | libtorch-cpu-linux-x64 2.10.0 |
| osx-arm64 | cpu | libtorch-cpu-osx-arm64 2.10.0 |
| win-x64 | cuda | libtorch-cuda-12.8-win-x64 2.10.0 |

`tools/publish.ps1 -Runtime win-x64 -NativeBackend cuda` creates the separate
Windows CUDA bundle for local validation. Unsupported combinations fail without
a CPU fallback. Linux CUDA packaging and real device qualification remain V1
requirements. `NativeBackend=mps` is not a supported distribution choice: MPS is
a future macOS compute-device qualification, separate from package selection.
CPU operations inside a CUDA bundle do not prove GPU kernels or model inference.
macOS retains the macOS 14 target and its existing native-loader/deployment-target
qualification reservations; selecting the macOS bundle does not qualify MPS.

Native consumers use separate output paths
`bin/native/<NativeRuntime>/<NativeBackend>/<Configuration>/net10.0/` (plus the
SDK's RID suffix when present), and intermediate paths under
`obj/native/<NativeRuntime>/<NativeBackend>/<build|publish>/`. Packaging supplies
a fresh artifact root partitioned by RID/backend too. Changing CPU/CUDA or RID
therefore cannot reuse a previous bundle's assets or leave its native DLLs in
the new variant. Ordinary `dotnet run` and `dotnet test` resolve these paths.

Native consumer locks are `packages.<NativeRuntime>.<NativeBackend>.lock.json`
for a build without `RuntimeIdentifier`, and
`packages.<NativeRuntime>.<NativeBackend>.publish.lock.json` for an explicit RID
graph. Pure libraries keep their generic or RID-specific locks. To regenerate
a publish graph deliberately, use, for example:

```powershell
dotnet restore src/ComfySharp.Host/ComfySharp.Host.csproj -p:RuntimeIdentifier=win-x64 -p:NativeRuntime=win-x64 -p:NativeBackend=cpu -p:SelfContained=true
```

Use the explicit singular `RuntimeIdentifier` property for this restore;
`dotnet restore --runtime` configures the SDK's plural runtime restore list and
is not a substitute for selecting the publish graph. Normal packaging always
restores in locked mode. A restored cross-platform graph is evidence of
dependency resolution only, not execution on that platform.

## Unresolved binary distribution requirements

The copied project GPL and `THIRD_PARTY_NOTICES.md` are not a complete set of
dependency license texts. No distribution SBOM is generated yet. The blocking
work is to establish and verify the exact published dependency closure for both
applications on each RID, including self-contained .NET/ASP.NET runtime packs
and native assets, and then ship all applicable license and notice texts.

NuGet license metadata alone is insufficient: for example, Avalonia 12.0.5's
cached package declares a license without including a standalone license text;
SkiaSharp.NativeAssets.Linux 3.119.4 includes both `LICENSE.txt` and a substantial
`THIRD-PARTY-NOTICES.txt`. Both applicable texts, including notices for embedded
native components, must be retained. Resolve license-expression or URL-only
packages to authoritative texts for the exact version, recording provenance.
Audit the actual published SQLite, Skia, Avalonia native assets and runtime packs,
rather than assuming that the top-level managed package license covers them.

Before enabling any binary artifact upload or release:

- Collect exact versions and package hashes from the RID-specific resolved
  assets/locks, including runtime packs selected by the SDK. Map the closure to
  published files and retain the applicable dependency license and notice texts.
- Produce a schema-valid SPDX or CycloneDX SBOM with exact component versions,
  package identities, cryptographic hashes, dependency relationships and license
  provenance. Include the distribution's file hashes. `SHA256SUMS` alone is not
  an SBOM, and the general dependency list is not evidence of notice completeness.
- Verify notice coverage and SBOM content for every RID, test the downloaded
  archive's contents and Desktop/Host executable modes on Linux and macOS, and
  explicitly remove the local-only restriction as part of that reviewed change.
- Upload only the validated `.tar.gz` files, never their raw source directories.

The Host distribution now includes TorchSharp/libtorch. Its CPU/CUDA native
closures, including all CUDA component packages when selected, require their
own exact notices and SBOM coverage. Desktop remains a separate closure without
libtorch. Do not infer notice completeness from successful restoration or archive
creation.
