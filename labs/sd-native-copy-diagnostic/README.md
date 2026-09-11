# Isolated SD native-copy diagnostic

This laboratory tests native library origin on one Linux x64 runner. It does not qualify a backend, modify the product or replace accepted references. The existing SD15 reduced square test still compares its result with the committed reference under the unchanged numerical profile; a failed test remains a failed process and workflow.

The workflow runs two fresh source processes and six fresh .NET processes:

| Dispatch request | Source | Original build | Identical NuGet copy | Wheel-native copy |
|---|---|---|---|---|
| Unset (`auto`) | Existing pinned source operator script | Existing test | Same test | Same test |
| `default` | Existing pinned source operator script | Existing test | Same test | Same test |

All three .NET variants use the same `dotnet vstest` invocation, filter, adapter selection and working directory. Only the absolute test assembly, adapter and evidence paths differ. The filter must execute exactly one existing SD15 square test. This controls the switch from the earlier project-based invocation separately from the native substitution. See the [Microsoft VSTest CLI documentation](https://learn.microsoft.com/en-us/dotnet/core/tools/dotnet-vstest).

`staging.py` inventories every original build file and directory, rejects symlinks and conflicting native identities, and makes two independent complete copies. All copied files are rehashed; no copied file initially shares an inode with the original. Managed assemblies, dependencies, runtime configuration, adapters, fixtures and the TorchSharp binding stay byte-identical. The wheel variant replaces every inventoried core probe and GNU OpenMP payload. It keeps the original dependency asset filenames and adds `libgomp.so.1` aliases in the native probe directories. All aliases of a selected library are hardlinks to one internal copied inode, never to the package or original build. The ELF SONAME and DT_NEEDED closure must match the explicit core/OpenMP/system dependency allowlist.

The experiment does not set `LD_PRELOAD` or prepend a wheel directory to `LD_LIBRARY_PATH`. The existing pinned guard accepts only an empty baseline or the verified setup-python interpreter library directory and preserves it. Other environment values, including `PATH`, remain unchanged. Dispatch and diagnostic trace variables are the only changes.

Actual `Process.Modules` evidence must contain exactly one selected image for each core library, the unchanged bridge, and exactly one selected OpenMP image. The existing pinned strict gate rejects mixed origins, duplicate cores, NuGet OpenMP in the wheel process, and a Python binding. A loader failure or missing trace is invalid origin evidence; it cannot establish a mathematical difference. No fallback changes the loader or files to obtain a valid map.

Every successful trace includes the 686 parameter identities, three inputs, 43 actual tensor records, shapes, strides, alignment and off/on/off output hashes. The observer must be bit-neutral within each process. Reports compare each valid product origin with its same-mode source, the NuGet copy with the original, and the wheel copy with the NuGet copy. Native-origin attribution remains deferred if the original/copy control differs in bytes or layouts, or any integrity check fails. Even an eligible contrast identifies a package-level difference, not a specific primitive or a qualified platform.

The six accepted source scripts and four reused diagnostic helpers are hash-checked before and after execution. The existing source script verifies the frozen ComfyUI blobs and exact AST; the workflow fetches only those pinned source files and restores the existing hashed CPU wheel dependencies. No model weights are downloaded. Complete original, package, source, fixture and copy inventories are checked again after the processes. Evidence contains relative inventory names, library basenames and hashes; it excludes copied binary payloads.

Run only with the pinned Python 3.12.10 / PyTorch 2.10.0 CPU environment and an already built Linux x64 test output:

```sh
python -I -B labs/sd-native-copy-diagnostic/run.py \
  --source-directory /absolute/frozen-source \
  --staging-directory /absolute/absent-disposable-copies \
  --output /absolute/absent-evidence
```

Staging and evidence must be separate, absent directories outside the repository, original build, source snapshot and Python package inputs; ancestor/descendant overlaps are rejected. The workflow uploads only evidence JSON, logs and TRX files. Large copies remain in a separate runner temporary directory and are never included in artifacts. The workflow triggers only on changes to this laboratory or its own workflow file. Local validation is limited to syntax, filesystem simulations and validators; actual native execution belongs to this isolated CI experiment.
