# Reduced U-Net and CFG native-origin suite

This isolated diagnostic extends the native-copy experiment to the eight existing reduced U-Net cases and six existing CFG policy results. It does not modify the product, source generators, accepted fixtures, numerical profile or dependency locks. It uses no pretrained model and establishes no stock-model or platform qualification.

One Linux host runs eight fresh processes: source, original build, identical NuGet copy and wheel-native copy, each under auto and explicit default. The six product processes execute the same fourteen existing reference tests through the same VSTest invocation. Only the assembly/adapter and evidence locations differ. Copies are fresh and separate from the original build, source, Python installation and artifact root.

## Two separate results

Each test still compares all three computed outputs with its **committed accepted reference**, using the unchanged bound. Those fourteen test results per product process remain visible in TRX; a failed assertion stays failed. No diagnostic source output is installed as a fixture or substituted into those tests.

Separately, the harness compares each captured result with the independent source execution on the **same host and requested mode**. Such comparisons are eligible for native-package interpretation only after provenance, loaded-origin and copy-control gates pass. The original and NuGet copy must be bit-exact for all inputs and outputs, with identical shapes, strides, alignment flags and parameter layouts. An ineligible contrast is explicitly listed and makes the workflow fail even if the existing tests happen to pass. It cannot be presented as evidence for a native package effect. Eligible contrasts still do not identify a specific primitive or compiler flag.

Wheel/NuGet inputs must additionally have identical values, shapes, strides and alignment flags, and their parameter layouts must match. A measured layout/alignment difference leaves all metrics visible but makes the package contrast confounded and ineligible; this suite has no exact primitive-input control that could remove that confound.

Source processes must exit zero. A product process is valid only with exit zero and fourteen Passed tests, or exit one and fourteen executed tests including at least one Failed assertion. TRX summary/counters must agree; other exit codes, aborts or global/teardown errors invalidate attribution. The only accepted error notifications in RunInfos are xUnit `[FAIL]` lines naming an actual Failed test exactly. Ordinary numerical failures remain failed acceptance results even when their diagnostic captures are valid.

## Source and observations

`source.py` hash-checks the six accepted `labs/sd-source` scripts, then calls the unchanged U-Net and CFG generators. Their frozen AST, operations, configurations, parameter recipe, case order and inputs remain the computational source. The new adapter enriches their existing tensor-record calls with stride/alignment metadata and wraps the exact bound `_forward` method only to inspect parameter identities after the completed call. A weak bound-method reference preserves the model's original lifetime. It does not implement any replacement model operation or read accepted expected outputs to calculate results.

Source U-Net retains its existing **on/off/off** hook sequence with three equal hashes. Product U-Net uses **off/on/off**, reusing its nine existing coarse observation points; there is no new product hook. CFG uses no model callback: all three captures are made after the forward completes. Both policy outputs retain their separate source references, including the source-compatible fallback for context lengths 2/10. No Separate≈Concat invariant is asserted.

The sole new test option is `COMFYSHARP_SD_NATIVE_SUITE_TRACE_DIR`. It is rejected in combination with the other U-Net/CFG trace or latent-offset options. Without it, the ordinary test computations and assertions keep their original order. With it, inputs and parameters are inspected, host buffers are allocated, three outputs are retained, and evidence is written atomically before the same three reference assertions. Coarse callbacks copy contiguous CPU bytes to preallocated host buffers and never allocate native tensors. Unexpected noncontiguous boundaries fail explicitly instead of being silently normalized.

These added metadata operations and retained outputs **can change allocation and timing compared with normal CI**. No equivalence to the uninstrumented allocation history is claimed. The three repeat hashes, callback neutrality and original/copy control are therefore explicit evidence requirements. A grouped fourteen-case process is not a first-use process for every case.

## Evidence and strict gates

Each product variant must produce exactly fourteen traces: eight sets of nine U-Net boundaries plus final output, and six CFG final outputs (**86 output/intermediate records**, plus full inputs). Records contain F32 values/SHA, shape, stride and alignment. Actual parameters are hashed before and after calculation, with layouts; input hashes are also checked before/after. Source records and product records are explicitly matched against the committed **input/configuration/parameter** corpus, including CFG scale, policy, LCM eligibility and prediction kind. Expected output values are never used in this matching step.

Operation provenance records the actual Python/PyTorch version and PyTorch commit, NumPy/einops versions, exact source AST hashes, and product TorchSharp version, declared LibTorch package, .NET identity and loaded binary hashes. A declared version does not replace loaded binary evidence. Source auto reports its observed ATen capability; product records its request without pretending that CPU flags prove actual dispatch.

Source native identities come from actual Linux process mappings after both generators finish. Product identities come from fresh Process.Modules enumeration after every case; the harness requires the fourteen snapshots within a process to agree. Distinct loaded paths remain distinct records even when basenames match. The original strict gate requires unique selected libc10/libtorch_cpu/libtorch images, the unchanged bridge, and one selected OpenMP image. Python bindings are permitted in the Python source process and prohibited in the C# processes. Missing or mixed maps invalidate attribution before numerical comparison.

The actual loaded managed `TorchSharp.dll` is recorded separately with its size/SHA and checked against the unchanged build. Only that exact assembly path is excluded from the native list; another mapped image sharing its basename remains visible and cannot silently bypass the native-origin gate.

Copy safety, ELF closure, internal hardlink aliases and the loader-environment guard are reused from hash-pinned older diagnostic helpers. The original, both copies, source snapshot, wheel package and accepted fixture inventories are checked before and after; managed code, adapters, runtime configuration and binding remain unchanged. No LD_PRELOAD is added, the verified baseline LD_LIBRARY_PATH and PATH are preserved, and no fallback substitutes an alternate execution after a loader failure.

The comparator validates serialized little-endian F32 bytes before computing the unchanged `3e-5 + 3e-5*abs(expected)` bound. This U-Net/CFG corpus is finite; an unexpected NaN or infinity fails its integrity check. Missing cases, tensor labels, repeats, parameters, layouts or TRX results are errors. Source/source and product/source comparisons are reported separately, with U-Net boundaries ordered by actual source progression. Only a later targeted observation with an exact primitive input could attribute a first operator difference.

## Execution

The new workflow triggers only for this directory or its own workflow file. It fetches six already pinned ComfyUI blobs, installs the unchanged hash-locked CPU environment, builds the existing tests once, then runs:

```sh
python -I -B labs/sd-native-suite-diagnostic/run.py \
  --source-directory /absolute/frozen-source \
  --staging-directory /absolute/absent-copies \
  --output /absolute/absent-evidence
```

Artifacts contain only JSON, logs and TRX; native copy trees are outside the upload root. Python syntax and stdlib validator tests are permitted locally. Native execution and .NET validation are coordinated separately before publication. Sampling, VAE, autograd/backward, latent-offset sweeps and stock cases are outside this step.
