# SD1.5 checkpoint assembly: contracts and evidence

This addition assembles three explicitly selected components from one safetensors
reader. It validates metadata, loads their weights transactionally and provides
independently disposable graph factories. It does not identify a model family or
execute a text-to-image workflow. The accompanying
[evidence record](sd15-checkpoint-assembly.json) records verification results and
their scope.

## Fixed public contract

`Sd15CheckpointLoader` fixes CLIP-L Large, `SdUnetConfig.Sd15` and the stock
classical VAE. It selects `cond_stage_model.`, `model.diffusion_model.` and
`first_stage_model.` respectively. The public API cannot substitute a reduced
profile or infer SD2/SDXL, EMA, prediction kind, latent scale or sampling schedule.

Without a CLIP projection, the stock schemas require **1,130 weight tensors**:
196 CLIP, 686 U-Net and 248 VAE. Their Float32 resident storage is
**4,264,941,228 bytes**. These are metadata calculations, not evidence of a
successful stock load. A recognized projection adds one tensor and 2,359,296
resident bytes; a malformed projection is rejected even though it is optional.

All three plans are inspected before the first native load. They remain bound to
the same reader instance; another reader with identical contents is rejected.
Storage dtypes F32, F16 and BF16 are supported by the component loaders and become
owned CPU Float32 weights. Admission requires an explicit weight-memory budget:

`EstimatedPeakWeightBytes = sum(component ResidentBytes) + max(component TemporaryBytes)`

Checked arithmetic guards the totals. This conservative estimate excludes
activations, runtime libraries, allocator overhead and process RSS; it is not an
available-memory or OOM guarantee.

Unclaimed tensors are rejected by default. The explicit
`ReportAndIgnoreOutsideComponents` option reports omissions outside the three
namespaces without reading them. It cannot hide an unknown tensor inside an
active namespace. Recognized CLIP omissions remain visible in its sub-plan.

## Reader, transaction and ownership

An internal guard checks that the reader is open and its file length still
matches inspection. Equal length does not prove unchanged payload bytes. The
caller must keep an externally supplied reader open and its file unchanged while
loading; the assembler neither reopens nor owns that reader. The path overload
opens once and closes its own reader through `using`.

CLIP, U-Net and VAE load sequentially. Completed and partially materialized banks
are released on failure or cancellation, including cancellation before the final
ownership transfer. No partial aggregate is returned. Acquired CLIP, U-Net and
VAE wrappers retain their own banks and survive disposal of the aggregate;
their last owner releases the corresponding weights.

`CreateClipEncoder()` uses the Sd1L wrapper with **unprojected pooling by default**.
Explicit projected pooling requires a real projection. `CreateUnet()` returns a
raw graph, and `CreateImageVae()` retains raw, unscaled latent conventions.

## Verification scope

The shared internal core uses a compatible reduced configuration for tests:
CLIP width16/two layers, U-Net base32/context16 and VAE base32. Combined files
contain **970 tensors**, or **971 with projection**. They are real safetensors
files with bounded synthetic payloads, not pretrained checkpoints.

There are **25 new tests: nine metadata/admission cases and sixteen reduced
native assembly cases**. Coverage includes optional projection, mixed storage
dtypes, budget and reader guards, rollback at partial and completed ownership
boundaries, independent factories and last-owner release. The corrected race test
uses a fresh aggregate with no keeper; it checks exactly which real tensor
wrappers survive a winning acquisition and that all are released afterward.
That corrected test has passed locally.

The complete local Windows CPU run finished with exit code zero: **1,322 tests
passed**, including 536 Inference tests, with no failures or ignored cases.
Six TRX reports and all 1,322 distinct test IDs were recounted. The 25 new cases
are included in that total. The CI runs them separately before the existing
numerical reference suites so their results can be collected even when those
later suites fail; this repetition does not add unique tests.

A separate fresh Windows process inspected the reduced combined metadata and
rejected an insufficient budget, a missing VAE and a closed reader. Its local
harness used direct managed references and a throwing TorchSharp import resolver:
zero native imports, component-load callbacks or matching native modules/bundled
files were observed. This additional local diagnostic does not establish native
isolation on other platforms or a successful stock load. The harness hashes and
the exact scope of its basename checks are recorded in the evidence JSON.

## Provenance and limits

Component layouts and wrapper conventions follow frozen ComfyUI
[`sd.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py)
at `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Existing
[CLIP](clip-source-210-d43ccfb.md) and
[SD component](sd-source-86812a9.md) source references remain separate numerical
evidence. This explicit C# assembly policy adds no new numerical oracle or
tolerance change.

No pretrained checkpoint, successful stock load, prolonged leak campaign or
end-to-end workflow is demonstrated. The new assembly tests exercise reduced
CLIP forwards for pooling/projection; U-Net and VAE factories are acquired and
retained, but these new tests do not execute their forwards after assembly.
Their existing individual graph tests retain their original scope. The public
path overload's successful stock-load path remains unexercised here.
