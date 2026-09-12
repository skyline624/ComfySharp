# Prospective IMAGE primitives source laboratory

This opt-in laboratory collects four real frozen ComfyUI node bodies:
`EmptyImage`, `ImageInvert`, `RepeatImageBatch`, and `ImageFromBatch`. It has
**not yet been executed** at protocol preparation. First execution requires
independent review, publication of these exact three files, and explicit ROOT
admission. No product output is used as an expected value.

The prospective profile is `image-primitives-cpu-f32-bits-v1`: CPU Float32,
rank-four NHWC images, positive batch/height/width, RGB or RGBA, finite inputs
and outputs. Comparison is **exact equality of logical little-endian Float32
payload bytes**, including signed zero; there is no numerical tolerance.
This does not qualify image codecs, GPUs, other dtypes, arbitrary channel
counts, gradient propagation, HTTP validation or whole workflows. `ImageBatch`
and its resizing/cropping are deliberately deferred to a separate profile.

## Frozen source and genuine schema

The backend is [ComfyUI `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a).
`protocol.json` records six canonical Git blobs and 59 exact AST selections,
including their line, source-segment hash and AST hash. Selected declarations
are parsed and compiled without rewriting them. There is no whole-module
import of `nodes.py` or the model-management module.

- [EmptyImage and ImageInvert](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L1936)
  execute their unchanged methods and real legacy `INPUT_TYPES`. Legacy
  metadata is explicitly an observation of those declarations, **not** a
  fabricated complete `/object_info` response.
- [RepeatImageBatch and ImageFromBatch](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_images.py#L260)
  use their unchanged classes, real V3 schema and `GET_NODE_INFO_V1`, including
  source module attribution after initial schema construction.
- V3 uses the genuine `io.Image` declaration and its `Type = torch.Tensor`,
  with the real Torch module injected only after execution admission. The
  surrounding V3/internal closure is reused from the published Autogrow
  laboratory; its broader type definitions are retained without claiming
  these IMAGE cases exercise every such definition.
- The actual `intermediate_device` and `intermediate_dtype` functions run with
  `gpu_only=false` and `fp16_intermediates=false`. The supplied `args` object
  contains these explicit runtime options; no replacement image body or
  dtype/device function is used. `MAX_RESOLUTION` is the actual source constant.

Source quirks are retained: EmptyImage creates three channel tensors and
concatenates them; it ignores its constructor's `self.device` in this source
version. ImageInvert preserves the alpha channel only for RGBA, and does not
clamp other channels. RepeatImageBatch repeats the entire batch sequence.
ImageFromBatch adjusts a negative index once, clamps it, bounds the length,
and clones the selected slice even when selecting the full batch.

## The 28 cases

Each node has seven small cases with all arguments supplied explicitly.

| Node | Cases fixed in the protocol |
|---|---|
| EmptyImage | Black, white, low-blue, asymmetric channels, red-orange, green and RGB; rectangles and batches 1/2/3 |
| ImageInvert | RGB, alpha-bearing RGBA, batch-two, spatial step-two view, NCHW-backed NHWC view, offset RGB, offset RGBA |
| RepeatImageBatch | Amount-one copy, whole batch sequence repeated three times, RGBA, spatial view, NCHW view, nonzero offset, batch-three |
| ImageFromBatch | First, middle, last, negative last, full batch via negative index, negative out-of-range clamp, positive out-of-range clamp |

Inputs include finite values outside [0,1], and RGBA alpha values include
negative zero, zero, one, -0.25, 1.25 and 0.5. Their storage is embedded as exact
little-endian F32 bytes with SHA-256. No RNG or model is involved. A native
CPU clone of that storage is used to construct the specified `as_strided`
view; no source input is silently made contiguous. The native storage bytes
are hashed after construction and checked against the protocol. A stdlib
preflight validates all view offsets, positivity, non-overlap, finite storage
and bounds before any Torch import. Individual storage is at most 1,024 F32
elements and a captured output at most 4,096 elements.

Three independent repeats each create fresh source module namespaces, input
storage and node instances/classes. No profiler or observer wraps the body.
The direct source invocation occurs exactly once per repeat: 84 body calls
for 28 cases. This does not exercise source mapper batching, prompt admission,
blockers or HTTP acquisition.

## Captures and copy observation

Every tensor capture includes logical F32 bytes, SHA-256, byte length, shape,
strides, storage offset, dtype/device, requires-grad and whether its first
element is 64-byte aligned. Raw addresses are never published. Capturing a
strided tensor may copy its logical contents after its original layout has
been measured; that capture copy is not supplied to the source operation.

For the 21 cases with an image input, each repeat captures input-before,
output and input-after-body and requires the input to be unchanged. Only
then a separate post-body observation adds F32 0.5 at the explicitly declared
per-case `mutationIndex`. For extraction cases this index lies inside the
selected slice, so the check cannot pass merely by mutating a discarded batch.
It records the before/after scalar bits, the mutated input and the output
after mutation, and requires that the input changed while the output bytes
did not. This is a controlled check of storage independence, **not** part of
the source algorithm, not a second source invocation and not a claim about
arbitrary aliases. EmptyImage explicitly reports that no input-image mutation
test applies. All three repeat records are retained and their complete
canonical JSON hashes must match. These include signed-zero payload bits;
JSON floating-point values are not an output oracle.

## Isolated environment and commands

Use the existing published CPU closure from
[`../clip-source`](../clip-source/README.md): CPython **3.12.10**, Torch
**2.10.0+cpu** on Windows/Linux or **2.10.0** on macOS ARM64, NumPy **2.2.6**
and einops **0.8.1**. No environment is created or installed by this collector.
The existing hashed `requirements-<target>.txt` files are unchanged. A prepared
external venv is required; if installation is separately admitted, use the
published lock workflow with isolated pip, `--require-hashes`,
`--only-binary=:all:` and the direct hashed URLs/no-index closure already in
those files. There is no dependency on Python in the application or .NET tests.

The default command is stdlib-only preflight; it imports verified inert helper
driver definitions and compiles selected AST, but **does not execute source
declarations or import Torch, NumPy or einops**:

```text
python -I -B -S labs/image-primitives-source/reference.py --source <backend-git-repository> --target win-x64
```

After review, publication and ROOT admission, from the prepared external
venv in a fresh process (omit `-S` so installed packages are available):

```text
python -I -B labs/image-primitives-source/reference.py --source <backend-git-repository> --target win-x64 --execute --output <new-absolute-external-directory>
```

Use the matching target `linux-x64` or `osx-arm64` on those hosts. The selected
target must match the actual OS/architecture. Keep `ATEN_CPU_CAPABILITY` unset:
this profile does not silently substitute a forced-dispatch experiment.
Intra-op and inter-op threads are both fixed to one, with no gradients. The
output directory must not exist and must be outside both Git repositories.
On a separately admitted CI workflow, use `RUNNER_TEMP` for environment and
output. This sub-lot adds no workflow and launches no collection.

## Provenance and publication

Before importing Torch, the collector compares all three public files with
the current Git commit. The protocol also pins four existing helper/origin
files against their public commits: the Autogrow driver/protocol/README and
the stock-source v2 driver. They are not rewritten. The latter supplies the
already-reviewed dependency-lock attestation, atomic JSON publication and
native module observation only; no stock model generation is invoked.

The three dependency locks have canonical Git LF hashes:

| Target | SHA-256 |
|---|---|
| win-x64 | `53becb18e5c1ea63de4ee8f6eacdd482bcd992827be25439a0a84a89cbc099d5` |
| linux-x64 | `84df56cd98339e8dfec9b6f765312758706476ed1328d5f95f6a06433b9bb719` |
| osx-arm64 | `329ab59df4b1cd1b5dd16eaa2a84d3f0e8e4c807f6bf984246c70290e406dec4` |

Only lock content accepts the explicit CRLF-to-LF substitution; raw bytes,
line-ending counts and canonical hashes are recorded. Even a pure LF/CRLF
change during a run rejects publication. Sources, helpers, protocol, driver
and README are compared as exact raw bytes, without normalization. Before
final publication the collector checks Git HEAD, all public/helper bytes,
the selected raw/canonical lock, all source blobs/AST and the executable again.

`collection.json` marks a started, incomplete collection. Per-case
`*.partial.json` files retain successful repeats without pretending to be a
final reference. Only after the final provenance check is `reference.json`
published atomically without overwrite. Failures leave `incomplete.json`
with an error type and completed-case count. Existing evidence is never
replaced. Interruptions return 130; other failures return 1; completion
returns 0. No final reference means no completed collection.

Native identity is observed after the first image body, once per process:
actual loaded binding/core/OpenMP libraries by distinct instance and basename,
size and SHA-256; unavailable observations remain explicit. Version strings
and lock pins do not attest all installed package bytes. The actual global
ATen capability and requested setting are distinct, and neither establishes
every operator's selected implementation. The build string is hashed, not
published; private paths are omitted. A completed source collection is still
`modelCompatibility=not_assessed`, not product qualification.
