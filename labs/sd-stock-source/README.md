# Synthetic stock U-Net source laboratory

This separate, opt-in laboratory executes the frozen upstream U-Net on CPU/F32.
It does not import the C# implementation, its weight schema or its output files,
and never writes acceptance fixtures. It does not load or download checkpoints.
Python and these laboratory dependencies remain outside the application and its
distributed .NET tests.

The prospective protocol is `sd-unet-stock-small-native210-cpu-f32-v1`.
[`cases-v1.json`](cases-v1.json) fixes the exact two RuntimeProbe-compatible cases:
SD15 or SD2 stock width, latent `[1,4,16,16]`, context length 77, timestep 0.125,
and three forwards sharing one bank. The source file checks the case document's
SHA-256 before collection. Finite comparison remains
`abs(actual-expected) <= 3e-5 + 3e-5*abs(expected)`. Nonfinite class and infinity
sign must match. This collector does not perform that product comparison and
does not declare a model or backend qualified.

Stock collection requires a reviewed protocol and explicit resource admission.
The initial implementation has only been exercised with the bounded self-test;
do not interpret its existence or `status=ok` as a completed stock qualification.

## Environment and frozen sources

Use a separate Python 3.12.10 environment with the unchanged hashed CPU closure
in `../clip-source/requirements-<runtime>.txt`, for `win-x64`, `linux-x64`, or
`osx-arm64`. Install with `pip --isolated --require-hashes --only-binary=:all:`;
the lock contains direct hashed wheel URLs and disables index resolution.
The collector checks the lock-file hash and the declared Torch 2.10.0 CPU,
NumPy 2.2.6 and einops 0.8.1 versions. Runtime-loaded native file fingerprints
are recorded separately; a version string does not establish binary identity.

Prepare an external snapshot containing these canonical Git blobs from
[ComfyUI `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a):

| File | SHA-256 |
|---|---|
| `comfy/ldm/modules/diffusionmodules/util.py` | `fb58652a35521fc23bdcb75d91adace8e4cc79e2d5b13af1617a38d0c0f7142e` |
| `comfy/ldm/modules/attention.py` | `9cafaafaf93ff53e8cbefb5e4a204014019985df2da8c1996bf40f1235fc2960` |
| `comfy/ldm/modules/diffusionmodules/openaimodel.py` | `9d27fb036cab8a262ef3d866a643f7fdc40994022616f1b8be14b7d919f57f96` |

The collector reuses only the individually hash-verified existing
`../sd-source/common.py` and `../sd-source/unet.py` helpers. Those helpers verify
the source bytes before executing the selected AST declarations, and record
the exact symbol and AST hashes. Old source labs and fixtures are unchanged.
The graph calls unchanged `UNetModel._forward` with explicitly selected frozen
`attention_basic`. Unsupported temporal/control/extensions remain excluded.

## Commands

The default is a metadata-only plan; it imports neither Torch nor NumPy and does
not allocate a model. Stock resident bytes are prospective constants from the
case document; an explicit execution independently checks the source module's
own schema before allocating CPU parameters.

```text
python -I -B labs/sd-stock-source/reference.py --model sd15 --inspect
python -I -B labs/sd-stock-source/reference.py --model sd2 --memory-budget-mib 6144
```

The ordinary bounded self-test checks input arithmetic and chunk boundaries,
including uint32 wrap, against the unchanged source recipe and independent scalar
arithmetic. It then runs both full-topology **reduced** source graphs with three
forwards, checking all 686 parameters in each graph against the old source input
recipe. These temporary comparisons occur only for reduced self-tests.

```text
python -I -B labs/sd-stock-source/reference.py --self-test --source-directory <absolute-snapshot> --output <new-absolute-external-self-test-directory>
```

Only after review and host admission, explicit stock collection is available:

```text
python -I -B labs/sd-stock-source/reference.py --model sd15 --synthetic --execute --source-directory <absolute-snapshot> --memory-budget-mib 6144 --chunk-elements 262144 --repeat 3 --output <new-absolute-external-directory>
```

Replace `sd15` with `sd2` in a **new process** and a new directory. Do not run
source/product or two stock banks concurrently on a constrained host. The
source profile fixes 1 MiB chunks and three repeats; other values are rejected.
Output must not exist and must be outside both the repository and source
snapshot. On CI, use `RUNNER_TEMP` for the venv, snapshot and output.

Run standard collection with `ATEN_CPU_CAPABILITY` unset. A deliberately selected
`default`, `avx2` or `avx512` override is recorded and defines a separate diagnostic
cell; it cannot replace an auto failure. The actual global ATen capability is
queried from PyTorch. This is not evidence of every operator or BLAS/oneDNN kernel
choice. Thread counts are fixed at one intra-op and one inter-op, with no gradients.

## Storage and evidence

The source constructs one reset-disabled module on `meta`, derives its real
parameter names/shapes, then uses `to_empty(cpu)` once. Each contiguous native
F32 parameter must be 64-byte aligned. A zero-copy NumPy view is filled in place
with the unchanged `sha256-name-lcg-high16-power2-v1` recipe. Two reusable 1 MiB
integer scratch arrays support each destination chunk of at most 1 MiB. No
second parameter bank or full-sized parameter temporary is retained. Hashes
are taken from the actual destination slices after writing, not from scratch.

SD15 has 3,438,083,856 resident weight bytes and SD2 3,463,642,896, each with 686
parameters. The largest parameter is 117,964,800 bytes. The requested budget is
checked against planning allowances before native import, against observed
process memory plus remaining weights/allowances before CPU allocation, and
against observed process memory during generation and between forwards. These
checks are neither a free-RAM measurement nor an OOM/peak guarantee; separate
host/container admission and timeout supervision remain necessary. The collector
does not procure resources or select a larger runner automatically.

`collection.json` records the prospective protocol, case/helper/script/lock hashes
and that collection started. A completed `manifest.json` records source AST and
configuration provenance, parameter hashes/layouts, three input payloads, one
complete output payload, repeat hashes, stage wall/CPU times and process memory.
Payloads use little-endian F32 with byte lengths, shapes, strides and SHA-256.
No parameter payload files or giant JSON value arrays are written.

Runtime identity is observed after the first forward. Loaded native libraries
are fingerprinted once per process, with basenames and distinct instance entries;
private paths and exception messages are omitted. Unavailable/partial observations
stay explicit. Exact library bytes can differ between the source wheel and the
product NuGet package at the same declared version. No equality is presumed.

Manifests are written with exclusive temporary creation and atomic, non-overwriting
publication. The collector rehashes its script, case definition, both reused
helpers, selected dependency lock and all three source blobs after computation
and before publishing a final manifest. The before/after provenance fingerprints
are recorded. A changed input prevents publication and emits `incomplete.json`;
the payloads remain partial evidence. A failed run can leave inputs, partial payloads or a temporary file;
these are not successful references. Return codes are 0 for completed operations,
3 for an insufficient process budget, 130 for interruption and 1 for other
failures. Repeated output drift writes an explicit `non_repeatable` manifest and
returns nonzero. A successful source run remains `modelCompatibility=not_assessed`.

There is no workflow or product comparator in this sub-lot. Future same-host
collection must run source and product sequentially on the same allocated machine,
verify configuration/parameter/input identity and all output payload bytes, then
apply the frozen tolerance. Stock small-spatial synthetic evidence does not
qualify full spatial sizes, batch-two CFG, pretrained weights, model families,
image quality or GPUs.
