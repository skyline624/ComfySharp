# ImageBatch source laboratory

This laboratory fixes its protocol before the first source collection.
Publish these exact files before execution; the driver verifies their Git
identity before importing Torch and again before publishing the reference.
The preflight only inspects and compiles selected source declarations.
No collection runs automatically and no dependency is installed here.

## Source and profile

The frozen backend is `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`.
Only **two canonical Git blobs and two exact AST declarations** are selected:

- [ImageBatch, nodes.py:1955](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L1955),
  blob SHA `f7b6a909bf9a47296b974b730e5db0ec2137a3c6a64dda045970951d7382d4c7`.
- [common_upscale, comfy/utils.py:1105](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/utils.py#L1105),
  blob SHA `f3ee59baf72f6910b7662224854e372c713ba3ad01c89739ba092e5ee22460e6`.

Their complete bodies are unchanged. The real Torch module is supplied after
admission; `comfy.utils` routes to the module containing that exact function.
No placeholder replaces padding, cropping, interpolation or concatenation.
`bislerp` and `lanczos` are unbound because this real ImageBatch body always
requests `bilinear`; no fake implementation of either branch is installed.
The higher-rank path of common_upscale is preserved but outside the rank-four
IMAGE profile. V3 is unnecessary for this legacy node: real INPUT_TYPES,
RETURN_TYPES, FUNCTION, CATEGORY, DEPRECATED and SEARCH_ALIASES are captured,
not a fabricated complete object_info response.

`image-batch-cpu-f32-v1` admits dense CPU/F32 NHWC RGB/RGBA tensors, positive
dimensions, finite storage with magnitude at most 32, at most 2,048 stored
elements per input and at most 4,096 output elements. Inputs are separate native
allocations with exact embedded storage bytes, shape, strides and offsets.
No model, random generator, C# output or product weight schema is read.

The comparison rule is prospective and fixed before the first collection:

- **Four no-resize cases:** exact logical little-endian F32 bytes, including
  signed zero. This includes channel promotion and concatenation.
- **Fourteen bilinear cases:** finite outputs and
  `abs(actual-source) <= 1e-6 + 1e-6*abs(source)`; shape/dtype/layout remain exact.
  The relative term is about 8.4 Float32 epsilons. It is an engineering margin
  for a small bilinear stencil and bounded small-image coordinates, not a
  universal floating-point error proof or evidence that any platform passes.
  It is much smaller than a wrong crop, wrong alpha rule or align-corners
  change in these asymmetric inputs. The collector does not apply it because
  it reads no product outputs. No threshold may be relaxed after collection.
- **One source-error case:** classify the actual body failure, without numeric
  output or a promise of equivalent .NET/Python exception wording.

This new resize profile does not change the previously published bit-exact
profile of EmptyImage/Invert/Repeat/FromBatch.

## Nineteen cases and observations

The protocol enumerates all inputs and their SHA-256, not expected outputs.

| Group | Fixed cases |
|---|---|
| Exact: 4 | RGB batches 1+2; RGBA strided views; pad first RGB; pad second RGB |
| Bilinear: 14 | Half-pixel upsample; downsample without antialias; wide/tall crops; margins 0.5 and 1.5 on both axes; two asymmetric view layouts; alpha downsample; padding before resize in both orders; distinct batches with nonzero offsets |
| Error: 1 | Positive 2×2 second image cropped to zero height for a 1×100 first image; source RuntimeError admitted only at the body boundary |

Three runs per case create fresh modules and both input storages. The sequence
is off/on/off, with **57 direct ImageBatch calls** prospectively: 54 returned
records and 3 classified failures. It does not exercise mapper or HTTP admission.
The middle run observes the actual common_upscale code object using
`sys.setprofile` at return, including the exceptional return of the crop-empty
case. It records actual x/y, ratios, target dimensions, crop mode and layouts
of samples/cropped/result; it does not reconstruct crop offsets independently.
Only metadata is captured in that callback, including zero-size crop metadata.
Its errors are retained separately and cannot be accepted as the expected
source RuntimeError. An external profiler is rejected.

The three neutral records, after excluding only the middle observation field,
must have identical canonical JSON hashes, including all F32 payloads and
layouts. All three records are retained, so that neutrality can be recomputed.
The source error retains input-before/after but no fabricated output.

After each successful body, a separate copy observation mutates the first
declared scalar of image1 and then image2 by F32+0.5. After each step it captures
both actual inputs and the unchanged result. There are two independently
created input storages; each mutation must change the intended input while
the output record stays identical. This is a controlled post-body property,
not part of the original algorithm or a second call. It does not claim to
prove all arbitrary aliases. Strided capture copies logical contents only
after recording the original layout; the capture copy never enters the body.
In some resize cases the declared image2 scalar is outside the retained crop,
so that mutation alone does not establish independence of the retained pixels.
The final source `torch.cat` and separate product tests that mutate whole inputs
provide distinct evidence; they must not be attributed to this scalar observation.
For the classified error, the observed middle run must additionally have raised,
returned exceptionally from common_upscale and captured a zero spatial crop
dimension. An arbitrary RuntimeError without that actual crop is not accepted.

## Reuse, environment and commands

The seven helper/origin files are pinned in the protocol against their public
Git commits: the three IMAGE829a files plus its existing four helper/origin
files. The verified IMAGE driver supplies native input storage and logical
tensor captures. Its verified Autogrow helper supplies AST selection and
compilation only. Its stock-v2 helper supplies lock attestation, atomic JSON
publication and native library observation only; no stock function is called.
Thus neither 59 V3 AST declarations nor a copied 400-line collector are needed.

Reuse ROOT's existing isolated CPython 3.12.10 CPU210 environment, or the same
already-published hashed CPU closure for the actual platform: Torch 2.10.0+cpu
Windows/Linux or 2.10.0 macOS ARM64, NumPy 2.2.6, einops 0.8.1. No environment
is installed or created here. The unchanged locks remain in
`labs/clip-source/requirements-{win-x64,linux-x64,osx-arm64}.txt` with their
published direct wheel URLs, hashes and no-index policy. Python remains
outside the application and its .NET test deployment.

Commands (execution requires the exact published files):

```text
python -I -B -S labs/image-batch-source/reference.py --source <backend-git> --target win-x64
python -I -B labs/image-batch-source/reference.py --source <backend-git> --target win-x64 --execute --output <new-absolute-external-directory>
```

`-S` is only for preflight; execution needs installed Torch visible. The target
must match the host. ATEN_CPU_CAPABILITY must be unset, intra/inter-op threads
are 1/1, no-grad is enabled. Output must be new and outside both repositories.
The reference laboratory is separate from the product and its .NET tests.
No model or additional dependency is downloaded by this collector.

## Integrity, privacy and failure

The driver pins the protocol bytes and checks source blob/segment/AST hashes.
All helper bytes and public origins are verified before helper imports. Before
native import, the public location and each of the three public files must
match current Git HEAD. At completion the full snapshot is recomputed and
compared: HEAD, these files, seven helpers, source AST, executable and selected
dependency lock. Only the lock explicitly accepts CRLF→LF normalization for
canonical content; its raw bytes/hash must remain unchanged during collection.

`collection.json` says started, per-case `*.partial.json` files remain partial,
and `reference.json` is atomically published without overwrite only after the
post-computation integrity check. Interrupted or failing collection leaves
`incomplete.json`; exit 130 indicates interruption, exit 1 failure and exit 0
completion. The expected crop-empty source case is recorded normally without
turning an unexpected error into acceptance.

Native libraries are observed once after the first body, by distinct loaded
instance, basename, size and SHA. The build string is hashed; private paths
are not published. The requested ATen setting is distinct from its queried
global capability; neither proves per-operator dispatch or binary equivalence
with NuGet. A completed source collection remains unqualified product evidence
(`modelCompatibility=not_assessed`); no image export or whole workflow is claimed.
