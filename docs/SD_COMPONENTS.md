# SD U-Net, image VAE and diffusion boundaries

The inference library contains actual parameterized CPU/F32 graphs for the plain
four-channel SD1/SD2 U-Net and the classical image VAE. These components are an
intermediate migration step. No pretrained generation workflow or model family is
qualified, and no model node is registered merely because a graph exists.

Sources are frozen at
[`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a).
The product and distributed tests execute C# and native libraries; the separate
source laboratory is described in [`labs/sd-source`](../labs/sd-source).

## Graphs and loading

| Component | Stock configuration | Learned tensors | F32 parameter bytes |
|---|---|---:|---:|
| SD1 U-Net | Base 320, context 768, eight heads, convolutional spatial projections | 686 | 3,438,083,856 |
| SD2 U-Net | Base 320, context 1024, head width 64, linear spatial projections | 686 | 3,463,642,896 |
| Classical VAE | Base 128, RGB three channels, latent four channels, compression eight | 248 | 334,615,452 |

These sizes count parameters only. They exclude activations, temporary tensors,
native libraries, allocator caches and the process. Describing a stock schema does
not establish that a stock-sized forward has been qualified.

`UnetCheckpointLoader` accepts explicit canonical, `model.diffusion_model.` and
plain Diffusers layouts. `ClassicalVaeCheckpointLoader` accepts explicit canonical,
`first_stage_model.` and classical Diffusers layouts, including documented legacy
quant/post-quant aliases. Metadata plans reject missing, ambiguous, unexpected or
incorrectly shaped selected parameters before native initialization. Unrelated
components outside an explicit selected namespace are not loaded. The plan is
bound to the same open safetensors reader used for loading. Stored F32, F16 and
BF16 parameters become CPU/F32; prediction mode and architecture are not guessed.

Weight banks normalize misaligned storage to native allocations aligned on 64
bytes, preserving every parameter bit. A controlled comparison found that buffer
allocation could change CPU linear reductions enough to exceed the fixed profile.
The factory validates the entire input before allocating replacements. On failure,
the caller retains its originals; on success, replaced wrappers are released and
the bank retains the canonical tensors. Already aligned weights are not copied.
This atomic external-tensor factory can temporarily need the input plus a second
bank. The checkpoint loader builds aligned native tensors one at a time and does
not construct a second complete bank.

The U-Net similarly copies a context whose first element is not 64-byte aligned
into a native allocation for the duration of that call. Controlled comparisons
with identical input bits isolated this second allocation-dependent difference.
Borrowed context is never mutated; aligned context is reused. This is CPU runtime
qualification, not a change to model parameters, equations or tolerance.

The U-Net implements the complete selected topology: time embeddings, residual
blocks, source group/layer normalization, noncausal self/cross attention, GEGLU,
skip concatenation and source down/up sampling. Upsampling follows the next skip's
spatial dimensions, including odd rectangular inputs. SD2's context width does not
mean a CLIP-H encoder has been ported.

The VAE implements the encoder, decoder, middle attention and legacy quant/post-
quant convolutions. Encoding returns the posterior mean without random sampling.
`EncodeMoments` exposes the eight raw quant-convolution channels for diagnostics.
`ComfyImageVae` accepts NHWC images, retains the first three channels, center-crops
spatial sizes to multiples of eight and applies the source input transform without
pre-clamping. Decode returns clamped NHWC images. Raw VAE latents remain unscaled.
Input views retain their source strides through the graph. Raw mean encoding
returns a four-channel view of the eight-channel moments storage. The image
wrapper copies that mean into its own NCHW output buffer; image decoding returns
an NHWC view of a normalized NCHW buffer. Callers own the returned wrapper and
must not assume every output is physically contiguous.

## Sampling boundary

`SdDiscreteSampling` implements the default 1,000-step square-root-linear beta
schedule, retaining source F64 construction and logarithm before the F32 tables.
It provides nearest-log-sigma timestep lookup, fractional log interpolation and
percent-to-sigma endpoints. Zero sigma is supported; NaN/negative/infinite sigma
inputs are explicitly rejected. Timesteps clamp at both ends, including infinity.

`SdDenoiser` retains a U-Net, scales its input, converts sigma to timesteps and
converts explicit epsilon or velocity predictions to denoised latents.
`SdSamplingMath` contains the corresponding independently tested math, initial
noise scaling and latent-format boundaries. Apply 0.18215 for SD15 or 0.13025 for
SDXL at the diffusion boundary, never again inside the VAE.

Whole-image CFG operates on denoised predictions. Its scale-one optimization uses
the source `math.isclose` threshold; omitted negative conditioning still passes
through the source CFG arithmetic. The separate-call batch policy is explicit.
Optional compatible concatenation repeats complete text sequences to their least
common multiple, subject to the source's ratio limit; incompatible lengths use
separate calls. Hardware-dependent batch planning is not yet reproduced.

## Ownership and qualification

### Synthetic stock-width diagnostic

The existing RuntimeProbe now accepts `sd --model sd15|sd2`. Its default mode
only describes the 686-parameter U-Net schema and memory estimate, without
initializing libtorch. For example, from an independently cloned repository:

```sh
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- sd --model sd15
```

An actual diagnostic requires `--synthetic --execute`, a positive
`--memory-budget-mib` and `--output` naming a new absolute directory. It uses
stock channel widths, a small `[1,4,16,16]` latent and 77 context tokens. It fills
one native allocation per parameter in chunks of at most 1 MiB, using the named
input recipe, and reuses that bank for one to three forwards. There is no model
download or pretrained checkpoint input in this command. SD2 here is a U-Net
component diagnostic; its text encoder is not supplied.

The plan includes resident weights (3,438,083,856 bytes for SD1.5 or
3,463,642,896 bytes for SD2), inputs and explicit allowances for runtime,
temporaries and headroom. Execution also checks the process working set before
generation and between forwards. These estimates and observations do not bound
every transient peak. A budget is an admission check, not a request to allocate
that amount of memory. Existing native allocator caches count toward it.

Output contains input/output F32 files, parameter hashes without weight payloads,
repeat hashes, timings and process memory observations. Files are created without
overwriting existing evidence. `manifest.json` is written atomically after the
forwards; a failed or cancelled operation can leave partial files without a
completed manifest. Evidence explicitly reports `modelCompatibility=not_assessed`
and `numericalQualification=not_performed`.

The tool's execution test uses a reduced channel configuration with the actual
graph; stock plans are additionally checked in a copy containing only managed
assemblies. Those checks do not prove a stock-width forward or source agreement.
An independently generated source comparison, pretrained weights and a complete
workflow remain separate requirements.

The [first local stock-width execution](qualification/sd-stock-diagnostic-2ee03aa.md)
subsequently completed three forwards each for SD15 and SD2 with small 16×16
latents and synthetic parameters. Parameter/input hashes were independently
recomputed. Source agreement, long-running memory behavior, pretrained weights
and full workflows remain unqualified.

### Tensor lifetime

Weight banks have shared deterministic ownership. A model or wrapper retains an
independent owner; a forward retains the bank throughout execution. Disposing one
owner does not invalidate active consumers. Returned tensors have independent
wrapper ownership and must be disposed by the caller. Borrowed inputs are not
mutated. Per-block scopes release temporaries; cancellation is checked between
operations and cannot interrupt a native kernel already running. These graphs are
currently inference-only and do not claim autograd/training support.

Metadata, contract, lifetime and numerical tests are separate. The prospective
[`sd-components-native210-cpu-f32-v1`](qualification/sd-components-native210-cpu-f32-v1.md)
profile compares independently executed frozen source on each CPU target.
Reduced-width references exercise the full selected topology, but do not validate
stock dimensions, real weights, a complete sampler trajectory, installation or
GPU execution. Each of those remains a separate qualification gate.

Required follow-up includes real model assembly and workflows, CLIP-H, samplers,
regions/masks, inpainting and other channel variants, controls, hooks/adapters,
VAE tiling, device execution/offload, training and all remaining families. The full
[migration plan](MIGRATION.md) remains the release contract.
