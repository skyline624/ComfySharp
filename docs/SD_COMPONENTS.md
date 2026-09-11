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
