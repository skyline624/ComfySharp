# SD component numerical protocol, version 1

This protocol was fixed before producing or comparing SD U-Net, classical image
VAE and diffusion-boundary reference outputs. It does not replace the earlier
CLIP protocol or waive its recorded cross-platform failures.

Reference: ComfyUI backend
[`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a).
Protocol identifier: `sd-components-native210-cpu-f32-v1`.

## Fixed execution and comparison

- Independent source execution on each declared target: Windows x64, Linux x64,
  macOS ARM64. An output generated on Windows is not presumed universal.
- Python 3.12.10, PyTorch 2.10 CPU, F32 contiguous NCHW graph inputs, no gradients,
  one intra-op and one inter-op thread. The default discrete schedule deliberately
  computes its initial tables in F64 before converting them to F32.
- Image-wrapper cases begin with NHWC inputs and preserve the source's resulting
  strides; no extra contiguous conversion may be added to alter dispatch.
- Product: TorchSharp 0.107.0 / libtorch 2.10, CPU/F32. Record native binary hashes,
  runtime build and ISA. Equal version strings do not establish binary identity.
- Exact frozen U-Net `attention_basic` and VAE `normal_attention`. A laboratory
  adapter may explicitly select the unsliced CPU attention route; it must not
  substitute another attention algorithm.
- Finite outputs: `abs(actual - expected) <= 3e-5 + 3e-5 * abs(expected)`.
  Nonfinite class and infinity sign must match; NaN payloads are not compared.
  Shapes, token IDs, schedule indices and structural contracts match exactly.
- Synthetic parameter/input recipes are public and independent of the C# graph.
  The source's own parameter names and shapes define the reference weight schema.
  Checkpoint loading, graph computation and synthetic input generation are tested
  separately. Expected outputs must never come from the implementation under test.

## Qualification levels

Reduced-width behavioral references, stock width/depth with small spatial inputs,
stock image/latent dimensions, real pretrained weights, complete workflows and
hardware qualification are separate evidence levels. None implies the next.
Passing this CPU component protocol does not qualify a model family, a GPU backend
or V1. Missing weights or machines remain missing evidence.

The first implementation tranche covers the plain four-channel SD1/SD2 U-Net,
explicit EPS or V prediction with the default discrete schedule, whole-image text
CFG and the classical four-channel image VAE with legacy quant/post-quant layers.
VAE latents are unscaled. SD15's 0.18215 and SDXL's 0.13025 are applied at the
diffusion boundary. These scales do not imply SDXL U-Net support.

CFG exposes a memory-conservative separate-call policy and an optional compatible
concatenation policy. The latter repeats complete context sequences to their least
common multiple only when the source's ratio limit permits it. It does not claim
to reproduce the source's hardware-dependent memory heuristic. Regions, controls,
hooks, adapters, inpainting, CLIP-H, VAE tiling and GPU execution remain required
future capabilities, not silently supported inputs.

The separate [`labs/sd-source`](../../labs/sd-source) laboratory generates source
evidence. Python is not invoked by the product or distributed .NET tests. Changes
to numerical thresholds require a new reviewed protocol before acceptance; a
failing comparison is not grounds to edit its expected output or tolerance.
