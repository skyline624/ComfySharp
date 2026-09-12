# SD latent masks and inpainting

The SD1.5 Float32 path accepts `noise_mask` with Euler, no-churn Heun and
DPM++ 2M, using the existing nine schedulers. `SetLatentNoiseMask` attaches a
reshaped MASK to a shallow copy of the LATENT map, retaining other values.
`VAEEncodeForInpaint` prepares pixels, runs the VAE encoder and returns samples
and a grown noise mask. Both nodes have native editor/compiler definitions.

The implementation follows backend
[`nodes.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L412),
[`KSamplerX0Inpaint`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py#L630),
[`reshape_mask`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/utils.py#L1348)
and default EPS noise scaling at the frozen revision.

## Sampling

White regenerates, black preserves the original latent prediction, and fractional
values blend. Every model evaluation first substitutes the appropriately noised
original latent outside the mask, using ordinary `sigma * noise + latent`, even
when the trajectory was initialized with maximum-denoise scaling. After guidance,
the result is blended with the original scaled latent again. The Heun corrector
uses this wrapper too. DPM++ 2M retains the blended prior prediction.

The mask is bilinearly resized with `align_corners=false` on its source device,
expanded to four channels, repeated cyclically or truncated to the batch, then
transferred to the model. No mask clamping or rounding is introduced in this path.
Inpainting data owns immutable snapshots; operations retain their own handles and
release intermediates on exceptions or cancellation. An empty denoise schedule
keeps its existing bypass behavior without evaluating the mask.

Black-mask preservation applies in latent space. VAE encoding/decoding and its
receptive field can change pixels outside the selected area; there is no final
pixel-compositing guarantee. A white mask matches the unmasked reduced-model
trajectory exactly in the tested cases, including all three samplers.

## Encoder preprocessing

The node runs preprocessing on CPU before transfer to the VAE device. It resizes
the mask to the original IMAGE dimensions, center-crops both to multiples of
eight, replaces rounded masked RGB pixels with 0.5, and leaves alpha untouched.
Growth uses convolution of the rounded mask with an all-one kernel, clamping to
[0,1]. Even kernel sizes produce an extra bottom/right pixel, which is cropped as
upstream does. Zero growth and round-to-even at 0.5 are preserved.

## Verification and limits

The [Windows CUDA campaign](qualification/inpaint.json) completes three native
Desktop → Host workflows at 512 × 512: Euler with VAEEncodeForInpaint in 12.1 s,
Heun with VAEEncode + SetLatentNoiseMask in 18.2 s, and DPM++ 2M with
VAEEncodeForInpaint in 13.2 s. These times exclude initial image upload. All three
load the same shared checkpoint directly, whose SHA256 was rechecked afterward.
The final suites pass 65 Inference, 23 selected Host, 263 Workflow and 115 Desktop
tests. Builds pass without warnings/errors. The initial native attempt exposed a
missing compiler definition, now covered by two regression tests and the three
successful native runs. Visual inspection shows a changed central region in the
procedural input; it does not establish semantic accuracy or photographic quality.

[Source fixtures](../tests/ComfySharp.Inference.Tests/Fixtures/inpaint.reference.json)
contain two reshape cases, nine wrapper evaluations and five preprocessing cases.
They were collected by executing frozen source declarations in a separate
[laboratory](../labs/inpaint-source/README.md), with absolute/relative bounds of
`1e-6` fixed before comparison. The analytical model and capture-only VAE establish
transform contracts only. Tests also use an actual reduced U-Net, check map and
tensor retention, invalid inputs, cancellation, disposal and workflow round trips.

The [workflow](workflows/sd15-inpaint.api.json) uses the alpha output of `LoadImage`:
transparent pixels become the white mask. Import an RGBA image with a transparent
region, select the existing checkpoint and execute. The original procedural
[RGBA test input](../tests/ComfySharp.Inference.Tests/Fixtures/inpaint-input.png)
is 22 KB and contains no model data.

The Host still limits SD1.5 to one Float32 image cropped to 32–512 pixels and
1–100 requested sampling steps. Nine-channel inpaint checkpoints,
`InpaintModelConditioning`, mask hooks/functions, batch-index noise, other model
families and full pretrained numerical comparisons remain pending. Linux CUDA
and Apple MPS are not hardware-qualified. No family or V1 completion is claimed.
