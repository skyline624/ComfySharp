# SD denoised-latent training objective

`SdDenoiser.DenoiseForTraining` now preserves autograd through input scaling,
the LoRA-patched U-Net and epsilon/velocity reconstruction of the clean latent.
The existing inference entry points still disable gradients. Both paths share
the same mathematical operations; the training entry point uses a differentiable
weight bank and restores the caller's gradient mode on exit.

`SdLoraTrainingObjective.CalculateLoss` implements the plain SD part of frozen
`TrainSampler.fwd_bwd`: add noise scaled by sigma to the diffusion-space latent,
predict the clean latent and compare it with the zero-noise target using the
selected Float32 loss. It returns the unnormalized scalar. Accumulation scaling
and backward belong to `LoraTrainingOptimizer`.

Latents must already be in the model's diffusion space. The helper does not guess
a VAE scale or process a dataset. It borrows all data without changing their
values, gradient flags or existing gradient buffers. Temporary noisy/sigma tensors
enable the input gradients required by the source training path; conditioning is
detached. The returned scalar retains its saved native graph through backward,
even after model wrappers are disposed. Saved activations are additional memory
beyond the patched-weight allowance; checkpointing and offload remain pending.

The [source collector](../labs/lora-training-source/denoising.py) executes the actual
frozen `TrainSampler.fwd_bwd`, `EPS`/`V_PREDICTION`, discrete timestep lookup and
full reduced SD1/SD2 U-Nets. A plain text-conditioning wrapper supplies the model
in place of external guider dispatch. Six cases cover zero, broadcast and per-image
sigmas. Predictions, losses, input/sigma gradients and both LoRA factor gradients
are compared at `3e-5` absolute plus `3e-5` relative, fixed before evaluation.
No previous reference or tolerance is replaced.

The opt-in `sd-lora-train` diagnostic accepts `--objective denoised-latent` with
`--optimizer`, `--loss` and `--accumulation-steps`. It uses supplied real SD1.5 weights
directly, miniature synthetic diffusion latents, explicit sigmas and independent
noise seeds; it exports a small adapter and verifies snapshot/reload equality.
Default `--objective raw` preserves the earlier raw-prediction diagnostic.

The [dataset and batch loop](TRAINING_DATASETS.md) now supplies source selection,
noise sequencing and the three dataset modes for the plain Float32 profile.
Conditioning regions, all adapter targets and initializers, mixed precision, checkpointing,
offload, other architectures and platform qualification remain required. The
[campaign record](qualification/lora-denoising.json) separates source primitive
comparisons from the pretrained CUDA diagnostic and their limitations.

The subsequent [Linux/macOS CI failures](qualification/lora-denoising-ci-eec43c8.md)
remain open with the original fixtures and tolerances unchanged.
