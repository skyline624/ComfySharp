# Frozen LoRA training references

Run `reference.py --source CHECKOUT --output NEW.json` in the separate PyTorch
2.10.0 CPU laboratory. The collector reuses the hash-verified SD source adapter
from `labs/sd-source`, extracting exact Git blobs into a temporary source snapshot
so Windows checkout line endings cannot change provenance checks. It runs the
actual frozen U-Net `_forward` with basic attention and actual `LoraDiff` weight
construction. `torch.func.functional_call` supplies differentiable replacements
without modifying frozen base parameters. External wrapper dispatch alone is
bypassed. No .NET output supplies expected values.

Two full plain topologies use reduced widths: SD1 convolutional projection and
SD2 linear projection. First/final convolution factors are explicit shared input
data; two SGD updates compare predictions, MSE, both gradients and updated factors.
The new training profile fixes `3e-5` absolute plus `3e-5` relative before testing.
It does not change existing inference or LoRA arithmetic profiles.

`verify_export.py --source CHECKOUT --adapter FILE --report DOTNET.json --output NEW.json`
independently decodes the .NET export's restricted F32/F64 byte format, checks
factor hashes against the native training report and executes frozen `load_lora`
and `LoRAAdapter` on both targets. This small decoder is explicitly not the Python
safetensors package. It checks interoperability of this export, not arbitrary files.

The laboratory neither downloads models nor runs in the product/distributed .NET
tests. No complete dataset training node, optimizer suite, pretrained numerical
parity or model-family qualification is claimed.
# Optimizer and loss corpus

`optimizers.py --source SOURCE --output NEW.json` executes only the two frozen
optimizer/loss factories from `comfy_extras/nodes_train.py` with PyTorch 2.10 CPU.
It compares all 16 combinations across three updates with two microbatches each,
including a parameter that becomes inactive for the middle update and reactivates.
Inputs use a small analytic LoRA linear layer. Four boundary-gradient cases cover
residuals zero, plus/minus one and both tails. This is a primitive comparison, not
a training-node or pretrained-model reference. Bounds are fixed before comparison.

## Denoised-latent objective corpus

`denoising.py --source CHECKOUT --output NEW.json` executes the frozen `TrainSampler.fwd_bwd`, discrete/EPS/V boundaries and full reduced U-Net with LoRA. A plain text-conditioning wrapper replaces external guider dispatch. Six cases compare predictions/losses and noisy-input, sigma and factor gradients for zero, shared and per-image sigmas. CPU Float32 autocast disables itself (its warning is suppressed); no mixed-precision path is emulated. Inputs are fixed diffusion latents/noise/sigmas, not a dataset sampler. The tolerance is fixed before evaluation and prior fixtures remain unchanged.
