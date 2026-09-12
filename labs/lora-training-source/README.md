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
tests. No complete dataset training node, pretrained numerical parity or
model-family qualification is claimed.
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

## Dataset and random sequence corpus

`batches.py --source CHECKOUT --output NEW.json` executes the frozen dataset helpers,
three `TrainSampler` batch methods, noise preparation and discrete sigma schedule.
Six recipes cover standard, concatenated, oversized, multi-resolution, bucket and
zero datasets, with four batches each. The collector records selected indices,
model-input latents, noise, sigmas and CPU random-state hashes. A recording loss
replaces model evaluation; the guide wrapper reproduces the conditional SD15
latent conversion. This captures the source's scaling asymmetry across modes.
Expected values never come from .NET. Exact local CPU parity for this corpus does
not establish end-to-end pretrained gradient parity or training-node coverage.

## All-target ordinary adapter corpus

`adapters.py --source CHECKOUT --output NEW.json --order-output NEW-ORDER.json`
executes the frozen adapter factories and setup loop on both reduced full plain
U-Net topologies. The native operation adapter supplies `weight_function` flags
for its Linear, Conv2d, GroupNorm and LayerNorm modules. The source traversal order
is also emitted as the product's static names-only resource; it is checked against
the full weight schema. The collector records all 686 targets, 1250 trainable leaves,
their initialization hashes and CPU RNG state. It includes the discarded random
initialization of the two `LoraDiff` Linear layers and the alpha leaf enabled by
the setup call's `requires_grad_(True)`.

Two SGD updates use source `LoraDiff` and `BiasDiff` through `functional_call`,
which replaces external wrapper dispatch. Full predictions/losses and selected
factor, norm, bias and alpha gradients/updates are recorded; every source leaf
must have a finite gradient. This is reduced CPU Float32 evidence, not a complete
node, existing-adapter import, other algorithm or pretrained comparison.

## Training bypass corpus

`bypass.py --source CHECKOUT --output NEW.json` executes frozen `LoraDiff.h`,
`BypassForwardHook._bypass_forward`, `get_module_type_info`, `BiasDiff` and the
reduced SD1/SD2 `_forward`. It attaches six factor adapters with trainable alpha
using the source hook constructor and forward method. Devices are already CPU/F32;
the device-dispatch part of `inject` is not executed. Two norm/bias differences
use `functional_call` to represent weight wrappers, as in the ordinary corpus.

Two SGD steps record outputs, losses and every selected gradient/updated parameter.
Absolute/relative tolerance `3e-5` is fixed before comparison. Fixture SHA-256:
`31e3879b3139c29a9c19d3e24c4e95df65e5006eda6dc758cf3dfc3797fa48dd`.
Source Git-blob and helper hashes are recorded in the fixture. .NET tests verify
and consume this static file without Python. New two-factor Float32 adapters only;
this does not qualify adapter resume, precision/offload or the complete training node.
