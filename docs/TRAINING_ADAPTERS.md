# Ordinary SD adapters, biases and trainable alpha

`SdTrainableAdapterSet` initializes all **686 canonical weight/bias targets** in
the frozen plain SD1/SD2 U-Net. The names-only embedded resource preserves original
module traversal order and must exactly cover the runtime weight schema. Shape
changes still come from that schema. This produces **1250 Float32 trainable leaves**:
LoRA matrices and alpha for multidimensional weights, plus additive vectors for
one-dimensional normalization weights and module biases.

Initialization follows the frozen source: up uses Kaiming uniform with `a=sqrt(5)`,
down is zero, alpha starts at one, and additive vectors are zero. The subsequent
`requires_grad_(True)` in source setup also enables alpha. The two temporary Linear
layers constructed by `LoraDiff` consume CPU random numbers before their initial
weights are replaced; that consumption is reproduced with a private generator.
CUDA factor initialization uses a private generator on the selected device, while
the temporary Linear initialization remains on CPU. Global generators are untouched.
Source RNG parity is currently demonstrated on Windows CPU only.
CUDA generator construction requires the small [native bridge](../native/ComfySharp.NativeGenerator/README.md)
which supplements TorchSharp's CPU-only native constructor. Missing bridge support
is diagnosed explicitly and does not select CPU random initialization.

`TrainableWeightPatch` gives the existing optimizer and differentiable U-Net a
common retained owner for ordinary LoRA and `TrainableDifferencePatch`. Constructors
snapshot supplied leaves. The existing explicit two-factor LoRA constructor keeps
fixed alpha by default; `trainAlpha: true` enables the additional scalar leaf.
The complete factory enables it. Float32 differences are added without mutating
the frozen base. Cancellation, optimizer cleanup and retained-owner lifetime rules
also apply to norm/bias/alpha parameters.

The factory's default 512 MiB budget covers persistent adapter leaves only.
Temporary initialization tensors, optimizer moments, patched full weights and
saved activations require additional memory. `SdLoraTrainingOptions.MaxPatchedWeightBytes`
allows a caller to admit the full differentiable weight bank explicitly; its
default remains 512 MiB. This implementation does not yet provide checkpointing,
offload or the bypass path needed for large/quantized training.

The [source collector](../labs/lora-training-source/adapters.py) executes original
setup/factory declarations and full reduced U-Nets. It records exact hashes for
every initialized parameter and the CPU RNG state, then compares predictions,
losses and selected factor/norm/bias/alpha gradients and updates across two SGD
steps. Every leaf must receive a finite gradient; the unmodified base remains
frozen. It replaces external wrapper dispatch with `functional_call` and supplies
the wrapper-capability flag in the native operation adapter. It does not compare
every gradient element for every target or run a complete training node.

`sd-all-adapter-train --checkpoint FILE --report NEW.json --device cpu|cuda:0`
performs an explicit real-SD1.5 all-target gradient diagnostic. Inputs are miniature
synthetic raw-prediction tensors. Only a metadata report is written: the shared
checkpoint is read directly, with no checkpoint or adapter copy. The
[qualification record](qualification/training-adapters.json) states its measured
results and limitations.

Full mixed-adapter persistence/reload, existing adapters, other training algorithms,
real image datasets, conditioning dispatch, precision modes and the complete
`TrainLoraNode`/`LORA_MODEL` integration remain required. This does not qualify a
model family or the training node. The separate [dataset noise discrepancies on
Linux/macOS](qualification/training-batches-ci-d1626ad.md) remain open, as do the
previous denoising-reference failures; comparison thresholds are unchanged.
