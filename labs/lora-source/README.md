# LoRA reference laboratory

The development-only collector executes the actual `LoRAAdapter.calculate_weight`
and `LoraDiff` declarations from backend
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, along with their base classes and
helpers. Git/AST extracts declarations without importing the ComfyUI runtime.
The sole device-cast shim is `Tensor.to(device, dtype)` for these dense CPU cases.

Run in a separate PyTorch `2.10.0+cpu` laboratory with `--source` pointing to a Git
checkout containing that commit and `--output` naming a new JSON file. Existing
output is never overwritten. Source blob hashes are recorded. No C# result is
used as an oracle; no checkpoint or package download is performed.

The 15 inference cases cover linear LoRA, convolutional LoRA, LoCon mid weights,
both DoRA normalization axes, positive/negative/zero strength and optional alpha.
Source DoRA intentionally uses the original weight for output-axis normalization
and the adjusted weight for input-axis normalization. This asymmetry is retained.

The training case calls actual source `LoraDiff`, computes mean-square loss,
compares gradients of both low-rank factors, performs a PyTorch SGD step at 0.05,
and captures updated factors and output. The .NET comparison applies that same
explicit SGD update to test gradients; it does not implement a training workflow
or announce a production optimizer. Absolute/relative bounds of `1e-6` were fixed
before comparison. Product and distributed tests never run this Python collector.
