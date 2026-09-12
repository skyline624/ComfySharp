# LoRA arithmetic and immutable model patches

`LoraMath.Apply` implements Float32 LoRA, convolutional LoRA, LoCon mid weights
and DoRA on CPU/CUDA tensors, preserving ambient autograd and returning an owned
tensor without changing its inputs. It follows
[`LoRAAdapter.calculate_weight`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/lora.py)
and [`weight_decompose`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/base.py#L275).
The source's unusual DoRA output-axis norm uses the original weight; the input-axis
norm uses the adjusted weight. Both branches and the Float32 epsilon are preserved.

`LoraWeightPatch` snapshots F32/F16/BF16 factors into immutable F32 storage. Baking
is an inference operation under no-grad; gradients are tested through `LoraMath`
directly. A snapshot can be retained independently and applied on the base weight's
device. Negative and zero strengths and absent alpha retain source arithmetic.
Nonfinite results, shape failures, invalid scalars and cancellation report errors
instead of silently retaining an unpatched weight as the source logging path can do.

`WithLora` on U-Net/CLIP weight sets and their model wrappers constructs a new
model without changing the original. Patched weights own new storage; other
weights use independently owned aliases to existing storage. Disposing the source
model or adapter cannot invalidate the patched model. Failures dispose partial
results and leave the source usable. A configurable byte allowance counts new
resident patched weights; intermediate matrix products and factor transfers are
additional memory, not covered by that allowance. Defaults permit 512 MiB of
new resident weights per call. Applications must still manage overall RAM/VRAM.

## Evidence

The [source collector](../labs/lora-source/README.md) captures 15 inference cases
and actual source `LoraDiff` forward values, gradients, a single SGD parameter
update and the resulting output. The .NET fixture is checked at fixed absolute
and relative bounds of `1e-6`. Eight tests cover these numerical cases, invalid
inputs, atomic failure, byte allowance and native resource release.

Actual reduced U-Net and CLIP graphs produce changed predictions after patching
while the original graphs reproduce their original predictions exactly. CPU
storage addresses establish that unmodified weights share storage and the changed
weight does not. Patched graphs execute after source model/adapter disposal.
These graphs use explicitly synthetic weights; they do not qualify pretrained
adapters or model families. See the [campaign record](qualification/lora-foundations.json).

## Work still required

This is the foundation for lots 6 and 9. The [safe file reader](LORA_FILES.md) now
loads adapters using explicit aliases. The [local LoRA nodes](LORA_NODES.md) now add
plain SD/standalone CLIP name mapping and chained Host/Desktop loading. There is
no integrated `TrainLoraNode` workflow. LoRA export/reload, advanced patch stacking policies,
other adapter types, `reshape_weight`, offsets/functions/hooks, quantized bypass,
optimizer state and end-to-end training remain required. The API currently takes
explicit canonical weight names and does not infer checkpoint architecture.

GPU arithmetic is implemented through native tensor operations but has no new
hardware evidence in this campaign. Pretrained adapter workflows, full model
gradients, Linux CUDA and Apple MPS qualification remain pending. No model was
downloaded or copied, and no capability matrix row is promoted by this milestone.
