# Mixed SD adapter persistence

`LoraTrainingFile.SaveTargetsNew` saves canonical trainable weight and bias targets
to a new safetensors file. Ordinary matrices retain their up/down factors and
learned Float32 alpha; one-dimensional weights use `.diff`, and biases use
`.diff_b`. Explicit double alphas in the older prefix-based API retain F64 encoding.
Exports snapshot CPU storage, enforce the byte allowance, reject nonfinite values,
and publish the completed file without overwriting an existing destination.

The inference loader supports these differences alongside ordinary LoRA factors,
including frozen ComfyUI precedence for `.w_norm`/`.b_norm`, `.diff`/`.diff_b` and
later aliases. Difference shapes must match the target exactly. SD/CLIP aliases
now include existing one-dimensional normalization weights. Unknown formats retain
the existing explicit diagnostic; set/padding and other adapter algorithms are not
claimed by this change.

Frozen snapshots own their storage independently of the source file and training
parameters. Both U-Net and CLIP weight banks can bake additive snapshots with
the existing component strengths and allocation budget. Tests cover disposal,
source precedence, invalid shape/data, export budget, cancellation and no overwrite.
The complete reduced SD test exports all 686 targets/1,250 tensors, mutates and
disposes the training owners, then checks exact reloaded prediction equality.

The real SD1.5 diagnostic also trained all targets for two SGD updates on CPU,
exported a 10,406,328-byte adapter and reloaded it with exact prediction equality.
The checkpoint was read directly from shared storage. The separate source lab
matched every parameter payload hash and accepted 282 LoRA, 109 normalization
weight differences and 295 bias differences with the original loader.

These are miniature synthetic input diagnostics using real checkpoint weights.
They do not establish image-dataset training, style quality, pretrained source
gradient parity, complete training nodes or GPU/platform qualification. The
[qualification record](qualification/mixed-adapter-files.json) distinguishes the
local tests, native run and independent source checks.

To perform an explicit diagnostic after building the probe, use:

```text
dotnet ComfySharp.RuntimeProbe.dll sd-all-adapter-train --checkpoint <existing-sd15.safetensors> --device cpu --report <new-report.json> --adapter <new-adapter.safetensors>
```

No weights are downloaded by the command. Paths must identify existing parent
directories and new output files. The optional CUDA execution additionally needs
the qualified native generator bridge and available GPU memory.
