# Safe LoRA files

`LoraFileLoader.Inspect` builds a metadata-only plan from an open `SafeTensorFile`
and an ordered list of explicit aliases, components, canonical target names and
shapes. `Load` accepts only the same reader, copies selected factors into immutable
adapter snapshots and returns a `LoraAdapterSet`. No code from the file is executed.
The adapter can outlive the file and apply to a matching U-Net, CLIP or individual
weight. Target shapes are checked again before application.

## Selection contracts

The seven supported suffix pairs follow frozen
[`LoRAAdapter.load`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/weight_adapter/lora.py)
priority: regular `lora_up/down.weight`, `_lora.up/down.weight`, `lora_B/A.weight`,
`lora.up/down.weight`, bare `lora_B/A`, `lora_linear_layer.up/down.weight`, then
`lora_B/A.default.weight`. Header order does not change this selection. A selected
up factor without its down factor is an error, with no fallback to another pair.

As in frozen
[`load_lora`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/lora.py#L40),
later aliases replace earlier bindings for the same canonical target. The plan
reports shadowed prefixes. It validates even shadowed candidates conservatively,
but only materializes final selected factors. The supplied alias order is explicit;
this API does not yet derive the architecture's source key map automatically.

Regular LoRA can carry `lora_mid.weight`; alpha and DoRA scale apply to all seven
forms. Factor storage accepts F32/F16/BF16, converted to F32. Alpha requires one
scalar supported by the safetensors reader, including F64 and I64; nonfinite alpha is rejected on loading.
Metadata checks catch incompatible rank, reshape, broadcast and byte allowance
before factor materialization. `reshape_weight` is explicitly unsupported.

Unknown or lower-priority tensor keys cause a diagnostic by default. Callers may
explicitly allow them and must surface `UnclaimedKeys`; an empty match always
fails. Norm/diff/set patches and other adapter providers remain unsupported.
This is stricter than ComfyUI's warning-and-continue behavior. Declared components
can have independent strengths even when that file has no match for one component.

Loading publishes only a complete owned adapter set. Cancellation or errors after
an earlier factor has been loaded release all snapshots created by that attempt.
The default 512 MiB allowance counts resident F32 factors. Reader buffers,
conversion temporaries and later baked model weights are additional memory.

## Evidence and remaining work

The [source collector](../labs/lora-loader-source/README.md) executes actual frozen
LoRA-only loading for seven suffix forms and two precedence scenarios. Nine cases
compare selected/unclaimed keys and numerical application at fixed absolute and
relative `1e-6` bounds. Fifteen existing LoRA/LoCon/DoRA source arithmetic cases
also pass through actual safetensors files and this reader. Native lifetime tests
cover failed partial loads, source-file disposal and a patched reduced U-Net.
See the [campaign record](qualification/lora-files.json).

The 15 local shared adapter headers were examined without reading tensor payloads
or copying files. None contained the searched SD-style U-Net prefixes. This is
candidate screening, not complete architecture identification or qualification.
No external adapter was downloaded and no pretrained adapter workflow is claimed.

The [local LoRA nodes](LORA_NODES.md) now supply plain SD/standalone CLIP aliases,
Host/Desktop loading and the shared file catalogue. Composite encoders, other
architectures, trained-adapter qualification, export/reload and integrated training
remain required. No capability matrix row is promoted by this reader's synthetic-file
tests. Cross-platform/GPU qualification and numerical CI failures remain open.
