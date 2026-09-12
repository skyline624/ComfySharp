# Local LoRA nodes

`LoraLoader` and `LoraLoaderModelOnly` are connected to the Host catalogue and
native workflow compiler. They read the shared models directory's `loras`
category without copying or downloading files. Nested safetensors names are
supported; escaping names and symbolic links/junctions are rejected.

The implemented path applies LoRA, LoCon and DoRA to the plain SD U-Net and a
standalone CLIP encoder in Float32 on the graph's existing CPU/CUDA device.
The output order is MODEL/CLIP for `LoraLoader`, MODEL for the model-only node.
Strengths are independent and accept negative values in the source schema's
[-100,100] range. Zero strengths return retained original inputs without reading
the file. Nodes can be chained; each patch creates independent graph handles
and shares unchanged weight storage. Both outputs are prepared before publication.

Model names follow the frozen
[U-Net and CLIP key maps](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/lora.py).
The alias comparison checks 1,692 U-Net and 290 CLIP-L aliases against the source
using the SD1.5 checkpoint header. Standalone L/G metadata cases cover projection
names and the 32-layer legacy naming limit. SD2 uses the same plain topology names
with its own linear-projection and context shapes; this does not qualify SD2 inference.
Concatenating independent L/G maps does not implement SDXL's composite encoder.

Unclaimed keys and shadowed aliases are reported in `comfysharp_lora` execution
metadata. Model-only loading can therefore report unused CLIP factors. A file
with no matching factors fails explicitly. This is stricter than source warning
behavior. The [safe reader](LORA_FILES.md) documents supported suffixes, shape
validation and the 512 MiB resident-factor allowance; baked weights have a separate
512 MiB allowance. Temporary conversion and computation allocations are additional.

The catalogue remains partial. Other adapter types, rank-one patches, offsets,
`reshape_weight`, quantized execution, hooks and composite encoders remain required.
No family is declared compatible and integrated training/export is not complete.
The [qualification record](qualification/lora-nodes.json) distinguishes exact
metadata comparisons, reduced graph tests and pretrained workflow execution.
