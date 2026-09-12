# Frozen model alias references

The collector executes the actual `model_lora_keys_unet`, `model_lora_keys_clip`
and `unet_to_diffusers` functions at backend commit
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. It extracts their declarations and
constant maps with AST, without importing ComfyUI or creating native models.
The model facade supplies state-dictionary keys and the plain SD configuration;
unrelated architecture type checks are false. It never reads tensor payloads.

Run the separate laboratory script with `--source` (frozen Git checkout),
`--checkpoint` (SD1.5 safetensors), and `--output` (a new JSON path).
The first cases use the real checkpoint header. Two explicit 33-layer CLIP
metadata cases exercise projection names and the source's 32-layer legacy alias
cutoff for standalone L and G encoders. Source file hashes are recorded.

Comparison is exact for aliases and their order **within each target**. Source
iteration over unrelated weights includes Python sets and is not a stable global
ordering contract. Rank-one and absent weights are excluded because the current
factor implementation cannot apply LoRA to them. Composite encoders and other
model architectures are not qualified. This laboratory is never invoked by the
product or distributed .NET tests.
