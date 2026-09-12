# Mixed adapter source laboratory

This separate Python 3.12.10 / PyTorch 2.10 CPU laboratory is not part of the
application or distributed .NET tests. Reuse the hashed CPU environment described
in the [CLIP laboratory](../clip-source/README.md).

`reference.py` extracts the original `load_lora`, `calculate_weight`, LoRA provider
and base definitions from Git commit `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`.
Five explicit cases capture LoRA plus bias, difference overriding LoRA, normalization
then explicit differences, later alias priority, and zero strength. The values are
exactly representable; expected values come only from those source functions.
Source-file hashes are embedded in `lora-differences.reference.json`.

`verify_export.py` independently decodes the limited F32 safetensors export,
compares every payload hash with the native training report, and passes all keys
to the frozen loader. It checks the 686 target kinds and exact additive application
on zero weights. This is deliberately a restricted decoder, not a claim to use or
reimplement the full safetensors package. It does not compare pretrained model
forward passes or gradients with Python.

The alias laboratory's `--include-vector-weights` option separately captures
full weight aliases, including normalizations, using checkpoint header metadata.
Its existing default and historical rank>=2 fixture remain unchanged.

All output destinations must be new files. No checkpoint payload is copied or
committed. See [mixed adapter qualification](../../docs/qualification/mixed-adapter-files.json)
for collector/fixture hashes, native evidence and limitations.
