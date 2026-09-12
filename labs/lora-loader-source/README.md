# Frozen LoRA loader references

The collector extracts actual `load_lora` and `LoRAAdapter` declarations and their
helpers from Git at `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. It supplies only
the LoRA adapter provider because these files contain LoRA factors, not LoHa,
LoKr, OFT or other providers. The device-cast shim is `Tensor.to` for CPU cases.

Run with PyTorch 2.10.0+cpu in a separate laboratory, `--source` for the Git
checkout and `--output` for a new JSON file. Existing output is never overwritten.
Source hashes, tensors, alias order, claimed/unclaimed keys and applied results
are recorded. The seven formats exercise F32, F16 and BF16 factors and F64 alpha;
two additional cases cover regular-vs-B priority and later-alias replacement.

Absolute/relative `1e-6` bounds were fixed before comparison. This corpus supplies
aliases explicitly and does not qualify architecture key mapping, other adapter
providers or pretrained workflows. Product and distributed tests never run Python
or this collector. No checkpoint or model download is performed.
