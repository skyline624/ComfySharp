# Frozen LoRA bypass oracle

Lab-only Python collector; not a product or distributed .NET test dependency.
Use a separate environment with PyTorch `2.10.0+cpu` and a local checkout containing
ComfyUI commit `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`:

```text
python collect.py --source <ComfyUI-checkout> --output <new-reference.json>
```

The collector reads frozen Git blobs, extracts the exact `LoRAAdapter.h` and
`WeightAdapterBase.g/bypass_forward` methods, and executes six explicit synthetic
linear/Conv2d cases. It records outputs and gradients for activations, base output
and factors. It does not import ComfyUI or read model weights. Alpha is a scalar,
as in inference loading; trainable alpha gradients are outside this corpus.

The fixture records SHA-256 hashes of both source files. ComfyUI contributor
copyright and GPLv3-or-later terms apply to the adapted source; see the repository
LICENSE and THIRD_PARTY_NOTICES. Tolerances were fixed at absolute/relative `8e-6`
before the C# comparison. The checked-in fixture SHA-256 is
`7cd58396213b27d4e6ed6963141ab19d4677906ca08e45b2e4c0ce4901aa8519`.
Tests read that fixture directly and verify its hash. This is a primitive oracle,
not a full pretrained-model or training-workflow qualification.
