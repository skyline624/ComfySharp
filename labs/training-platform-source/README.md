# Training source references on each CPU platform

The original training corpora were collected with PyTorch 2.10 CPU on Windows.
Linux/macOS CI exposed differences in their exact noise and initial-factor hashes,
and in some denoising gradients. This laboratory collects the **same existing
cases** on Windows x64, Linux x64 and macOS ARM64 before considering platform
fixtures. It does not use .NET outputs to construct expected values.

`protocol.json` pins the collectors, helper code, unchanged hashed environment
locks and upstream Git blobs. Preflight runs with Python's site imports disabled,
before loading native libraries. The workflow then installs the platform's
existing CPU wheel lock and runs each collector in its own process, with one
intra-op and one inter-op thread. PyTorch's macOS wheel uses the version string
`2.10.0`; Windows/Linux CPU wheels use `2.10.0+cpu`.

```text
python -I -S -B labs/training-platform-source/collect.py --source FROZEN_CHECKOUT --validate-only
python -I -B labs/training-platform-source/collect.py --source FROZEN_CHECKOUT --output NEW_DIRECTORY
```

The output contains batch, denoising and all-target adapter references, the observed
parameter traversal order, and a manifest with source/input/artifact hashes,
platform/runtime information and CI identity. Existing tracked fixtures are never
rewritten. The Windows outputs must first reproduce the existing Windows corpora;
other platform outputs require provenance review and comparison against .NET on
the matching platform. A successful collection alone is not a passing product test.

Exact batch/initialization checks and the original `3e-5` absolute plus relative
gradient/prediction bounds remain unchanged. Recipes, source revision, model
dimensions and parameter-update steps remain unchanged. Any disagreement between
same-platform ComfyUI and .NET remains a failure to investigate.

This is a separate source laboratory. Python and its packages are not application
dependencies and are not needed by the distributed .NET tests. No pretrained
weights are used or downloaded. The complete training node, real image datasets,
GPU numerical parity and model-family qualification remain separate work.
