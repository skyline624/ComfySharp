# Isolated CLIP source laboratory

This directory collects **diagnostic upstream executions**. It is independent of the ComfySharp application, solution, distributed .NET tests and model loaders. None of those components starts Python or imports this laboratory. Laboratory dependencies are installed in a temporary environment by a separate workflow; they are not product dependencies.

The [existing investigation](../../docs/qualification/clip-cpu-investigation-5810a4f.md) found identical synthetic weights and embeddings, small differences beginning at LayerNorm and stock-output failures against the original Windows reference. An upstream-only dispatch experiment also failed that bound. The following protocol is recorded before this laboratory's target-platform results. **It introduces no replacement fixture, new tolerance, accepted backend profile or model-compatibility claim.**

## Question and controlled inputs

Compare independently executed frozen ComfyUI source under PyTorch 2.10 CPU with the existing TorchSharp 0.107/libtorch 2.10 CPU traces on Windows x64, Ubuntu 24.04 x64 and macOS 14 ARM64. This controls the declared native release more closely than the original Windows PyTorch 2.13 reference. Equal version numbers do not establish equal native binaries, compilers, BLAS builds or dispatch. Record their identities and do not call this a matched binary experiment unless hashes establish that fact.

Keep the backend commit `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, exact source ASTs, stock L/G configurations, SHA-derived synthetic parameter recipe, 77-token batches, pooling counts, CPU/F32, one thread, no-grad and `attention_basic`. The script validates the existing stock corpus SHA-256 `edc3470a883c96f79e75d4c222b2ba8089ba4d4f78de0d5b941dc03175135f6c`; it consumes only its inputs, never its expected outputs as computed source results.

Only the selected frozen attention and CLIP class definitions execute. A documented operation adapter uses PyTorch Linear/LayerNorm/Embedding without random initialization; every parameter is assigned before a forward. Embedding's additional `out_dtype` argument is forwarded to the weight dtype. The source files are hash checked before AST extraction. No ComfyUI installation, custom-node import, model download or inference service is used.

## Collection and interpretation

1. Restore exact hashed CPU laboratory dependencies in a temporary virtual environment. PyTorch macOS wheel build identifiers must remain explicit: different wheels may share the same Python package version. The requirements files identify the actual chosen wheel.
2. Fetch only the four public source/configuration files from the frozen commit into a separate temporary directory and verify their hashes. The source script performs its own verification too.
3. Check configurations, token batches and all 197 L / 517 G parameter hashes. Run the original stock options three times and record whether output hashes repeat. Collect an additional all-layer forward, including source hooks for normalization/projections and actual post-block, final and pooled tensors.
4. Save tensor shapes, byte counts and SHA-256, source/AST provenance, Python/Torch versions, CPU capability, threading/build information and native-binary hashes. Output stays in a new diagnostic directory. The script cannot overwrite the accepted fixtures.
5. First validate artifact integrity and exact discrete inputs. Then compare source versus product on each target environment and source versus source across environments, including the original Windows reference. Preserve dimensions and layout when isolating an operator. Report full elementwise errors and failures under the **existing** `3e-5 + 3e-5 * abs(expected)` bound; do not use aggregate averages to replace failing elements.

A successful workflow means the diagnostic collection completed. It does not turn existing stock failures into passing tests. If a same-platform source/product discrepancy remains, investigate its first operator and native identity. If source/product agrees but source/source differs, record the distinction between port fidelity and arithmetic portability. Neither outcome alone establishes an error bound for different parameters, pretrained weights, other devices or generated media.

Any future qualification profile requires a separately reviewed protocol and thresholds fixed before acceptance, independent source evidence and held-out cases. The original corpus, failing results and full V1 requirements remain intact.

## Run separately

Use Python 3.12.10 and the requirements file for the selected runtime in a dedicated environment. The separate `CLIP source laboratory` GitHub workflow supplies verified source snapshots and invokes:

```text
python -I -B labs/clip-source/reference.py --source-directory SNAPSHOT --inputs tests/ComfySharp.Inference.Tests/Fixtures/clip-stock.cpu-f32.json --output NEW_DIAGNOSTIC_DIRECTORY
```

`SNAPSHOT` contains the four `comfy/...` paths from the pinned source. The output directory must be separate from the checkout, source snapshot and accepted fixture locations. These instructions are for laboratory contributors; Python is not required to build, test or run ComfySharp.
