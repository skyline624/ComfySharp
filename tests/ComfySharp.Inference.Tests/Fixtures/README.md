# Synthetic safetensors fixture

`metadata-contract.safetensors` is a 218-byte contract fixture created for these tests. It contains an F32 vector `[1, -2.5]` and a BF16 scalar `1`, with 10 payload bytes. Its metadata states that it is synthetic. It contains no model weights and provides no evidence of model-family parity.

The CLI can inspect it in a fresh process whose project has no native CPU/CUDA package:

```sh
dotnet run --project tools/ComfySharp.ModelInspect -- tests/ComfySharp.Inference.Tests/Fixtures/metadata-contract.safetensors --tensors 2 --sha256
```

Expected summary: two tensors, ten tensor bytes, one metadata entry, one F32 tensor and one BF16 tensor. SHA-256 is opt-in and can be compared against an independent file hash.

## Sigma schedule references

`sigma-schedules.cpu-f32.json` contains 46 numeric schedules and four invalid-domain cases produced by the exact functions in the pinned ComfyUI source, in an isolated PyTorch 2.13 CPU laboratory. These are reference computations, not model weights or C#-generated expectations. The .NET tests embed this JSON and do not invoke Python. Source hashes, parameter sets, Float32 bits and payload hashes are included.

The comparison profile and intentional invalid-domain diagnostics are defined in [SIGMA_SCHEDULES.md](../../../docs/SIGMA_SCHEDULES.md). Whole-file SHA-256: `c78e03c13d3feed5aaf876a9efa5cc2f9f53822765be5cb6c97b0469f63412e1`. Do not regenerate or loosen the tolerances to make a failing port pass.
