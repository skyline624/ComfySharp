# Synthetic safetensors fixture

`metadata-contract.safetensors` is a 218-byte contract fixture created for these tests. It contains an F32 vector `[1, -2.5]` and a BF16 scalar `1`, with 10 payload bytes. Its metadata states that it is synthetic. It contains no model weights and provides no evidence of model-family parity.

The CLI can inspect it in a fresh process whose project has no native CPU/CUDA package:

```sh
dotnet run --project tools/ComfySharp.ModelInspect -- tests/ComfySharp.Inference.Tests/Fixtures/metadata-contract.safetensors --tensors 2 --sha256
```

Expected summary: two tensors, ten tensor bytes, one metadata entry, one F32 tensor and one BF16 tensor. SHA-256 is opt-in and can be compared against an independent file hash.
