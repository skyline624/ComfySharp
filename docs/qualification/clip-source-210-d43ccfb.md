# CLIP source comparison with PyTorch 2.10

The [isolated laboratory protocol](../../labs/clip-source/README.md) was recorded in commit `d43ccfbe01224a9f22880bfe1d464a73f1f513e8` before its target-platform results. Its [collection campaign](https://github.com/skyline624/ComfySharp/actions/runs/34627039048) completed successfully on Windows x64, Ubuntu 24.04 x64 and macOS 14 ARM64. Collection success is not an acceptance test or a model-compatibility claim.

The laboratory executes verified ComfyUI source at `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a` under CPU PyTorch 2.10, using the existing synthetic stock inputs. It does not update expected outputs in the .NET tests. Product traces come from the [5810a4f campaign](https://github.com/skyline624/ComfySharp/actions/runs/34625013231); subsequent changes add diagnostics and documentation, without changing the encoder arithmetic.

## Integrity and same-platform observations

Independent inspection verified all 1,758 source tensor payloads, totaling 1,869,742,080 bytes, against their SHA-256, dimensions and byte counts. The four source/configuration files, two selected AST hashes, stock input hash, configurations, tokens and options match the pinned protocol. All 197 L and 517 G parameter hashes agree with the product on each OS. Each source case performs three identical original forwards; its additional all-layer output agrees with the source hooks and selected intermediate state.

Comparison with source 2.10 on the corresponding platform gives:

| Runtime | Encoder | Final / all post-blocks / raw pooled | Projected pooled maximum absolute difference | Elements outside existing bound |
|---|---|---|---:|---:|
| Windows x64 | L | Bit-identical | 1.7881393432617188e-7 | 0 |
| Windows x64 | G | Bit-identical | 4.76837158203125e-7 | 0 |
| Ubuntu x64 | L and G | Bit-identical | 0 | 0 |
| macOS ARM64 | L and G | Bit-identical | 0 | 0 |

Product diagnostic embeddings, first LayerNorm and Q/K/V also match the corresponding actual source hooks exactly. Those early product boundaries remain independent reconstructions from the same bank, whereas final/all/pooled outputs are actual product forwards. The comparison uses the unchanged elementwise bound `3e-5 + 3e-5 * abs(expected)`; no threshold was adjusted.

Source 2.10 on Linux/macOS reproduces the failures against the original Windows 2.13 reference, including the first post-blocks outside tolerance: Linux L/G zero-based indexes 6/3 and macOS L/G indexes 1/0. This separates agreement with the local source execution from portability to the original reference. It supplies evidence of port fidelity for these particular synthetic inputs and environments, without explaining every native arithmetic difference or validating other weights.

## Native identity and limits

All three source runtimes report PyTorch source commit `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`, CPU/F32, one intra-op and one inter-op thread. Source ATen capability is AVX2 on Windows/Linux and DEFAULT on ARM64. The source packages report MKL 2025.3 on Windows, MKL 2024.2 on Linux and Accelerate on macOS.

The source wheels' `torch_cpu` hashes differ from the previously inspected NuGet CPU libraries on all three platforms. Therefore this controls the declared release and platform, **not** identical native binary bytes. Windows and macOS OpenMP library bytes agree with the inspected NuGet bundles; Linux OpenMP differs. Package-file inspection does not prove which libraries were loaded into a remote process. These distinctions remain part of the provenance, even though the observed tensor arithmetic agrees.

The full source corpus is generated and read in an isolated diagnostic environment. Python and its laboratory packages are not application, model-loader or distributed .NET test dependencies. No pretrained model file was used or downloaded.

## Product CI remains explicit

The [normal CI at d43ccfb](https://github.com/skyline624/ComfySharp/actions/runs/34627038991) still reports 1,152 passing tests on Windows and 1,150 passing plus two stock failures on each of Linux/macOS, with zero skips. The 157 first-use repetitions per OS are excluded from those unique totals. All restores, builds, routine tests and six subsequent checks now run successfully on all three OSes: catalogue consistency, safetensors inspection, text tokenization, CLIP inspection/encoding, actual Desktop/Host startup and 25 CPU probe repetitions. The stock comparison was moved after those checks; its assertions and failure status were retained.

The original `clip-basic-cpu-f32-v1` cross-platform qualification remains unsatisfied. No fixture was replaced, no failing test was converted to a pass and no new accepted profile was adopted. A separately scoped backend qualification still needs a protocol and thresholds fixed before acceptance, independent held-out scenarios and product repetition/resource checks. Real checkpoints, conditioning consumers, complete workflows, GPU and training validation remain necessary for model-family and V1 claims.
