# Normal CI with the optional native-suite collector

The [normal CI at 6b1347c](https://github.com/skyline624/ComfySharp/actions/runs/34649263342)
fails overall. Windows records eleven first-use SD failures, Linux eight, and
macOS retains the two previous stock CLIP failures. This is separate from the
[isolated native-suite diagnostic](sd-native-suite-6b1347c.md) at the same commit.

| Platform | First-use executions | Aggregate plus stock CLIP tests | Later six CLI/Host checks |
|---|---:|---:|---|
| Windows | 171 pass, 11 fail | Not executed | Not executed |
| Linux | 174 pass, 8 fail | Not executed | Not executed |
| macOS | 182 pass | 1,278 pass, 2 previous CLIP failures | Pass |

First-use cases repeat existing tests and are not added to the expected 1,280
unique total. Only macOS reaches that total, including 494 Inference tests and
128 SD-related tests; the latter include 35 tool/runtime checks. Windows fails
five U-Net and all six CFG cases; Linux fails two U-Net and all six CFG cases.
The seven first-use processes continue within their step, so VAE and sampling
references are still exercised and pass on all three platforms.

All nine published artifacts were downloaded and all 28 TRX files recounted,
with no ignored or unexecuted individual result. Restores and builds pass with
zero compiler warnings or errors. Aggregate, catalogue, CLI, Desktop/Host,
native CPU probe and stock CLIP steps are not reached on Windows/Linux. Their
additional upload failures report absent `StockTraces` after stock tests were
not executed; no SD trace artifact was lost. macOS passes the six non-stock
checks and fails only the existing stock CLIP L and G final-output assertions.

## Captures and hardware observations

All three platforms produced eight U-Net traces. The audit decoded and rehashed
240 observed F32 intermediate/output records and checked their shapes and
metrics against the committed platform references, using the unchanged
`3e-5 + 3e-5*abs(expected)` bound.

| Platform | Exact U-Net captures | Final values outside bound | Largest final absolute error |
|---|---:|---:|---:|
| Windows | 0 / 80 | 76 | 0.00017480552196502686 |
| Linux | 0 / 80 | 42 | 0.00016261637210845947 |
| macOS | 80 / 80 | 0 | 0 |

All intermediate captures remain within the bound. Windows final failures are
SD15 square (11 values), odd rectangle (3), distinct-time batch (5), SD2
distinct-time batch (4) and shared-time odd batch (53). Linux final failures
are SD15 odd rectangle (7) and SD2 shared-time odd batch (35). These first-forward
captures agree with the U-Net TRX failures. Failed tests stop at their first
assertion, so they do not establish three successful repeats. Normal CI emits
no complete CFG output payloads; its CFG failures are retained as TRX assertions,
without inventing full-output comparisons.

Windows observes Intel Xeon Platinum 8573C with AVX512 available to managed
intrinsics; the [preceding run](sd-cpu-variation-babb14b.md) observed AMD EPYC 7763
and passed. Linux again observes EPYC 7763 with AVX2 and no AVX512. Both record
threads 1/1, no requested ATen capability override, and actual ATen dispatch
explicitly unavailable. Loaded native basename/size/SHA sets match the preceding
run; Windows enumeration order differs, while Linux's complete recorded runtime
identity is unchanged. macOS's native Process.Modules capture remains explicitly
unavailable. None of these observations establishes a CPU-dispatch cause.

The macOS CLIP audit also rehashed eighteen finite F32 payloads. All eighteen
hashes and 714 parameter records, tokens and configurations match the preceding
campaign. L fails at output index 27 and G at index 5 as before. Reconstructed
diagnostic operations remain distinct from true-forward outputs and do not
prove that those operations observed actual internal forward execution.

## Scope of the change

Product source, tools, dependency properties, accepted fixtures and the normal
workflow are unchanged since `2ee03aa`. Tests are not unchanged: the two SD
reference classes now call an optional native-suite helper. The suite environment
option is absent from this normal workflow, so that branch returns without
performing the diagnostic. Unchanged model mathematics does not establish
identical JIT behavior or allocation history, and no causal attribution to this
test change or the observed CPU is made.

[Machine-readable evidence](sd-components-6b1347c.json) retains counts, stage
outcomes, runtime identities, numerical metrics and audit provenance. The
separate local stock-width source agreement does not erase these failures.
No fixture, tolerance, pretrained model, workflow or platform is newly qualified.
