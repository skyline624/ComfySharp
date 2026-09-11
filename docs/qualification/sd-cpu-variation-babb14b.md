# Normal CI after the stock collector provenance correction

The [normal CI at babb14b](https://github.com/skyline624/ComfySharp/actions/runs/34647655904)
fails overall. Windows passes all 1,280 unique tests, while macOS retains its
two previous stock CLIP failures and Linux fails eight SD comparisons at first
use. Product, tool, test, dependency and normal workflow code is unchanged from
`2ee03aa` to this revision; the collector correction does not explain or fix
the numerical variation.

| Platform | First-use executions | Aggregate plus stock CLIP tests | Later six CLI/Host checks |
|---|---:|---:|---|
| Windows | 182 pass | 1,280 pass | Pass |
| Linux | 174 pass, 8 fail | Not executed | Not executed |
| macOS | 182 pass | 1,278 pass, 2 previous CLIP failures | Pass |

First-use runs repeat cases and are not added to the 1,280 unique total. The
128 cases in SD-related classes include 35 tool/runtime checks. Linux fails
six CFG and two U-Net cases; VAE and sampling reference tests pass on every OS.
There are no ignored individual results in the 35 audited TRX files.

Windows again observes AMD EPYC 7763, AVX512 flags false and all 80 U-Net
captures bit-exact. Its four loaded DLL identities are identical to the
[preceding Intel run](sd-cpu-variation-4b63561.md), which had 76 output values
outside the profile. Linux now also observes AMD EPYC 7763 with AVX512 flags
false. Its 80 U-Net hashes match `2ee03aa` exactly, although that run observed
EPYC 9V74. There are 42 final values outside the fixed bound: 7 in SD15 odd
and 35 in SD2 shared odd batch. Its native library identities are unchanged.
Actual ATen dispatch remains unavailable in the product records; these
associations neither identify a kernel nor establish causality.

All 12 artifacts were downloaded, including complete SD traces and all
expected TRX files. The Linux upload failure is for `StockTraces`, absent
because its stock test step was not executed after first-use failure. No SD
trace artifact was lost. The audit independently decoded and rehashed 240
observed records and 240 source counterparts; accepted fixtures and source
scripts retain their identities.

[Machine-readable evidence](sd-components-babb14b.json) preserves the test
counts, observed runtimes and comparisons. The separate
[stock-width source experiment](sd-stock-source-babb14b.md) remains a local
synthetic comparison and does not close these CI failures or qualify a model
family. No reference or tolerance changed.
