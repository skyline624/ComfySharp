# SD CPU variation with observed runtime identities

The normal CI campaigns at
[`3583d61`](https://github.com/skyline624/ComfySharp/actions/runs/34645018396)
and [`4b63561`](https://github.com/skyline624/ComfySharp/actions/runs/34645660414)
both failed overall. Product calculations, dependencies, normal workflow and
accepted references are unchanged from `2ee03aa` through these revisions.
The observations therefore preserve variation rather than replacing a failing
reference or relaxing the numerical profile.

| Commit / platform | First-use tests in separate processes | Aggregate suite plus stock CLIP tests |
|---|---:|---:|
| 3583d61 / Windows | 182 pass | 1,280 pass |
| 3583d61 / Linux | 175 pass, 7 fail | Not executed |
| 3583d61 / macOS | 182 pass | 1,278 pass, 2 previous CLIP failures |
| 4b63561 / Windows | 171 pass, 11 fail | Not executed |
| 4b63561 / Linux | 175 pass, 7 fail | Not executed |
| 4b63561 / macOS | 182 pass | 1,278 pass, 2 previous CLIP failures |

There are 1,280 expected unique test cases. The 182 first-use executions repeat
cases and do not increase that total. The 128 cases in SD-related classes
include 35 tool/runtime-identity checks; they are not 128 source comparisons.
All reported individual TRX results were executed, with none ignored. Where
first-use checks failed, the aggregate, stock CLIP and six later CLI/Host checks
were not executed. Missing uploads for those later checks add no numerical
test result. VAE and sampling source comparisons pass on all three systems.

## Windows observations

At `3583d61`, the process observes AMD EPYC 7763 and the 80 U-Net captures agree
exactly with source. At `4b63561`, it observes Intel Xeon 6973P-C, with
AVX512F/DQ/BW/VL exposed, while the earlier AMD process reported those flags
false. The four loaded native DLL names, sizes and SHA-256 identities are
identical, as are the recorded .NET 10.0.12, TorchSharp, libtorch and thread
settings. Actual ATen dispatch is explicitly unavailable in these observations.

The Intel run fails six CFG and five U-Net comparisons. None of its 80 U-Net
captures is bit-identical to source, with differences already in all eight
time embeddings. There are 76 final values outside the fixed profile: 11 in
SD15 square, 3 in SD15 odd, 5 in SD15 distinct batch, 4 in SD2 distinct batch
and 53 in SD2 shared odd batch. The largest absolute error is
`1.7480552196502686e-4`. None of these 80 hashes matches either the earlier
successful Windows campaigns or the previously observed failing variants.

This associates measured CPU exposure with different results; it does not
establish the causal instruction path or infer the CPU used in older runs
that lacked these observations.

## Linux and macOS observations

Linux has four CFG and three U-Net failures in both campaigns. Its 80 U-Net
hashes and complete runtime identity agree between these two runs. There are
42 final values outside the profile: 7 in SD15 odd, 11 in SD15 distinct batch
and 24 in SD2 shared odd batch. These hashes match `feeb97f` and differ from
most of those observed at `2ee03aa`. Both newer runs observe AMD EPYC 9V74
with AVX512 flags true; `2ee03aa` observed the same brand with those flags
false. The five native library identities are unchanged. This again remains
an association, with actual ATen dispatch unavailable.

macOS preserves all 80 exact U-Net captures and its two earlier stock CLIP
failures. CPU CPUID and actual native module mapping remain explicitly
unavailable there. No missing observation is reconstructed from runner labels.

## Evidence and limits

Independent audits counted 12 artifacts/35 TRX at `3583d61` and 9 artifacts/28
TRX at `4b63561`. For each campaign, all 240 observed records and their 240
source counterparts were decoded, rehashed and compared. The 15 fixture files,
manifests and six source scripts retain their accepted identities. Each set
of eight runtime observations agrees within its process; the underlying
snapshot is captured after the first traced forward and cached, not refreshed
for every case or possible later native load.

The reports [for 3583d61](sd-components-3583d61.json) and
[for 4b63561](sd-components-4b63561.json) contain counts, per-case errors and
runtime identities. The separate [native-copy experiment](sd-native-copy-3583d61.md)
is a valid one-case package comparison; its success does not close these normal
CI failures. Multiplatform acceptance, pretrained models and full workflows
remain open.
