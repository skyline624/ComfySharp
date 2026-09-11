# Frozen-source Euler trajectories and local C# comparison

The [independent source collection](https://github.com/skyline624/ComfySharp/actions/runs/34650726593)
succeeded on Windows x64, Linux x64 and macOS arm64 at ComfySharp commit
`96fec72dd928276fc3f0ddc93a17406ea8249073`. The eight cases and their input hashes
were published before computation. The [profile](sd-euler-native210-cpu-f32-v1.md)
retains the fixed `3e-5 + 3e-5*abs(expected)` bound.

## Source integrity

Two independent stdlib audits verified the immutable protocol, eight upstream
files, nine selected AST records, two helpers, three canonical dependency locks
and all six imported manifest/payload files. The source parameter recipe was
independently recomputed for 686 unique parameters; all 2,058 parameter records
agree across the three targets.

The audit decoded and rehashed 345 Float32 records: per target, 31 inputs,
eight outputs and nineteen pre-update steps containing four tensors each.
All 93 inputs match their prospective hashes. Each case has the expected step
count and exact scalar sigmas, with stable off/on/off output hashes. Source
input and parameter preservation are checked by the verified collector. Its
after-state parameter records are not separately exported; that distinction is
retained in the evidence.

| Target | Source capability observation | Manifest SHA-256 |
|---|---|---|
| win-x64 | AVX2 | `b528a5f0055ddbae5dad409dffc1bac7a74d08295b638920165791441eaf17a8` |
| linux-x64 | AVX2 | `b1aaf6e2e1a6c8d51fe261012cd20257980d8e4a4561f5bbb016c915be72fa1e` |
| osx-arm64 | DEFAULT | `002da5ae9cb823cdf88523c9e8a22328cd2305bbafd82620ab708e8eb323292f` |

Source used Python 3.12.10 and PyTorch 2.10 CPU with one intra-op and inter-op
thread. Its wheel inventory hashes describe package contents, not actual loaded
library maps. Global CPU capability does not identify every operator's dispatch.
GitHub archive digests are declared archive identities; extracted tensor payloads
and provenance files were independently rehashed.

## Local product integration

The C# trajectory implementation passed nine behavior tests and eight numerical
reference cases on the local Windows CPU. A subsequent full solution run passed
**1,297 unique tests**, including **511 Inference tests**, with no failure or
ignored test. The new seventeen tests are included in these totals.

The full run also recorded the eight Euler cases before numerical assertions.
Independent artifact comparison found all **81 observed comparison records
bit-identical** to the Windows source: nineteen steps with latent, denoised and
sigma snapshots, plus three final outputs for each of eight cases. Inputs and
all 686 parameters per case agree before/after; repeated outputs are exact.
SigmaHat is the same scalar sigma because churn is zero, not a separately
reconstructed native operation. Runtime library information retains its explicit
process-cache timing limitation.

Behavior checks cover retained ownership, disposal during an active operation,
input preservation, output lifetime, no-grad restoration, cancellation and
observer exceptions, invalid schedules, plateau sigmas and strided schedules.
They use actual reduced graphs and are distinct from the numerical oracle.

These are short CPU trajectories with synthetic weights and reduced dimensions.
They do not validate a pretrained checkpoint, stock-width sampler, complete
text-to-image workflow, GPU, sustained memory behavior or model family. C# Euler
execution on Linux/macOS remains a separate CI step. Existing Windows/Linux
component failures and macOS CLIP failures remain open under their accepted
profiles; no reference was replaced to obtain this local result.

[Machine-readable evidence](sd-euler-source-96fec72.json) records source audit
provenance, immutable import hashes, local test identities and comparison metrics.
Product/test file hashes bind the local result to the reviewed implementation;
private testhost paths and raw local logs are not published.
