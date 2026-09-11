# Euler comparison in the three-platform CI

[Build and test 34651839113](https://github.com/skyline624/ComfySharp/actions/runs/34651839113)
tested commit `2087602a4865cd9e074d20a04ff32ce9f5324950`. Windows passed;
Linux and macOS failed. The new Euler trajectory passes its eight source
reference cases on Windows and macOS, while seven cases fail on Linux.
The [comparison profile](sd-euler-native210-cpu-f32-v1.md) and all accepted
references retain their original bounds.

## Executed tests

| Target | First-use results, 190 cases | Unique aggregate results | Euler first-use | Euler behavior / reference in aggregate |
|---|---|---|---|---|
| Windows x64 | 190 passed | 1,297 passed | 8 passed | 9 / 8 passed |
| Linux x64 | 175 passed, 15 failed | Not executed | 1 passed, 7 failed | Not executed |
| macOS arm64 | 190 passed | 1,295 passed, 2 failed | 8 passed | 9 / 8 passed |

All three locked restores and builds succeeded with zero compiler warnings or
errors. The audit recounted 38 TRX reports from 15 artifacts. Their recorded
results contain no ignored tests. The first-use tests repeat identities from
the aggregate and are not added to its 1,297 unique cases. The Windows/macOS
aggregates each contain 511 Inference tests.

Linux's first-use loop continued through Euler after the existing U-Net/CFG
failures. That combined step then failed, so the general tests, catalogue,
CLI, Desktop/Host, CPU probe and stock CLIP steps did not execute. Its nine
Euler behavior tests cannot be reported as passing. The missing StockTraces
upload is a secondary workflow error, not another numerical test failure.
Windows passed the later checks; macOS passed the non-stock checks and retained
the two CLIP stock failures.

## Euler records and provenance

The source references come from the [independent collection at 96fec72](sd-euler-source-96fec72.md).
The protocol was published before computation, with SHA-256
`7868368a38fdf895fbe8698d4ab5bbbea1352a11df6dd7b3bd6c1e776c55cd05`.
The audit verified all three committed source manifest pins and each payload's
size and SHA-256 before comparison.

Each target supplied eight complete trace files: 31 input records and 81
comparison records. The latter comprise nineteen pre-update steps with latent,
denoised and sigma snapshots, plus three final outputs for every case. SigmaHat
equals the observed sigma because churn is zero; it is not counted as another
observed native operation.

All inputs, configurations, guidance options and scalar sigmas match the source.
Each case's 686 parameter name/shape/hash records match before and after its
three trajectories. Inputs remain unchanged, caller grad mode is restored,
observed internal grad mode is disabled, and output tensors require no gradient.
The three output hashes are stable in every case, including the failed Linux
cases: all three runs complete before numerical assertions.

| Target | Bit-identical comparison records | Largest absolute difference | Outside-bound observations |
|---|---:|---:|---:|
| Windows x64 | 81 / 81 | 0 | 0 |
| Linux x64 | 27 / 81 | 0.0021082162857055664 | 1,587 |
| macOS arm64 | 81 / 81 | 0 | 0 |

The Linux total counts observations across all 81 records, including the three
identical final-output repetitions. Counting one final output per case instead
gives **260 distinct final-output elements outside the bound**. These numbers
measure different things and must not be interchanged.

| Linux case | First record outside the bound | Final elements outside the bound, one repeat |
|---|---|---:|
| one-step | None | 0 |
| three-step-batch | step-0/denoised | 48 |
| unequal-separate | step-0/denoised | 72 |
| unequal-concatenated | step-0/denoised | 48 |
| near-one-null | step-1/denoised | 5 |
| near-one-present | step-1/denoised | 5 |
| near-one-disabled | step-1/denoised | 5 |
| velocity-three-step | step-2/denoised | 77 |

Every Linux case already has a nonzero difference in its first denoised snapshot,
before the first Euler update; the initial latent and sigma match exactly.
One-step remains within tolerance. This observation locates a difference in the
guided-denoising result preceding integration. It does not identify a native
primitive or prove a package or dispatch cause. Later evolving latents differ,
so their subsequent states cannot isolate an Euler operation by themselves.

The source and product runs use the same declared platform/release, with source
collection performed on a separate CI host. They are not a same-host or
identical-binary experiment. Windows product reports AMD EPYC 9V74 and Linux
AMD EPYC 7763, with one intra/inter-op thread and no requested ATen override.
Actual product ATen dispatch remains unavailable; CPU feature flags do not
measure it. Product library identities use a process cache that may predate an
Euler case. macOS's loaded-module evidence remains explicitly unavailable.

## Existing failures and remaining gates

Linux retains the two U-Net and six CFG first-use failures, bringing its total
to fifteen with Euler. Its 80 U-Net capture hashes are unchanged from the two
preceding audited runs and retain 42 final-output elements outside the bound.
macOS retains 80 source-exact U-Net records. Windows's U-Net hashes match the
earlier successful AMD campaign, rather than the intervening Xeon campaign.
The [earlier CPU variation](sd-cpu-variation-6b1347c.md) remains relevant;
this Windows success does not qualify all Windows CPUs.

The 36 available Windows/macOS CLIP payloads were rehashed and match the
corresponding earlier captures, as do their parameters, tokens and configuration.
macOS's two CLIP failures remain open. No old profile, expected output or
dependency was changed to obtain the Euler results.

These observations cover short CPU trajectories, synthetic weights and reduced
dimensions. They do not qualify stock-width sampling, pretrained checkpoints,
sustained memory behavior, other devices, complete image workflows or model
families. The full three-platform CI remains failed.

[Machine-readable evidence](sd-euler-ci-2087602.json) records the commit, source
manifest and payload hashes, artifact identities, test and step outcomes,
per-case metrics and public runtime metadata. Archive digests are GitHub's
declared identities; extracted tensor records were independently rehashed.
The audit used stdlib only and ran no native computation.
