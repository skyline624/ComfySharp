# Reduced SD1.5 pipeline source collection

The [source workflow run 34656804132](https://github.com/skyline624/ComfySharp/actions/runs/34656804132)
completed successfully on all three operating systems at collector commit
[`6ad8207ffe46d3d1e57780b9fc4c7288b686638d`](https://github.com/skyline624/ComfySharp/tree/6ad8207ffe46d3d1e57780b9fc4c7288b686638d/labs/sd15-pipeline-source).
Its three artifacts passed an independent standard-library audit. This establishes
source collection and artifact integrity only. It does not establish C# parity or
qualify a pipeline, pretrained checkpoint, stock configuration or model family.

The [prospective protocol](../../labs/sd15-pipeline-source/protocol.json) and
[comparison profile](sd15-pipeline-native210-cpu-f32-v1.md) were published before
these forwards. The collector executes declarations from frozen ComfyUI backend
[`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a).
No C# output is used to generate or validate the source outputs reported here.
Full file hashes, job identities and native observations are in the
[machine-readable evidence](sd15-pipeline-source-6ad8207.json).

| Source target | Successful steps | Skipped steps | Cases | Capture records | Selected parameters |
|---|---:|---:|---:|---:|---:|
| Windows x64 | 12/12 | 0 | 4 | 64 | 971 |
| Linux x64 | 12/12 | 0 | 4 | 64 | 971 |
| macOS arm64 | 12/12 | 0 | 4 | 64 | 971 |

The pipeline uses CLIP hidden width 16, U-Net base width 32 with context width 16,
and classical VAE base width 32. The selected synthetic weights comprise 37 CLIP,
686 U-Net and 248 VAE parameters. CLIP projection is present and the exact source
computes it as an unused auxiliary result; the SD1 wrapper selects unprojected
pooling, and only hidden sequences condition U-Net. This corpus does not test
absent projection.

Four fixed source token-chunk cases use epsilon prediction, separate CFG,
churn zero, explicit F32 sigma sequences and latent scale 0.18215. Latents have
shape `[1,4,4,5]` and decoded images `[1,32,40,3]`. Noise comes from the published
integer recipe, not a Gaussian or seed-compatibility claim. The maximum starting
sigma retains the bits of the previously published source schedule.

Each case runs off/on/off. Per operating system, eight Euler steps per pass
produce 45 U-Net calls across the repetitions, 24 CLIP encodings and 12 VAE
decodings. Only the middle pass captures the eight pipeline boundaries per case
and four values per Euler step. The final diffusion latent, raw VAE latent and
image repeat hashes agree within each case. No extra VAE decode creates a
diagnostic capture.

The independent audit reconstructed and rehashed all 192 F32 capture records,
the 20 unique input recipes and all 971 parameter recipes using standard-library
integer and byte operations. All 2,913 parameter records across the platforms
match those recipes and the pinned schemas. Parameter bits/layouts and borrowed
native inputs remain unchanged. All 48 captured sigma/sigmaHat scalars match the
exact schedule bits and step indices; no activation tolerance applies to them.
All captured values are finite.

Each platform's 36 protected file identities agree before and after collection
and match the appropriate frozen Git blobs. This includes 15 source files, their
CLIP config, the collector and workflow, six helpers, three CPU locks, the profile
and existing source-input donors. The 18 recorded AST selections per platform,
including the VAE normalization assignments, were independently checked. The
ten helper/lock/profile canonical admissions used LF bytes in CI; the strict raw
before/after checks remain separate from CRLF admission rules.

| Target | Observed Torch CPU capability | Actually mapped libraries | Wheel inventory | Identical overlapping identities |
|---|---|---:|---:|---:|
| Windows x64 | AVX2 | 8 | 9 | 7 |
| Linux x64 | AVX2 | 7 | 11 | 6 |
| macOS arm64 | DEFAULT | 7 | 7 | 6 |

All source processes report Python 3.12.10, PyTorch 2.10.0 CPU, NumPy 2.2.6,
einops 0.8.1 and one intra/inter-op thread. Requested dispatch is automatic.
Mapped-library observations come from each source process and are distinct from
its broader package inventory. Their hashes were checked for consistency in the
artifacts; the binary packages were not downloaded or rehashed by the local
audit. Source build metadata reports MKL 2025.3 on Windows, MKL 2024.2 on Linux,
and Accelerate on macOS. These observations do not establish the cause of any
numerical difference.

Source output hashes are OS-specific: Windows/Linux share 24 of 64 record hashes,
and either comparison with macOS shares 22 of 64. These are hash comparisons
between source runs, not product comparisons or tolerance results. No universal
cross-platform golden is inferred.

All nine extracted JSON files were inventoried and checked for private-path
patterns; none were found. The three manifests and three pipeline JSON files
were copied byte for byte to
`tests/ComfySharp.Inference.Tests/Fixtures/sd15-pipeline.<target>.{manifest.json,json}`.
Their source JSON remains unchanged, including the original component filename
inside each manifest. GitHub's archive digests are retained as service metadata;
the audit independently recalculated the extracted file hashes. No expected
value, protocol or tolerance was adjusted during import.
