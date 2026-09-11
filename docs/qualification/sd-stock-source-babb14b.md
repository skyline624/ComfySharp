# First stock-width U-Net source comparisons

SD15 and SD2 produced bit-identical source and C# outputs on the local Windows
CPU with synthetic parameters, stock channel widths and small 16×16 latents.
The source collector at
[`babb14b40524c7fa6df998156048d23fec88827e`](https://github.com/skyline624/ComfySharp/tree/babb14b40524c7fa6df998156048d23fec88827e/labs/sd-stock-source)
ran each model in one fresh process, three forwards per process, sequentially.
Comparison reuses the [original product executions at 2ee03aa](sd-stock-diagnostic-2ee03aa.md).
It does not declare a compatible pretrained model or a completed workflow.

| Synthetic component | Parameter records matching source | Exact input files | Exact output scalars | Maximum absolute error | Outside fixed bound |
|---|---:|---:|---:|---:|---:|
| SD15 U-Net, context width 768 | 686 | 3 | 1,024/1,024 | 0 | 0 |
| SD2 U-Net, context width 1,024 | 686 | 3 | 1,024/1,024 | 0 | 0 |

Each side repeats its output hash identically three times. The source derives
its own parameter schema from the frozen upstream model, then fills its native
bank using the declared deterministic recipe. All parameter names, shapes,
byte counts and SHA-256 hashes agree with the product. Input F32 files and
complete output F32 files were independently decoded and rehashed by two audits;
no comparison relies on the JSON preview decimals. There are 16 input/output
payloads across the two models and two implementations. Weight payloads are
not retained; their hashes attest the bytes observed during generation.

The prospective `sd-unet-stock-small-native210-cpu-f32-v1` profile keeps
`3e-5 + 3e-5 * abs(expected)`, exact input/shape/dtype requirements and explicit
nonfinite handling. No reference or tolerance was replaced. Output SHA-256s are
`beaba5318df8b92baa5975225e07b75f1a28da91dd416c4e878add220254e8c1`
for SD15 and
`f47f5f8b52112233c16e8a483df0154365ec16be95c3c504aa3edb2f68693b21`
for SD2.

## Source provenance and the corrected collector

Both new source processes use Python 3.12.10, CPU PyTorch 2.10.0, its recorded
Git revision, NumPy 2.2.6 and einops 0.8.1 in the separate laboratory. Threads
are 1/1, gradients are disabled and actual global ATen capability is AVX2.
Three frozen ComfyUI blobs and the selected AST hashes were independently
reconstructed. Collector, cases, two helpers, dependency lock and three source
files retain their eight raw fingerprints before and after execution.

The first collector at `4b63561` mistakenly pinned the local CRLF bytes of all
three dependency locks. Its local outputs were exact, but a canonical Git LF
checkout would reject those pins. Those artifacts and findings remain
historical evidence. Collector version `stock-source-v2-canonical-lock-content`
now admits the exact canonical Git content after only CRLF-to-LF substitution,
while retaining a separate strict raw byte count and SHA check across execution.
Even changing only line endings during a run prevents final publication.

The new runs record the unchanged Windows CRLF lock: 3,209 bytes, raw SHA
`db3ba7b6998392497b69501a1d72c74499562503b5a41d98d72d9e12ae7794e3`.
Its canonical LF form is 3,178 bytes, SHA
`53becb18e5c1ea63de4ee8f6eacdd482bcd992827be25439a0a84a89cbc099d5`.
Before/after raw and canonical attestations match exactly. All three canonical
lock pins were checked against Git blobs. Author and independent review ran
32 and 46 bounded stdlib controls respectively, including real content
corruption and changes during publication. Metadata plans ran from a minimal
LF checkout without site packages; because plan mode does not inspect locks,
the lock validator was also exercised directly. Full stock calculations here
used the recorded CRLF lock, not a second complete LF checkout.

Source module maps are captured after the first forward and list the loaded
native images and hashes. The original C# stock processes did not collect an
actual native module map. Observations from other test processes cannot fill
that gap; this comparison therefore makes no native-origin attribution.

## Resource observations and remaining work

Both source processes passed measured admission with more than 45 GiB of
available physical RAM and a requested 6 GiB budget. Each uses one native bank:
3,438,083,856 parameter bytes for SD15 or 3,463,642,896 for SD2, with 2 MiB of
bounded fill scratch. The budget is an admission estimate, not a guarantee of
peak memory. The earlier product report's retained working set after disposal
is still unresolved; these short runs do not establish freedom from leaks.

This campaign adds two new source processes and reuses the two original product
processes. It is not a two-fresh-processes-per-implementation campaign. Stock
channel widths on one square latent do not cover odd sizes, larger batches,
full image dimensions, pretrained checkpoints, SD2 text encoder integration,
VAE, a sampling trajectory, gradients, GPUs or clean installation. Earlier
Windows/Linux SD failures and macOS/Linux CLIP failures remain open under their
accepted profiles. No model family or platform is newly marked qualified.

[Machine-readable evidence](sd-stock-source-babb14b.json) retains both source
manifests, collection metadata, all parameter records, runtime observations,
resource admission and the independent comparison results. The
[full migration plan](../MIGRATION.md) remains the release contract.
