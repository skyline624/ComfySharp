# Reduced U-Net and CFG native-origin suite

The [isolated Linux suite](https://github.com/skyline624/ComfySharp/actions/runs/34649263359)
at commit `6b1347ccb3f1a4aab96e337925978ea7740992f0` completed all eight
processes and retained complete evidence, but the workflow **failed**. Five
product processes failed accepted-reference assertions, and 22 of 28 case/mode
contrasts did not satisfy the copy and input-layout gates required for native
package interpretation. Artifact integrity does not turn these failures into
qualification.

This extends the [earlier single-case copy diagnostic](sd-native-copy-3583d61.md)
to eight existing reduced U-Net cases and six CFG policy outputs. It uses
synthetic parameters, CPU/F32 and one Linux host. It does not cover stock widths,
pretrained models or a complete image workflow.

## Accepted references and same-host source comparisons

Each product process executes the same fourteen existing tests against the
unchanged accepted corpus. The separate comparison uses the newly executed
source on the same host in the same requested mode. These are distinct results.

| Mode | Product native origin | Accepted tests: pass / fail | Exit | Exact captures against same-host source | Captured values outside fixed bound |
|---|---|---:|---:|---:|---:|
| AUTO | Original NuGet | 9 / 5 | 1 | 23 / 86 | 54 |
| AUTO | Identical NuGet copy | 9 / 5 | 1 | 23 / 86 | 54 |
| AUTO | Wheel-native copy | 14 / 0 | 0 | 86 / 86 | 0 |
| DEFAULT | Original NuGet | 6 / 8 | 1 | 86 / 86 | 0 |
| DEFAULT | Identical NuGet copy | 6 / 8 | 1 | 86 / 86 | 0 |
| DEFAULT | Wheel-native copy | 6 / 8 | 1 | 86 / 86 | 0 |

The six TRX files contain **50 passes and 34 failures across 84 executions of
14 existing cases**. These are repeated executions, not 84 new tests. Counters,
test identities, runner notifications and exit codes agree; no individual
result was ignored or unexecuted. Both source processes exited successfully.

The source reports global AVX512 in AUTO and DEFAULT when explicitly requested.
Its fourteen AUTO outputs equal the accepted corpus byte for byte. Its DEFAULT
outputs differ from all fourteen accepted outputs and exceed the fixed bound
on eight cases, totaling 106 output values. The DEFAULT product/source agreement
therefore does not resolve the eight accepted-reference failures in each product
process. No reference or tolerance was replaced.

The five AUTO NuGet failures are SD2 shared-time odd and both CFG policies for
context lengths 2/3 and 3/3. Their final out-of-bound element counts are 4,
16, 1, 23 and 10 respectively. All comparisons keep
`abs(actual-expected) <= 3e-5 + 3e-5*abs(expected)`. The finite corpus rejects
unexpected nonfinite values. CFG Separate and ConcatenateCompatible retain
their own source results; no equivalence between the policies is assumed.

## Copy controls limit interpretation

Original and NuGet-copy captures are bit-identical in both modes: 172 of 172
intermediate/output records. However, borrowed input alignment flags vary
between processes. Exact output bytes do not establish equal input layouts or
allocation history. The stricter gates remain applied.

Only these six case/mode contrasts satisfy both the original/copy control,
including layouts, and equal wheel/NuGet input and parameter layouts:

- AUTO: SD15 odd rectangle, SD2 odd rectangle, SD2 distinct-time batch.
- DEFAULT: SD15 square, SD15 odd rectangle, CFG 2/3 Separate.

The other **22 of 28** contrasts remain explicitly ineligible for native package
attribution. Their numerical results are retained as observations. Even an
eligible contrast does not identify a specific primitive, compiler flag or
binary: the three core libraries and OpenMP change together. The nine coarse
U-Net observation points do not localize an exact first differing operator.
No runtime dependency is adopted into the product by this experiment.

## Integrity and provenance

The three artifacts contain 95 JSON files, six TRX files and eight process logs.
An independent stdlib audit reconstructed and rehashed 1,072 F32 records: 688
intermediate/output captures and 384 inputs. It separately rehashed fourteen
accepted outputs, then recomputed 946 capture contrasts and 528 input contrasts.
All metrics and gate outcomes agree with the collector. These are artifact
calculations, not additional native forwards.

All fourteen cases per process have the required 686 parameter identities,
complete layouts and matching before/after records. Parameters, inputs,
configurations and CFG options match the accepted input corpus. Three output
hashes agree in every case. U-Net source uses its existing on/off/off observer
sequence, product U-Net uses off/on/off, and CFG uses off/off/off. Instrumentation
can change allocation and timing; these checks do not assert equivalence to an
uninstrumented normal-CI process.

Six pinned helper files, three diagnostic scripts and six accepted source scripts
match committed blobs. Six upstream source blobs were independently rehashed,
and seven distinct selected ASTs were reconstructed without execution. The
accepted 23-file fixture inventory remains unchanged. The Linux dependency lock
is identical before and after: 3,280 bytes, canonical Git LF SHA-256
`84df56cd98339e8dfec9b6f765312758706476ed1328d5f95f6a06433b9bb719`.

Before/after inventories match for the original build, both copies, wheel input,
source snapshot and fixtures. Original and NuGet copy each have 121 files;
wheel copy has 122. The wheel copy preserves 117 files, replaces the three cores
and OpenMP, and adds one GNU OpenMP SONAME alias. Internal hardlinks remain
independent of the package input. No wheel path or LD_PRELOAD is added; the
validated baseline loader environment is preserved. Large native files and
inode relationships are producer attestations, since these files were not
uploaded for independent local rehashing.

## Observed native identities

The host reports Intel Xeon Platinum 8573C. Source uses Python 3.12.10,
PyTorch 2.10.0+cpu at its pinned Git revision, NumPy 2.2.6 and einops 0.8.1;
intra/inter-op threads are 1/1. Product records .NET 10.0.12, TorchSharp 0.107
and declared libtorch 2.10. A version string does not establish binary identity.

Actual product maps were checked after each of fourteen cases in each of six
processes: **84 snapshots**, with exactly the selected c10, torch_cpu, torch,
unchanged bridge and OpenMP images. Each process's snapshots agree. The managed
TorchSharp.dll identity is checked separately 84 times against the original
build. No duplicate core, mixed OpenMP or Python image appears in product maps.
The two Python processes have their own actual loaded maps, including the
permitted Python binding; these are not merely wheel inventories.

The NuGet CPU image is 461,211,481 bytes with SHA-256
`0f4b3e14ed8468219fefe110605e741afc6bfbec27cc777a224301edb9f24c41`.
The wheel CPU image is 443,509,856 bytes with SHA-256
`4dc5b3a649b61d8f39be127866bc258dda19d74e339f449fe18103db63e8d76f`.
Source global capability observations do not establish every operator or BLAS
dispatch; product requests and CPU flags do not replace that missing evidence.

[Machine-readable evidence](sd-native-suite-6b1347c.json) preserves inventories,
maps, script identities, all comparison metrics, ineligible contrasts and audit
results. Archive digests are GitHub-reported identities, not locally rehashed
ZIPs. The [normal CI of this revision](sd-cpu-variation-6b1347c.md) remains
separate and failing. Neither campaign qualifies a model family or platform.
