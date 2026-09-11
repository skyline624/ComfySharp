# Reduced U-Net and CFG with controlled input allocation

The [isolated Linux run](https://github.com/skyline624/ComfySharp/actions/runs/34657176279)
at `996f8a3b05709974fe22b76ca30431fb8cf02bfc` completed its eight processes
and passed the diagnostic integrity gates. **All 28 case/mode native-bundle
contrasts are eligible under the prospective protocol. The workflow still
failed:** its six product processes recorded **35 passes and 49 failures**
against the unchanged accepted references, with no ignored tests.

This is a synthetic, reduced CPU/F32 experiment: eight U-Net cases and six CFG
policy outputs, three repeats each. It neither qualifies a model family nor
adopts a different product dependency. The [protocol](../../labs/sd-native-aligned-diagnostic/protocol.json)
was published before calculation; its SHA-256 is
`27f4b82a8b26cbdfc2ac384d48a0433c2e351c9142efb7592f8f8ccac44ca3e2`.

## What the new input control establishes

Every source/product case unconditionally copies each input into its own native
buffer, preserving F32 bytes, shape, contiguous strides and zero storage offset.
The same copies are passed through all three repeats. Actual input metadata is
captured immediately before and after each forward; every prepared address has
residue zero modulo 64. The experiment checks the resulting alignment rather
than retrying allocations or selecting a path based on the borrowed pointer.

An independent artifact audit verified **384 prepared input records and 2,304
before/after input observations** across 112 case executions. All parameter
name/shape/hash identities and layouts agree before/after and with the accepted
input corpus: 686 parameters per case. Prepared inputs are byte-identical across
origins. Of 288 product borrowed-input records, 255 were initially unaligned;
the remaining 33 also received copies. The source's 96 originals were aligned.

Original-build and identical-NuGet-copy controls match in both modes: **172/172
captured outputs/intermediates are bit-identical**, with equal input/parameter
layouts, offsets and alignment residues. Wheel/NuGet input and parameter gates
also pass for all 28 contrasts. All six product processes execute the same
fourteen cases in the same recorded order. U-Net observers run off/on/off;
CFG has no model observer. Each case's three final output hashes agree.

The [historical run](sd-native-suite-6b1347c.md) had 22 ineligible contrasts due
to input alignment differences. Its Intel AVX512 host differs from this AMD
AVX2 host, and the new protocol changes allocation and observation history.
These runs therefore are not a controlled numerical pair. The new measured
controls establish eligibility for this run; they do not retroactively change
the historical gates or prove identical internal allocation histories.

## Accepted tests and same-host comparisons remain separate

| Mode | Product native origin | Accepted tests: pass / fail | Exit | Exact captures vs same-host source | Captured values outside fixed bound |
|---|---|---:|---:|---:|---:|
| AUTO | Original NuGet | 6 / 8 | 1 | 19 / 86 | 110 |
| AUTO | Identical NuGet copy | 6 / 8 | 1 | 19 / 86 | 110 |
| AUTO | Wheel-native copy | 5 / 9 | 1 | 86 / 86 | 0 |
| DEFAULT | Original NuGet | 6 / 8 | 1 | 86 / 86 | 0 |
| DEFAULT | Identical NuGet copy | 6 / 8 | 1 | 86 / 86 | 0 |
| DEFAULT | Wheel-native copy | 6 / 8 | 1 | 86 / 86 | 0 |

The bound remains `abs(actual-expected) <= 3e-5 + 3e-5*abs(expected)`.
No expected output, tolerance or assertion changed. Separate and concatenated
CFG retain distinct references. Both source processes exited 0; all six product
exit codes agree with their complete TRX results and ordinary assertion-failure
notifications. No global runner/teardown error was found.

The new AUTO source itself exceeds the accepted reference bound on nine final
outputs, totaling 253 values; DEFAULT exceeds it on eight, totaling 346 values.
No new source final output is byte-identical to its accepted counterpart.
Consequently, wheel/source bit identity does **not** resolve the accepted-test
failures. AUTO/DEFAULT source comparisons differ in all 86 captures, with 225
values outside the bound, while source files, operation versions and native
images remain identical.

For AUTO NuGet versus same-host source, all fourteen final outputs differ in
bytes. Five exceed the bound: SD15 square (2 values), CFG 2/3 Separate (7),
CFG 2/3 ConcatenateCompatible (43), CFG 3/3 Separate (20), and CFG 3/3
ConcatenateCompatible (38). The largest captured absolute difference is
`0.00030618906021118164`, in CFG 2/3 ConcatenateCompatible. The first unequal
U-Net coarse boundaries range from down0 to down3; timeEmbedding is exact in
all eight cases. These boundaries do not identify an exact differing primitive.

## Native identity and evidence limits

The host reports AMD EPYC 7763. Source records Python 3.12.10, PyTorch
2.10.0+cpu, NumPy 2.2.6, einops 0.8.1 and threads 1/1. Global source capability
is AVX2 under AUTO and DEFAULT under the explicit request. Product records
TorchSharp 0.107 and declared libtorch 2.10; actual loaded hashes supply the
binary identity evidence. Global capability is not a per-operator dispatch trace.

All **84 product loaded-map snapshots** contain exactly the selected c10,
torch_cpu, torch, unchanged LibTorchSharp bridge and OpenMP images, stable
within each process. TorchSharp.dll is verified separately as managed. Both
source maps match the wheel inventory, including the source Python binding.
No duplicate core, mixed OpenMP or Python image occurs in product maps.

The original and NuGet-copy inventories each contain 121 files; wheel-copy has
122. Only three cores and OpenMP change, with one additional internal hardlink
for the OpenMP SONAME. The managed assemblies, bridge and remaining files are
unchanged. ELF dependency closure, alias observations and all six before/after
inventories pass. The validated Python-library baseline environment is preserved;
no LD_PRELOAD or wheel path is added to the loader environment.

The three artifacts contain 95 JSON files, six TRX files and eight logs. The
independent stdlib audit reconstructed and rehashed **1,072 F32 payload records**
(688 captures and 384 inputs), separately rehashed fourteen accepted outputs,
and recomputed 946 capture contrasts plus 528 input contrasts. Six source blobs
and eight selected AST groups were independently verified, including the exact
existing `denoise`/`separate` adapter declarations. Six pinned helpers, five new
diagnostic scripts and six accepted source scripts match the committed blobs.
The 29-file fixture inventory and raw Linux lock remain unchanged before/after;
the lock is 3,280 bytes, SHA-256
`84df56cd98339e8dfec9b6f765312758706476ed1328d5f95f6a06433b9bb719`.

[Machine-readable evidence](sd-native-aligned-996f8a3.json) preserves all metrics,
identities, inventories, case controls and extracted-file hashes. Native payloads
and inode relationships were observed by CI but not uploaded for independent
local rehashing; archive digests are GitHub-reported. The eight child processes
took about 136 seconds in total. No additional native run was performed for
the audit.

This supports a same-host **native-bundle contrast** within the measured scope.
Three core libraries and OpenMP change together; it does not isolate a primitive,
compiler flag or individual library, establish stock/pretrained behavior, or
select a qualified production backend.
