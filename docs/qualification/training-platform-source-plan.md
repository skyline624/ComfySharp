# Prospective same-platform training source collection

The training reference suites currently embed Windows CPU source results. CI at
`6fd5c3d` failed its two all-target initialization cases on both Linux and macOS,
at the first up-factor hash, while four other adapter tests passed. The expected
hash begins `81cc7594`; the observed hash on both platforms begins `25e8628c`.
The preceding generator-state check passed. The earlier dataset noise and shared-
sigma denoising discrepancies are documented separately; these observations alone
do not establish their native root cause.

The [source protocol](../../labs/training-platform-source/protocol.json) freezes
the existing recipes, source commit, exact batch/initialization assertions and
`3e-5` absolute plus relative gradient bounds before collecting on other platforms.
Only the collector's accepted CPU version-string spelling and one-thread inter-op
configuration were adjusted. No expected .NET result supplies an oracle.

A local Windows execution passed source/input preflight and reproduced all three
existing corpus files byte for byte:

| Corpus | SHA-256 |
|---|---|
| Batches | `c4d0696847ab0b770e6068ab9119852d57dabfafb9dcc9a826a603cadfa140f0` |
| Denoising | `6a014fc429a878fd4d06ef2832fae574810637c1c01ae4096f2bda9303aa4e0a` |
| Adapters | `ae390cc8351a5070c3355f6c0bc788f4a2309c66b0d0af255ff0bbc738bf0a0a` |

The new source workflow collects the same cases on all three CPU platforms with
the existing hash-locked environments. Its output requires provenance review and
same-platform .NET comparison before it can become acceptance evidence. Existing
tracked fixtures and product tests are unchanged by this collection milestone.
No GPU, complete training node or model family is qualified by source collection.
