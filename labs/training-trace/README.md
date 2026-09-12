# Native training candidate comparison

Protocol v4 prepares native files and notices directly from the pinned archive
using the offline [C# bundle tool](../../docs/NATIVE_BUNDLES.md), then composes the
candidate with that tool. Python remains confined to independent source collection
and comparison. The laboratory verifies the composed inventory, actual loaded
images and unchanged originals. Native aliases have identical verified bytes;
the product tool does not require hard links.

The current workflow compares the Linux NuGet build, an unchanged independent
copy, and a copy using the pinned source laboratory's native CPU core/OpenMP
libraries. It reuses the established ELF, copy, inventory and loaded-library
controls. Only runner temporary copies are modified; no Python interpreter or
binding is loaded into .NET. The protocol now pins these reused controls too.

The identity bridge is built with CMake and the matching pinned source SDK.
Candidate CPU dispatch and build configuration must match the source's actual
reports. The ordinary suite now admits added tests above its established 865-test
baseline and requires exactly the same executed count for both bundles, without
skips. Historical reports retain their original counts.

`COMFYSHARP_TRAINING_TRACE_COMPLETE=1`, with capture enabled, defers numerical
comparison assertions until both updates finish. Every failed assertion is
retained and the original test still fails. Other exceptions, initialization
checks and nonfinite gradients still stop execution. Captures now include losses
and selected updated parameters. `COMFYSHARP_TRAINING_NATIVE_IDENTITY=1` records
actual loaded native hashes. These are test controls, not product options.

`native_candidate.py` requires identical records from the original and copied
NuGet builds before attributing native differences. It compares both complete
SD1/SD2 traces to the independently observed same-host source, and also runs the
ordinary 865-test inference suite against original and candidate builds without
observers. Original TRX failures are retained. The experiment gates complete
same-host training comparisons, not success against every immutable cross-host
oracle or qualification of the whole runtime distribution.

`compare.py --require-complete --require-tolerance` rejects missing captures,
incomplete runs, nonfinite values, changed base weights and deviations beyond the
existing bounds. Losses use absolute `3e-5`; tensors use `3e-5 + 3e-5*abs(ref)`.
Five stdlib-only laboratory tests exercise these failure gates. The original
distributed source fixtures are never overridden at runtime.

Local Windows controls passed three tests with and without complete capture;
both complete traces, including losses and updates, were bit-identical to source.
This does not qualify the Linux candidate or the pretrained families.

## Earlier localization experiments

This diagnostic targets the two numerical failures in the all-target adapter
checks after exact initializer/RNG parity was established. It does not replace
the committed source corpus or change numerical acceptance thresholds.

`source.py` first verifies the pinned original training laboratory and these
additional diagnostic inputs. It executes the unmodified adapter collector twice
in separate processes: without observation, then with hooks on the same coarse
U-Net boundaries exposed by the C# diagnostic observer. The complete output
corpus and traversal order must remain byte-identical between those source runs.
Source hooks capture managed values; original forward/backward/update math remains
in the frozen collector and upstream definitions.

The existing .NET tests optionally capture base weight hashes, input values,
intermediate tensors and compared gradients. Ordinary execution does not capture
anything. `COMFYSHARP_TRAINING_TRACE_DIR` enables the observer; the accompanying
`COMFYSHARP_TRAINING_INTEROP_ONE=1` requests the existing test runtime's one-thread
interop setting solely for a separate controlled process. It is not an application
option or an established fix. A trace is written even when an original assertion
fails, preserving the point where execution stopped.

The earlier Linux workflow ran the original checks, observed checks, and observed
one-interop-thread checks separately. It continues after a failing diagnostic
variant only to collect the other controls and upload artifacts; that job
explicitly remained failed when any reference check failed. Mixed-file contracts
ran independently so the earlier gradient failure did not hide their results.

`compare.py` reports unchanged base weights, first differing captured boundary,
maximum absolute differences and counts outside the original `3e-5 + 3e-5*abs(ref)`
bound. It never writes expected tensors. Missing later captures indicate that
execution stopped early, not that those operations passed.

Local Windows checks passed in both observed modes (three existing tests each).
All captured inputs, coarse intermediates, compared gradients and base weights
were exactly equal to the independently observed source for both topologies;
changing the interop pool from its default 64 to one made no difference there.
This local control is not evidence of the cause of the Linux failures. No models,
new native dependencies, or product inference code are included in this change.

The [first Linux campaign](../../docs/qualification/training-coarse-diagnostic.json)
located the first differing coarse boundary at `down3` for both topologies.
Changing interop from four threads to one left every captured .NET value and
both failure verdicts unchanged. All 12 independently run mixed-file/alias tests
passed. The original source corpus reported AVX-512 dispatch; the source on this
runner reported AVX2, and its output corpus differed. A same-runner gradient
discrepancy still exists, so that hardware difference is not the complete diagnosis.

The follow-up enables the already existing detailed observer for the last
downsample and its two residual blocks. No product operation is modified. The
comparator restores Float32 values from shortest JSON decimals before computing
differences, so serialization text precision is not counted as a numerical error.
