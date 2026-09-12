# Linux training divergence observation

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

The Linux workflow runs the original checks, observed checks, and observed
one-interop-thread checks separately. It continues after a failing diagnostic
variant only to collect the other controls and upload artifacts; the final job
explicitly remains failed when any reference check failed. Mixed-file contracts
run independently so the earlier gradient failure does not hide their results.

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
