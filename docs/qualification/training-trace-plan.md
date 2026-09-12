# Training divergence diagnostic controls

The [mixed-adapter revision ec02f23](https://github.com/skyline624/ComfySharp/commit/ec02f23a3d7c7b0f24ec88c55965f8301262a261)
completed [CI 34712941387](https://github.com/skyline624/ComfySharp/actions/runs/34712941387):
Windows succeeded; Linux again failed all-target numerical adapter checks;
macOS again failed only the separate full CLIP comparison. Thus the earlier
Windows global tensor-count failure did not recur in this run; its cause is not
established and it is not being hidden by an assertion change.

The next [diagnostic workflow](../../.github/workflows/training-trace.yml) preserves
the original reference verdicts while comparing source/.NET base weights,
inputs, intermediates and gradients on Linux. The source collector and tolerance
profile remain unchanged. The optional one-interop-thread control investigates a
measured configuration difference without assuming it is causal.

The [protocol](../../labs/training-trace/protocol.json) pins the added collectors,
workflow and test observer before collection. Local Windows source collection
with and without observation produced byte-identical original corpora. All base
weight hashes and captured .NET tensors match the source exactly in both default
and one-interop-thread controls; each local control passed its three existing tests.
Linux outcomes will be recorded from the workflow artifacts after completion.

This is evidence collection toward resolving required numerical parity, not a
new model-family qualification, a relaxed comparison or a claim that CI is fixed.
