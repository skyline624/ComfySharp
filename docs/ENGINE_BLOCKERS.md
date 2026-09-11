# Execution blockers in the static DAG engine

An execution blocker is a typed control value that prevents a dependent node
invocation from running. It is distinct from JSON null, a thrown exception and
cancellation. This implementation covers the existing static DAG engine,
execution-list mapping and aggregate lazy-input contract. It does not implement
graph expansion, persistent caches or the full ComfyUI scheduler.

## Returning and receiving a blocker

A runtime node can block one output slot through its invocation context:

```csharp
return new NodeExecutionOutput([context.Blocker(), usableValue]);
```

`context.Blocker()` has a null message and blocks silently. A non-null message,
including an empty string, is reported when a consumer encounters the value.
To block every declared output slot, use:

```csharp
return NodeExecutionOutput.Blocked("This result is unavailable.");
```

The whole-call form also accepts explicit producer UI. Its `BlockExecution`
property takes precedence over ordinary `Result` slots. Existing
`NodeExecutionOutput(Result, Ui)` construction and two-value deconstruction
remain supported. A whole-call block works with zero declared outputs.

Typed results expose `RuntimeValueKind.Blocker` and `value.Blocker.Message`.
Blockers follow the same lease rules as other runtime values: inputs are
borrowed, context-created values belong to the invocation, and retained output
leases survive that scope. Retaining a blocker or a containing list/map creates
an independent lease. Disposing one does not invalidate another retained lease.

## Mapping, lists and lazy inputs

The engine checks blockers after slicing inputs with the existing repeat-last
rule and before calling the node. It examines direct argument values in the
prompt's input order; the first blocker wins. A silent first blocker therefore
suppresses a later message-bearing blocker for that invocation.

An ordinary execution list can mix usable values and blockers. Only blocked
positions skip their consumer. A direct blocker at an `IsList` output occupies
one merged position; other returned execution lists are flattened as before.
For an `InputIsList` node, a blocker among the direct members of any incoming
execution list skips the entire call. The engine does not recursively search
nested lists or maps. JSON objects resembling a blocker remain ordinary data.
The existing empty-list rules and errors still apply.

Lazy hooks receive no blocked rows. For mixed mapped inputs, the engine first
expands repeat-last slices jointly, then selects the active rows for the
aggregate `GetRequiredLazyInputs` hook. This preserves pairing between inputs.
An entirely blocked call skips the hook and resolves no lazy dependency.
Messages are not consumed during lazy selection. After required lazy inputs
arrive, the engine checks all actual arguments again before execution.

The hook still returns the union of lazy names needed by its supplied rows.
There is no new per-row dependency scheduler or asynchronous/iterative lazy
hook contract. A lazy input requested by one row becomes available to the node's
whole mapped invocation set under the existing behavior.

## Messages, errors and output boundaries

When a consumer encounters a message-bearing blocker, it emits an
`execution_error` event with `Execution Blocked: {message}` and records an
`execution_blocked` diagnostic identifying that consumer. It produces fresh
silent blockers for its output slots. The original producer's memoized marker
remains unchanged, so separate first consumers can each report it. Merely
selecting the producer as a raw target does not consume its marker.

A block is successful control flow: it does not fail independent targets or
change an otherwise completed typed/UI job to status `error`. Ordinary node
exceptions still produce execution failures. Cancellation checks remain active
on skipped calls, including after an awaited diagnostic event. Transport or
resource-disposal exceptions retain the engine's existing behavior.

Skipped invocations contribute no UI. Real producer UI and surviving mapped UI
returns retain the existing snapshot and merge rules. `ExecuteUiAsync` releases
runtime values before its terminal event as usual.

`ExecuteValuesAsync` and `ExecuteUiAsync` are the blocker-aware boundaries.
`RuntimeValue.ToJson()` explicitly rejects a blocker, including one nested in a
container. Consequently `ExecuteAsync` reports `runtime_value_not_json` if a
selected target's raw output contains one. It never silently turns a blocker
into null or a magic JSON object. This is the C# raw-JSON projection policy;
it is not a claim that upstream exposes its runtime slots in that form.

## Source and verification scope

The behavior is based on frozen ComfyUI commit
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`:

- [Marker definition](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_execution/graph_utils.py#L140-L155).
- [Mapped input admission and result merging](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L243-L404).
- [Lazy checks and message consumption](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/execution.py#L503-L545).
- [Upstream blocked-list regression](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_execution.py#L648-L672)
  and [asynchronous producer regression](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/tests/execution/test_async_nodes.py#L286-L314).

The added managed tests cover propagation, messages, prompt order, list
positions, lazy pairing, UI, cooperative cancellation, exceptions and counting
resource ownership. They contain no tensors, model weights or expected output
derived from a new source run. Their build/execution results must be reported
separately; this document does not assert they have passed or establish full
source parity. Graph expansion, V3 conversion quirks, custom validation hooks,
`IS_CHANGED` and persistent caches remain outside this implementation.
