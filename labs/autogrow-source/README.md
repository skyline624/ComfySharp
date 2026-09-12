# Prospective Autogrow / CreateList source laboratory

This laboratory records exact, bounded source behavior from ComfyUI
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. The protocol is fixed **before the
first execution**. It contains no expected C# outputs and is not imported by the
product or its tests. Publishing these three files is required before collection.

Use Python **3.12.10**, with the standard library only. Preflight reads canonical
Git blobs, checks their raw SHA-256 and all selected AST/segment identities, and
compiles the selected declarations without executing them:

```sh
python -I -S -B labs/autogrow-source/reference.py --source /path/to/frozen/ComfyUI
```

After independent review and publication, the first source execution is explicit:

```sh
python -I -S -B labs/autogrow-source/reference.py --source /path/to/frozen/ComfyUI --execute --output /new/absolute/external/directory
```

The destination must be new and outside both repository trees. No source or model
is downloaded. The collector admits one Git HEAD, checks the three current lab
files against that commit, then checks HEAD/files and the frozen source identities
again before writing `reference.json` exclusively. Failed admission or unexpected
source errors do not produce a reference. No fixture or accepted profile is written.

## Exact closure, with bounded infrastructure

The protocol pins **nine raw source blobs and 72 selected declarations/statements**.
Each selected class is complete, including nested methods, bases and decorators;
no body is rewritten. The actual closure includes:

- `_io.py`: the ComfyType decorator/base/input/output hierarchy, real widget and
  AnyType/MatchType classes, DynamicInput, complete Autogrow/DynamicCombo/DynamicSlot
  declarations and their actual registration code, Hidden/V3Data, Schema/NodeInfoV1,
  finalization, nested-input helpers, ComfyNode and NodeOutput.
- `comfy_api/internal`: actual class copying, override detection, properties,
  class cloning/locking and normalized method dispatch; internal bases are real.
- `comfy_extras/nodes_toolkit.py`: the unmodified CreateList class. Its loader
  assignment `RELATIVE_PYTHON_MODULE = "comfy_extras.nodes_toolkit"` is prepared
  explicitly after `GET_SCHEMA`, following [nodes.py:2329](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/nodes.py#L2329).
- The actual execution-list acquisition, input-info helper, V3 mapper, async
  resolver, merger and execution-block callback; actual ExecutionBlocker/is_link
  and CurrentNodeContext definitions. The real legacy typing declarations close
  the input-info helper's eagerly evaluated annotations.

General source modules are never imported. The retained `_io.py` future import
keeps its annotations deferred exactly as written. Tensor types, UI output
implementations, price-badge paths, API registry/extension startup and model
imports are not reached and are excluded. DynamicCombo and DynamicSlot are loaded
only because the real registration function names them; their behavior is not
qualified. Any unexpected dependency fails; no torch/Tensor, Schema, ComfyType or
ComfyNode stand-in is supplied.

Named namespace wiring supplies actual standard-library dependencies. The source
names `typing_extensions.final` and `typing_extensions.NotRequired` are bound to
Python 3.12's real `typing.final` and `typing.NotRequired` respectively; this is an
explicit annotation/decorator dependency adaptation. It does not replace source
constructors, strip decorators or defer annotations in files that lack a future
import. The generated temporary module objects support real dataclasses and are
removed after each case.

The only runtime infrastructure adapters are a fixture cache supplying declared
output-slot lists and a recording server for the real execution-block callback.
There are no fake producers, numerical operations or tensors. Generic TemplatePrefix
cases use a laboratory ComfyNode subclass returning a **real source Schema** with
real source constructors; its execution delegates to the unchanged CreateList
method. It is not presented as the original CreateList schema.

## Thirty-six prospective cases

Twenty cases exercise original CreateList: one/two/ten ports, holes and reversed
prompt order, empty execution lists versus literal empty-list items, nested items,
duplicate links, missing required inputs, four unknown-literal shapes, unknown
links with/without a cached value, message/silent/empty-message blockers, and a
nested marker that must not cause recursive blocking.

Sixteen cases exercise bounded TemplatePrefix constructors/finalization: min=0,
max=1/100, min above max, optional prototype versus optional group, real widget
forceInput, presentation options, a different prefix, invalid bounds, a real
dynamic prototype, and ordered/invalid/empty MatchType allowed types. The protocol
states the rationale and port classification per case.

The C# tranche intentionally diagnoses unknown dynamic input names early and
supports a bounded TemplatePrefix surface. Cases marked `unsupported-input`,
`unsupported-template`, `constructor-error` or `source-only` retain their actual
source observations; they are **not** automatically port-success expectations.
The optional group, widget prototype, unqualified custom extra_dict option and
non-wildcard allowed-types cases are explicitly `unsupported-template`, even when
the source constructor succeeds; their values and captures are retained unchanged.
In particular, source finalization/acquisition is not PromptExecutor's required
input validation. A missing `input0` can reach these helpers without establishing
that a complete upstream prompt would pass validation. Unknown literals may be
ignored while unknown links can survive as extra Python keyword arguments.

## Captures and comparison

Each case records original object_info and INPUT_TYPES, finalized required/optional
leaves, explicit key order, dynamic paths, acquired flat values/order, missing keys,
cache reads, outputs/UI or typed source errors, and unchanged prompt/cache inputs.
Exact values and explicit ordered-key lists are compared without numeric tolerance.
No property-order assumption is hidden in a generic unordered JSON comparison.

The middle execution records real `CreateList.execute` call arguments and actual
`build_nested_inputs` return values with `sys.setprofile`, selecting the exact
source code objects. No hook replaces either calculation. Fresh executions run
**off/on/off**; the full non-observer records must have identical canonical hashes
before a result is written. These observations distinguish prompt order for blocker
admission from template order for concatenation. An existing profiler is rejected.

Only cached fixture values decode a sole `$blocker` key into the actual source
marker. Prompt JSON remains ordinary data. This laboratory encoding is not a
product sentinel contract. Source exceptions for prospectively unsupported cases
are recorded with their stage/type/message; unexpected exceptions abort collection.

This qualifies neither complete PromptExecutor scheduling/validation nor the
frontend's automatic ports and MatchType propagation, dynamic graphs, persistent
caches, API nodes, models or full V1 compatibility. The chosen Desktop tranche has
ten fixed ports; this source laboratory does not test that UI.
