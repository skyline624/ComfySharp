# Prospective Autogrow.TemplateNames source laboratory

This is an independent, stdlib-only source collector. The protocol and collector must be reviewed and published before the first `--execute`. No expected values are produced by preflight, copied from C#, or supplied in the protocol. The laboratory does not run as part of normal .NET tests.

The frozen backend is [ComfyUI 1d48d9c](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a). The exact source closure is ten Git blobs and 73 AST declarations, each pinned by raw bytes, SHA-256, AST hash, source segment hash and starting line in `protocol.json`. It reuses the unchanged [Autogrow laboratory published at ae241a0](https://github.com/skyline624/ComfySharp/tree/ae241a0ae32f4e053f0bc5754e8d4d0ebfc7d5f4/labs/autogrow-source), whose three raw files and publication commit are verified before importing its inert driver definitions. Preflight parses and compiles source AST but never executes those declarations.

## Source boundary

The real [_AutogrowTemplate and TemplateNames](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_api/latest/_io.py#L1087), Schema, AnyType, MatchType, WidgetInput, ComfyNode and NodeOutput are used. Actual source finalization, input acquisition, list mapping, blocker callback, result merging and `build_nested_inputs` perform their operations unchanged. The inherited 72-declaration closure also includes CreateList, but this collector never invokes its body. The additional declaration is the whole real [StringFormat class](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_string.py#L9), supplied with real `io` and Python's real `string` module.

There is no general upstream import, Torch import, fake Schema, Tensor type, or substitute dynamic-input algorithm. The inherited closure supplies actual stdlib types/decorators; unused tensor, model, dynamic-combo/slot, HTTP and executor dependencies remain outside the selected declarations or unreachable type annotations. Python 3.12.10 with assertions enabled is required for execution.

For 34 cases, `FixtureNames` is an explicitly synthetic subclass of the real ComfyNode. Its real Schema contains real TemplateNames instances. Its transparent body is exactly `execute(**kwargs): return io.NodeOutput(kwargs)`. This body carries the arguments for inspection; it is not an upstream node implementation, formatter implementation, or evidence that an arbitrary typed node would accept those arguments. It intentionally permits the unknown-link diagnostic to expose an ungrouped root argument. Two separately classified cases invoke the real StringFormat body; their formatting results are source-only and are not port acceptance requirements.

`GET_SCHEMA()` runs before assigning `RELATIVE_PYTHON_MODULE`, matching the pinned loader's ordering. The initial schema module and subsequent object-info `python_module` are separate source fields. The source class is attributed to `comfy_extras.nodes_string`; the fixture class is explicitly attributed to `laboratory.autogrow_names`.

## Prospective corpus and comparison

The 36 named cases comprise 25 comparable structural cases, six unsupported-template observations, one constructor-error case, two source-only StringFormat cases and two unsupported-input observations. These classifications delimit the planned port; they are not results or claims that a source invocation succeeded.

- Explicit names and prompt orders differ; scalar mapping uses unequal execution-list lengths and repeat-last, while InputIsList is tested separately. Two groups and ordinary inputs expose root and member ordering independently.
- Empty, one, 100 and 101 names cover the true `names[:100]` behavior. Truncation precedes validation of effective names. The ambiguous 101st name is discarded, and both truncation cases are comparable because their supplied inputs lie within the first 100. No derived `max` or `prefix` is invented in metadata.
- Minimum zero, equal to count and above count, optional prototypes, real wildcard MatchType, caller-list mutation after construction, omitted optional values and explicit null are separate cases. The fixture mutates only its private caller-list copy, never the protocol.
- Blockers with nonempty, empty and null messages, first-blocker prompt order, InputIsList checks before grouping, a nested marker, and scalar slicing of an empty execution list are covered. The empty-list mapper IndexError is prospectively permitted. The nested-dynamic prototype and negative minimum permit the source constructor's AssertionError. Unexpected exceptions fail collection.
- Duplicate, dotted and empty effective names, an optional outer group, widget prototype and dynamic prototype remain unsupported-template observations. Unknown literals and links remain unsupported-input observations; source acquisition is not replaced by the port's stricter policy.

Comparison is exact over encoded values and the explicit order arrays. Metadata, required/optional partitions, dynamic paths, acquired flat inputs, nested arguments, cache reads, source callback reports and merged results are retained. Dictionary equality alone does not prove order: `finalizedOrder`, `dynamicPathOrder`, `flatOrder`, `cachedInputOrder`, and observed root/member orders carry it explicitly. Input/cache snapshots must remain unchanged.

Each case runs in fresh source namespaces **off/on/off**. Only the middle run uses `sys.setprofile` to observe call arguments and return values of the real binder and selected body code objects. It does not rewrite source or reconstruct binder results. Complete unobserved result encodings must have identical SHA-256 in all three runs after removing only the observation field. The reference retains the middle result plus all three hashes; a later file audit can recompute the middle hash, while off-run hash evidence comes from the collector's comparison. An existing profiler is refused.

This boundary starts at source finalization and `get_input_data`, not full prompt validation. Missing required leaves, literal arrays and unknown inputs here do not establish HTTP acceptance. The upstream `__value__` wrapper also exists outside this acquisition boundary; this laboratory neither qualifies nor rejects its HTTP behavior. No lazy/async task lifecycle, interruption, frontend, memory lifetime or whole-node parity is claimed.

## Admission and commands

Use the existing Python 3.12.10 interpreter; no dependencies need installation. Supply the local frozen backend Git repository. Preflight is the default:

```powershell
python -I -S -B labs/autogrow-names-source/reference.py --source <backend-git-repository>
```

After review and publication of all three exact LF files, the first source collection may be explicitly authorized:

```powershell
python -I -S -B labs/autogrow-names-source/reference.py --source <backend-git-repository> --execute --output <new-absolute-external-directory>
```

The output must be absent and outside both repositories after resolution. `--output` without `--execute` is rejected. Before source execution, the collector admits one repository HEAD and requires every current laboratory file to equal its raw Git blob at that commit. Old helper files must equal their pinned published blobs. After calculation, the HEAD, current laboratory files, helper files, all ten raw source blobs and all 73 AST identities are rechecked before the exclusive creation of `reference.json`. Existing results and accepted fixtures are never overwritten. Output creation before the run may leave an empty directory on failure; failure does not produce a reference result.

The collected provenance includes the collector commit, raw file identities, source evidence, Python/platform, case classifications, exact source errors and observer neutrality. The collector does not download sources, modify repositories, install packages, execute .NET, or import C# output. A failed first run is a diagnostic requiring review, not permission to change the protocol or silently substitute an implementation.
