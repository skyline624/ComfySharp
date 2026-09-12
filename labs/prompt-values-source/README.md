# Prompt values — prospective source laboratory

This laboratory observes **28 fixed cases**, each in three fresh source namespaces. It compares no C# output and makes no parity claim. Its protocol must be published and reviewed before the first source execution.

The four observations are deliberately separate:

1. The exact source `is_link` predicate on the original JSON value, with missing keys explicitly distinguished from null.
2. The exact source `get_input_data` on a fresh original prompt/cache fixture, without validation.
3. The exact source `validate_prompt` / `validate_inputs` on a fresh copy, including the resulting prompt mutations and rejection diagnostics. A real source `PreviewAny` selects the output; its `main` method is never called.
4. The exact source `get_input_data` on the normalized prompt, only after source validation accepts it. A rejected validation leaves a visible `not-run` dependent observation, not an accepted case or a skipped test.

Three wrapper cases additionally validate the already-mutated prompt again, with a fresh validation cache. This second pass is labeled separately and never replaces the first pass. Acquisition receives its own copy; the original protocol prompt and fixture cache must remain unchanged.

## Why validation and acquisition are separate

The pinned backend reserves every raw list for linked-input validation. `get_input_data` instead uses `is_link`, which requires exactly two items, a string node ID and an int/float slot. Thus direct acquisition can treat `[]` as a literal even though prompt validation rejects it. Python bool slots satisfy `isinstance(slot, int)`; float slots satisfy the predicate but cannot index the source return-type tuple. These observations must not be collapsed into a single acceptance rule.

The source validator already supports the `__value__` envelope. It unwraps once and mutates the prompt, while direct acquisition does not unwrap. A wildcard envelope containing a link-shaped list can therefore become an execution link after validation. This laboratory measures that boundary without changing product semantics or claiming that all existing C# envelope guarantees match it.

The cases cover list shape, string/numeric/bool node IDs, bool/float/negative/out-of-range slots, missing nodes, null/maps, reserved-key and nested envelopes, a missing required input, an explicit optional null, and four small scalar conversions. They do not cover custom validators, rawLink, arbitrary numeric parsing, non-finite values, model families, hidden execution contexts, the frontend or an HTTP server. One real output per synthetic producer means a false slot accesses zero; the negative slot accesses the same sole output. Numeric node IDs in link arrays are not aliases for JSON object keys.

## Exact source and infrastructure

The backend is pinned by commit, full-blob SHA256, exact AST hash and source-segment hash in `protocol.json`. The namespace builder and existing 72 declaration selections come from the published Autogrow laboratory at its separately pinned commit; all three helper files must match their raw Git bytes. That laboratory is unchanged. Its module builder runs only after this new laboratory passes its own publication gates.

Additional exact declarations include both validators, `full_type_name`, `validate_node_input`, the scalar primitives, the real legacy `PreviewAny` class, and the V3 Float/Combo classes and numeric enums. `Combo` is needed because source validation reads its type ID even when the current value is not a combo. Annotation-only types outside these reached paths are not fabricated. Real `Schema`, `ComfyNode`, primitive and CreateList constructors come from the pinned source closure.

There is no fake Schema, validator or Tensor. The fixture registry contains only the real source classes. The cache is explicit infrastructure: its public JSON values represent pre-existing outputs and are not obtained by running producer nodes. Dependency cache reads are recorded. The existing source mapper declarations are present in the reused closure but custom validation is forbidden by an execution-time guard, and no node compute method may run.

An unchanged-code profile guard rejects calls to every registered node's `execute`/`main` code object and remembers an attempted call even if source validation catches the exception. In particular, the exact PreviewAny class is created without a `torch` binding, and its Torch-using `main` must remain unreachable. No Torch import, fake Torch module or replacement print implementation is provided. All execution observations concern validation and input acquisition only.

## Commands and gates

Use Python **3.12.10**, stdlib only. The default command reads Git blobs, hashes files, parses and compiles selected AST nodes; it performs **no exec, upstream import, constructor call, validation or acquisition**:

```powershell
python -I -S -B labs/prompt-values-source/reference.py --source <frozen-backend-repository>
```

Only after review and publication of the exact three new files, the coordinator may run:

```powershell
python -I -S -B labs/prompt-values-source/reference.py --source <frozen-backend-repository> --execute --expected-commit <published-HEAD-SHA> --output <new-absolute-external-directory>
```

Execution checks the explicit HEAD, raw UTF-8 LF bytes of all three new files against that commit, exact helper Git identities, backend blob/AST hashes, exact Python version, enabled assertions and absence of a conflicting profiler/Torch module. The output must be a new directory outside both repositories and the interpreter directory; ancestor overlaps are also rejected. HEAD, all new files, helper identities and source hashes are checked again before a final reference is written. No download, package installation or product build is performed.

Successful execution writes a new `reference.json`, bounded to 16 MiB, and prints only its case count, size and hash. Failure writes an `incomplete.json` marker when execution has already created its destination, and produces no accepted qualification result. There is no fallback that turns an unexpected missing dependency into a source result. Only per-case prospective acquisition errors and validation exception types are admitted as observations; these allowlists do not require an error or turn it into acceptance. In particular, NameError/AttributeError from missing source dependencies abort collection even if the source validator caught them. Namespace, assertion and serialization failures also abort collection.

## Public encoding and interpretation

Every observed value has an explicit type: null, bool, decimal int, float64 with exact little-endian bits, string, ordered dict items, list or tuple. This prevents bool/int/float and empty/missing distinctions from disappearing through JSON equality. The original protocol bytes remain separately pinned. There is no numeric tolerance. Source error traceback arrays are replaced only by an explicit omitted-frame-count object; stack text is not a comparison target. Other synthetic values and error messages remain observable. The writer scans decoded strings for protected local paths, so JSON backslash escaping cannot hide them.

Three fresh observations must have identical canonical hashes before the case is recorded. The collector does not call source node compute functions, does not execute the frontend and does not exercise HTTP transport. A future comparator must report phase-specific results and declared prompt adaptations. It must not reinterpret direct acquisition as HTTP acceptance, treat a validation exception as a successful literal, or manufacture expected output from C#.
