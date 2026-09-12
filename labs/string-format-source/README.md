# Prospective StringFormat source laboratory

This laboratory must be reviewed and published before its first source execution. Preflight is the default and produces no expected results. The collector invokes the **real frozen StringFormat class and Python's builtin `str.format`**, not a fixture body, `string.Formatter`, C# output or a replacement formatting algorithm.

The protocol `string-format-text-v1` describes a prospective partial product profile. A later C# node may preserve the source class_type and input/output schema while documenting its limited formatting surface. This laboratory itself registers no product node and makes no C# qualification claim.

## Exact source and helpers

Backend commit: [1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a). The ten raw source blobs and 73 declarations are listed in `protocol.json`, with byte count, SHA-256, AST hash, source-segment hash and line. The true [StringFormat declaration](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_string.py#L9) calls `f_string.format(**values)` without positional arguments. Its real Schema, TemplateNames, AnyType, ComfyNode, NodeOutput, finalization, acquisition, mapper, binder and return merging are used unchanged.

The collector reuses the published Names driver at `4312bdafabe2906b74bc51cdca678aa26aeec5ee` and its Prefix helper at `ae241a0ae32f4e053f0bc5754e8d4d0ebfc7d5f4`. All six helper files are verified against their raw pins and published Git blobs **before importing driver definitions**. Their source execution functions are called only after admission. The old fixture classes and CreateList body are not invoked. The source modules receive real stdlib objects and actual source classes; no Torch type, Schema constructor, builtin formatter or source body is mocked. Only cache/server infrastructure is adapted explicitly.

The Python interpreter for source collection is **3.12.10 with assertions enabled**. No packages need installation. No general upstream module imports or native model dependencies are reached. The builtin formatter belongs to this interpreter, not to a copied Python approximation. The [Python formatting documentation](https://docs.python.org/3.12/library/string.html#format-string-syntax) is explanatory; the prospective interpreter version and exact source call define the collected results.

## Forty-four cases, with three separate partitions

| Partition | Cases | Meaning |
|---|---:|---|
| `comparable` | 24 | Prospective successful text-profile comparisons; no C# outputs have yet been compared |
| `source-error` | 12 | Exact source phase/type admitted before calculation; error message is observed, not made a C# equality requirement |
| `unsupported-profile` | 8 | Source behavior remains recorded, but successful source results are not successful port expectations |

The 24 successful cases cover six basic substitutions/braces/mapping cases, ten string specifications and eight null/bool/Int64 cases. The specification cases distinguish default/left/right alignment, odd centered padding, an astral fill, astral text precision, combining code points, zero precision, explicit `0>6` fill and a single zero width. The scalar cases distinguish empty formatting from `!s` followed by alignment/precision. Integers at both Int64 boundaries are JSON integers and never round-tripped through a float. An unused float value confirms that acquisition stays eager while only referenced arguments require formatting.

The error cases cover unmatched opening/closing braces, missing named/automatic/explicit positional arguments, malformed conversion, an earlier missing field before later malformed syntax, and an unknown conversion before a nested spec's missing argument. Four further cases fix priority within the current field: missing name before unknown conversion, missing name before a product-unsupported spec, malformed conversion syntax before missing-name lookup, and an unclosed field before lookup. These distinguish structural field parsing from later semantic conversion/formatting. The source phase is `mapper`, where the real body is invoked. Their prescribed categories are ValueError, KeyError or IndexError. A different exception or unexpected successful return aborts collection; it does not manufacture an expected result.

The unsupported cases observe repr/ascii, dictionary indexing, integer attribute access, nested width, float formatting, container rendering, integer numeric formatting and leading-zero width `05`. The combined repr/ascii case observes both conversions in the source; a product refusing at the first conversion still needs a separate product test for its `!a` refusal. In particular, the valid Python leading-zero form must not be reported as invalid Python syntax merely because the first product profile excludes it.

The future profile is bounded to valid Unicode text, ordinary named fields, escaped braces, explicit text width/alignment/precision and `!s`; null/bool/Int64 support empty formatting, or text formatting after explicit conversion. Unicode lengths are code points, not UTF-16 units or graphemes. No reflection, attribute/item access, nesting, float/container/native conversion or numeric-spec fallback is authorized by these observations. Invalid UTF-16, cancellation, native ownership and small injected resource-budget tests belong to product-only checks, not this JSON-domain source corpus.

## Observation and comparison

Each case uses fresh source namespaces in **off/on/off** order. The middle run's `sys.setprofile` observes calls to the actual StringFormat code object and returns from the actual `build_nested_inputs` code object. It records `f_string`, grouped values and their member order from named frame locals; the binder's actual root argument order is a separate record. It never intercepts or substitutes the builtin formatting calculation. An existing profiler is refused.

All three complete result encodings must have identical SHA-256 after removing only the middle observation field. The artifact retains the middle result and all three hashes. A later auditor can recompute the retained middle hash; off-run equality remains the collector's evidence because those raw runs are not duplicated in the artifact. Input and cache snapshots are compared via exact canonical JSON before/after, preserving integer/bool/float spelling distinctions and dictionary order. All output strings and source exception types/messages are preserved without tolerance or normalization.

Object-info captures retain the real source description, including its full Python capability statement; this is **source metadata**, not a description of a future partial product implementation. Source loader ordering calls GET_SCHEMA before assigning `RELATIVE_PYTHON_MODULE`; source `python_module` is `comfy_extras.nodes_string`.

The result has no HTTP/PromptExecutor/global-required-validation claim. The fixture cache supplies actual input values to the source mapper; its producer nodes are not executed. Positional arguments remain absent. There is no C# result generation, accepted fixture import, native operation, dependency installation, source download or fallback on a failed source case.

## Commands and publication gate

Before publication, only preflight is authorized:

```powershell
python -I -S -B labs/string-format-source/reference.py --source <backend-git-repository>
```

After review and publication of the exact three LF files, root may authorize the first collection:

```powershell
python -I -S -B labs/string-format-source/reference.py --source <backend-git-repository> --execute --output <new-absolute-external-directory>
```

Preflight verifies the prospective protocol, all helper pins and source identities, then parses/compiles AST without executing source declarations. `--output` requires `--execute`. Execution admits a single collector repository HEAD and requires all three local laboratory files to match its raw committed blobs. The destination must not exist and must resolve outside both repositories. Python version/assertions and all provenance checks precede the first source call.

After calculation, the collector rechecks HEAD, laboratory files, all six helper files and all raw/AST source identities before writing `reference.json` exclusively. Existing files, accepted fixtures and old laboratories are never overwritten. A failure may leave an empty output directory, but cannot produce a reference file. The output records the publication commit, platform/interpreter, profiles, source/helper pins, exact outputs/errors and observation neutrality. Failures require a reviewed prospective follow-up; they do not authorize silently changing the protocol or parser.
