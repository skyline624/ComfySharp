# Prospective CaseConverter source laboratory

Review and publish these three UTF-8/LF files before the first `--execute`. The default preflight verifies raw identities and compiles exact AST declarations without executing them. No expected value is generated from C# code, a product table or an existing accepted fixture.

## Exact node and bounded closure

The actual [CaseConverter, lines 106–136](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_string.py#L106) belongs to frozen backend `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Its schema has required `string` STRING (multiline) and `mode` COMBO, in that order, and a STRING output. Choices are **UPPERCASE, lowercase, Capitalize, Title Case**. There is no declared source mode default. The unchanged body calls the corresponding `str.upper`, `str.lower`, `str.capitalize` or `str.title`. An unrecognized mode returns its input unchanged.

Ten source blobs and 74 exact declarations are pinned by raw bytes, SHA256, line, AST and source-segment hash. The real Schema, String, Combo, NodeOutput, finalization, acquisition, nested-input binder, mapper and execution-block callback are used. No Schema constructor, Tensor type or node body is replaced. General upstream modules are not imported.

The driver adapts the published TextComparison collector infrastructure at `5466f97312d8dbdbe5c656a6c31061895afc9af3`; all three origin files are pinned unchanged alongside the three Prefix helper files at `ae241a0ae32f4e053f0bc5754e8d4d0ebfc7d5f4`. The Prefix driver is imported only after all six files match their raw hashes and immutable Git blobs. The TextComparison origin is attested, not executed as a collector. The shared source closure still defines unused classes such as CreateList/Autogrow/Boolean: their presence does not imply their bodies are exercised. Real stdlib types/decorators supply the explicit dependency namespace. Only cache/server infrastructure is adapted.

The genuine source loader order is retained: GET_SCHEMA runs before setting RELATIVE_PYTHON_MODULE to `comfy_extras.nodes_string`. Initial schema-module state and loader module attribution are not conflated.

## Forty prospective node cases

The exact strings, typed values, order and links are in protocol.json. Its partition is fixed before collection:

- **28 comparable**: six scalar texts for each of the four modes, two genuine cache/link list mappings with repeat-last, one blocker with no body call, and one prompt whose `mode` precedes `string`.
- **Eight source errors**: null and integer input for each recognized mode. The body attempts the selected method and AttributeError at the mapper stage is prescribed. The actual message is observed. Engine STRING coercion and local typed rejection are distinct contracts, not claimed identical exceptions.
- **Four source-only invalid modes**: wrong-case or empty choice values, deliberately reaching the real body without global COMBO admission. Their identity fallback is observed separately from any future managed admission rejection.

The six scalar texts include empty, mixed ASCII with initial nonletters, apostrophes and digits; full expansions U+00DF/U+FB03/U+0130; titlecase digraphs; final/medial sigma with U+0345 and long runs of ignorables; and astral characters, NUL, newline and combining marks. The same fixed six texts are used in each mode. No output is guessed in the protocol. Scalar order and expansion behavior are obtained only from the real source execution.

Capitalize titlecases the first scalar rather than uppercasing it, then lowercases the rest with original full-string sigma context. Title uses the immediately preceding original scalar's Cased property; it does not skip Case_Ignorable for that state. Sigma context still skips ignorables. The code here implements neither algorithm: the real Python builtins perform them. No NFC, casefold, swapcase, word splitting or locale conversion is added.

Each case executes in fresh source namespaces **off/on/off**. The middle run's sys.setprofile observes actual body-call and binder-return code objects without intercepting their results. Body arguments are captured in source signature order; the binder's actual root order is recorded separately, including the reversed prompt-order case. The encoded prompt and cache must remain unchanged. Removing only the observation field must leave identical complete encodings in all three runs. The artifact retains the middle record and three neutral hashes, not duplicate raw off-runs. External profilers are refused.

This boundary does not execute upstream global HTTP, required-input or COMBO validation. It is not a general malformed-input oracle. Invalid CLR UTF-16, cancellation, ownership and resource exhaustion are local product concerns.

## Eighty-five builtin digest plans

Execution requires **CPython 3.12.10, Unicode 15.0.0**, and enabled assertions. Every valid scalar 0 through 0x10FFFF is visited in ascending order, excluding 0xD800–0xDFFF. Plane 0 has 63,488 entries; the other 16 planes have 65,536 each: **1,112,064 scalars**.

The five fixed recipes, applied independently for each scalar, are:

1. `chr(cp).upper()`
2. `chr(cp).capitalize()`
3. `('A' + chr(cp) + 'Σ').capitalize()`
4. `('A' + chr(cp) + 'Σ').title()`
5. `('AΣ' + chr(cp) + 'A').title()`

That is **17 planes × five recipes = 85 records**, **5,560,320 inputs per passage** and **16,680,960 builtin applications across three actual passages**. Count, input/output byte totals and hashes must match across the three traversals of every plan. Existing 68 lower digests remain a separate historical regression; this collector neither regenerates nor counts them.

Independently for input and output, each frame is `uint32LE(cp) || uint32LE(strict_UTF8_byte_length) || strict_UTF8_bytes`. SHA256 starts empty per plane/recipe. No BOM, delimiter, terminator, normalization or JSON-number round trip participates. The codepoint distinguishes equal outputs. The bounded artifact retains each count, byte totals, input hash and three output hashes, without a million-row output file. No product table, static extractor or private Unicode native entry point is read. Exhaustive scalars within these five fixed constructions do not prove every Unicode string or context.

## Admission, provenance and commands

Preflight only before publication:

```text
python -I -S -B labs/case-converter-source/reference.py --source <frozen-backend-git-repository>
```

After review, publication and root admission:

```text
python -I -S -B labs/case-converter-source/reference.py --source <frozen-backend-git-repository> --execute --output <new-absolute-external-directory>
```

All three local files must equal their raw blobs at one admitted collector HEAD before source declarations execute. The six helpers/origins and frozen source blobs are checked before calculation and again afterward, along with collector HEAD and the laboratory files. The output must be absent and resolve outside both repositories. Runtime version, Unicode version, assertions and executable identity are checked before creating it. Executable evidence includes basename, size and SHA256 only; a venv launcher hash does not attest the dynamically loaded interpreter DLL or all runtime libraries. No machine path is written into the artifact.

Only after all 40 cases, all 85 digest plans, repeats and post-execution provenance checks succeed is `reference.json` written exclusively. Failure can leave an empty directory but no partial reference. No old laboratory, accepted fixture, source file, lock or product resource is modified. This is source evidence only, with no C# or cross-platform qualification claimed before future comparisons.
