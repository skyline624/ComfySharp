# Prospective text comparison and Unicode lower source laboratory

Publish and review these three files before the first `--execute`. The default preflight verifies identities and compiles exact AST without executing source declarations. This laboratory reads no C# output, generated table, accepted fixture or native model dependency.

## Real node source

Backend [1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a) is pinned by raw blobs, line numbers, AST and source-segment hashes in protocol.json. The actual StringContains and StringCompare declarations call the interpreter's builtin lower and predicates. Their real Schema/String/Boolean/Combo, finalization, acquisition, binder, mapper and NodeOutput are used unchanged. No body or Schema constructor is replaced.

Three published Prefix helper files at `ae241a0ae32f4e053f0bc5754e8d4d0ebfc7d5f4` are verified against both raw hashes and Git blobs before importing the inert driver. Its source namespace supplies real stdlib typing/decorators; general upstream modules are never imported. The shared closure defines some classes not invoked here, including CreateList and Autogrow. They are pinned, and their presence does not imply those bodies are exercised. Only cache/server infrastructure is adapted explicitly. Module attribution follows real loader ordering: GET_SCHEMA first, then RELATIVE_PYTHON_MODULE.

The 40 prospective cases comprise **36 comparable, two prescribed mapper type errors, and two source-only invalid Compare modes**. Contains supplies 24 cases; Compare 16. Both case modes, all three comparison operations, empty strings, astral characters, NUL/newline, expansion of İ, final/medial sigma and U+0345 plus runs of ignorables are included. Cases distinguish lower from casefold and preserve composed/decomposed differences. Both complete arguments are lowered independently before comparison.

The invalid modes deliberately bypass global COMBO validation at this mapper boundary. The actual source body may return None and fail when merged; the protocol admits merge TypeError and records the observed outcome. These are not product success expectations. Source exception messages/types remain observations, not identical local diagnostics. This is not an HTTP, frontend or global required-input validation oracle. Invalid UTF-16, resource budgets, cancellation and ownership are separate product contracts.

Each case runs in fresh source namespaces off/on/off. The middle run uses sys.setprofile to observe real body arguments and real binder returns by code-object identity. It does not intercept the builtin calculation. Encoded prompt/cache snapshots must stay unchanged. The three complete encodings, after removing only observation, must hash identically. Raw off-runs are not duplicated; their equality is collector evidence. The middle encoding can be independently rehashed. An external profiler is refused.

## Independent exhaustive helper_lower

Runtime must be **CPython 3.12.10**, assertions enabled, `unicodedata.unidata_version == 15.0.0`. The helper calls builtin lower directly, never the table extractor or C# implementation. No native library is imported or loaded by the collector to obtain private Unicode functions.

There are **17 planes × four modes = 68 records**. Plane 0 visits 63488 scalars; every other plane visits 65536. All scalars 0 through 0x10FFFF are visited in ascending order except 0xD800–0xDFFF: 1112064 valid scalars, 4448256 input strings per pass. The modes construct exactly:

1. `chr(cp)`
2. `chr(cp) + 'Σ'`
3. `'A' + chr(cp) + 'Σ'`
4. `'AΣ' + chr(cp) + 'A'`

Each result is `(constructed_string).lower()`. Every plane/mode is repeated three times; output digest, input digest, counts and byte totals must match. Context modes probe Cased/Case_Ignorable effects around sigma; the multi-ignorable node cases supplement them. This is not an exhaustive proof over all Unicode strings and does not alone qualify every lower context.

The exact frame, independently for both input and result, is `uint32LE(cp) || uint32LE(UTF8_byte_length) || UTF8_strict_bytes`. Hash SHA256 from an empty state per plane/mode. No separators, BOM, terminator, normalization or JSON-number round trip occur. cp is included even when outputs coincide. Each record retains the plane, mode, scalar count, framed input/output byte totals, input hash and three result hashes. There is no million-line artifact. The scheme permits an independent C# implementation to regenerate the framing without calling Python.

The executable's basename/size/SHA is recorded and rechecked; no private path is written. That hash does not claim to attest all dynamically loaded Python libraries. Python version/Unicode version are actual runtime checks, not a package installation request.

## Admission and commands

Preflight only, before publication:

```powershell
python -I -S -B labs/text-comparison-source/reference.py --source <backend-git-repository>
```

After publication and explicit root admission:

```powershell
python -I -S -B labs/text-comparison-source/reference.py --source <backend-git-repository> --execute --output <new-absolute-external-directory>
```

All three local LF files must equal the raw blobs of one admitted collector HEAD before source execution. Output must not exist and must resolve outside both source and collector repositories. Helpers and frozen source are verified before calculation and again afterward, together with HEAD and laboratory files. Only after all cases, 68 digest plans, repeats and provenance checks succeed is reference.json written exclusively. Failure can leave an empty directory but no partial reference. No old lab, accepted fixture, source file or C# table is overwritten.

The future product may keep local resource/admission limits, but these do not justify an ASCII-only lower profile or a fallback to the host's Unicode tables. This laboratory creates source evidence only and reports no C# result or platform qualification.
