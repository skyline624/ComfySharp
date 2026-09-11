# Separate SD source laboratory

This optional laboratory executes verified declarations from the frozen ComfyUI
backend to produce independent CPU reference outputs. It is separate from the
application, its build and its distributed .NET tests. It does not read model
checkpoints, download weights, contact a service or write acceptance fixtures.

The prospective comparison contract is
[`sd-components-native210-cpu-f32-v1`](../../docs/qualification/sd-components-native210-cpu-f32-v1.md).
The product continues to require no Python or Node runtime.

## Reproduce

Use a separate Python 3.12.10 environment with the existing exact hashed CPU
dependency closure from `../clip-source/requirements-<runtime>.txt`. Supported
runtime IDs are `win-x64`, `linux-x64` and `osx-arm64`. Run pip with
`--isolated --require-hashes --only-binary=:all:`. These laboratory wheels are not
product dependencies. No GPU wheel is accepted by the runner.

Prepare the source files listed in
[`sd-source-laboratory.yml`](../../.github/workflows/sd-source-laboratory.yml) from
the exact frozen commit. Use canonical Git blob bytes (for example the pinned raw
URLs), not a checkout that has converted line endings. Each file SHA-256 is
checked again before its selected source declarations execute. The generated
evidence records the selected declarations and their AST hashes.

Run the laboratory's interpreter, substituting absolute local directories:

```text
python -I -B labs/sd-source/reference.py --source-directory <snapshot> --output <new-output-directory> --components sampling unet vae guidance
```

The output must be outside both the source snapshot and the ComfySharp repository,
and absent or empty. Existing evidence is never overwritten. The `--components`
option can select a subset. The default selects only `sampling`.

## Evidence and limits

`manifest.json` records runtime, build, ISA, native library hashes, laboratory
script hashes, component document hashes and elapsed execution times. Component
JSON records exact inputs and source outputs. Nonfinite values use explicit
strings so the documents remain strict JSON.

Graph parameters are synthetic and use the public
`sha256-name-lcg-high16-power2-v1` recipe in `common.py`. The source model's own
parameter names and shapes define the schema; expected model outputs are never
computed by C#. All parameter and input scaling uses exactly representable binary
factors. There is no framework RNG dependency in the parameter recipe.

The three CPU CI targets produce independent evidence under the fixed profile.
Results must be reviewed and imported separately into the .NET fixtures; this
laboratory does not update them automatically or decide acceptance. Reduced graphs
and synthetic weights do not establish pretrained-model or GPU compatibility.
Public CPU CI does not replace Windows 11, Linux/NVIDIA or macOS/MPS hardware
qualification. The original CLIP profile and its recorded failures are unchanged.
