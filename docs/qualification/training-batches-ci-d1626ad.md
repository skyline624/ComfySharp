# Training batch noise discrepancies in CI

The [CI run for d1626ad](https://github.com/skyline624/ComfySharp/actions/runs/34709271629)
completed with Windows successful and Linux/macOS unsuccessful. The separate
dataset step passed six tests and failed its six source-reference recipes on each
of Linux x64 CPU and macOS ARM64 CPU. The four training-loop and two admission/
cancellation cases passed. No tests were skipped within that step.

Each reported source failure is an exact Float32 noise-array comparison. Examples
from the first failing array elements include:

| Platform / recipe | Expected | Observed |
|---|---|---|
| Linux / standard | -1.39380682 | -1.39380693 |
| Linux / buckets | -0.429795295 | -0.429795325 |
| macOS / standard | 0.0675485358 | 0.0675485283 |
| macOS / buckets | -0.429795295 | -0.429795325 |

These displayed log values are rounded and are not maximum-error statistics.
The initial RNG-state hash, dataset metadata and earlier assertions were reached
successfully, but failure on the first noise array prevents the later state and
batch checks from establishing complete sequence parity on these platforms.
The logs do not identify the native math implementation responsible for the
differences. A same-platform source/native noise comparison remains necessary.

No expected array, exact assertion, native dependency or numerical threshold was
changed to hide these failures. The subsequent sampler/denoising steps did not run
after this failure, so this CI is not new evidence about their previous failures.
The Windows-local batch qualification remains limited to its recorded runtime.
