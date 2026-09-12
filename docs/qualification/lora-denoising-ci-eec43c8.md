# Denoising training references: Linux/macOS failures

The [CI run for eec43c8](https://github.com/skyline624/ComfySharp/actions/runs/34706821339)
completed with Windows successful and Linux/macOS unsuccessful. The training step
on each failing platform passed 94 tests and failed two of the six new denoising
reference cases. Both failing cases use shared sigma 0.65, one epsilon and one
velocity prediction.

| Platform | Case | Comparison reached | Absolute error | Allowed at that value |
|---|---|---|---|---|
| Linux x64 CPU | SD1 epsilon | Input/sigma gradient comparison | 3.0566006898880005e-5 | 3.0460359659045936e-5 |
| Linux x64 CPU | SD2 velocity | Input/sigma gradient comparison | 4.2412430047988892e-5 | 3.0948559083044531e-5 |
| macOS ARM64 CPU | SD1 epsilon | Denoised prediction | 3.9502978324890137e-5 | 3.6273563057184221e-5 |
| macOS ARM64 CPU | SD2 velocity | Denoised prediction | 4.4807791709899902e-5 | 3.434649333357811e-5 |

These are the first reported failing elements, not maximum errors over the whole
case. The Linux assertion line contains both input and sigma checks; the log does
not distinguish which of the two failed. The logs establish a platform discrepancy,
not its root cause. Existing native-bundle investigations are relevant context,
but do not prove that this new discrepancy has the same cause.

No fixture, comparison tolerance or test is changed or skipped as a result.
Windows-local source comparisons and CUDA diagnostics remain their stated local
evidence. Neither these runs nor independent dataset/RNG work qualifies Linux,
macOS, a complete training node or a model family. A focused same-platform source
and native-operation comparison remains required before accepting these profiles.
