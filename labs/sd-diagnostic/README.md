# Isolated SD runtime experiment

This opt-in diagnostic is separate from the source-reference producer and product.
Its own push-triggered workflow runs on one Ubuntu x64 host with the unchanged
hashed CPU laboratory dependencies and eleven frozen source blobs.

Three fresh source processes execute the same four components in the same order:
automatic ATen dispatch, `ATEN_CPU_CAPABILITY=avx2`, then `default`. The lowercase
override is set before starting each process. The six source scripts are unchanged
and must match the hashes recorded in the committed source manifest. Each mode's
four product reference suites then run in separate fresh processes, using their
committed expected outputs and the unchanged numerical profile.

An independent allocation control runs sixteen further fresh U-Net test processes
under automatic dispatch, with `COMFYSHARP_SD_LATENT_OFFSET=0` through `15`. Only
the diagnostic latent allocation changes; its values and strides are checked.
The baseline has this variable absent. Weights, context and timesteps are unchanged.
Each offset has its own trace/results directory inside the offset artifact.

Source and product artifacts are separated by requested mode. Control evidence
records CPU flags, the requested environment allowlist, native binary basenames
and hashes, process exit status, payload integrity and comparisons. Product CFG
traces are written before assertions, so failure does not discard their outputs.
The product's requested ATen override is not presented as an independently measured
dispatch capability. Native binary identity remains distinct from package versions.

`comparisons.json` reports source/source, product/source across all requested modes,
and offset-product/automatic-source measurements separately. Sampling and VAE
remain committed-reference control tests; quantitative product traces cover U-Net
and CFG. No comparison imports a fixture or grants qualification. A failed
committed test keeps its nonzero status even if a diagnostic source variant is
closer. Numerical thresholds, source scripts, lock files and accepted fixtures
are never changed by this runner.

The workflow uploads evidence even when collection records failing tests. It is
triggered only by pushes changing its own workflow or `labs/sd-diagnostic/**`.
Do not treat its output as a replacement reference without a separate reviewed
qualification decision.
