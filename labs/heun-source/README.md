# Frozen Heun trajectory reference laboratory

This development-only collector executes `sample_heun`, `to_d` and
`append_dims` from ComfyUI commit
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. It extracts those declarations
from a separate Git checkout without importing the application or changing
the checkout. The application and .NET tests do not execute this laboratory.

Use the existing isolated `torch==2.10.0+cpu` environment, with one intra-op
and inter-op thread. No pretrained model or download is needed:

```text
python -I -B labs/heun-source/reference.py --source <frozen-git-checkout> --output <new-reference.json>
```

Output creation refuses to overwrite an existing file. Four explicit sigma
schedules use initial Float32 shape `[2,4,1,1]` and an analytical denoiser
`x * 0.25 + sigma * 0.125`, broadcast per batch. Every call's input values,
sigmas and the final output are recorded. The purpose is to verify the ODE
algorithm and evaluation order, not to substitute an analytical function
for neural-network qualification. No output from ComfySharp is fed to this
collector. New absolute and relative bounds `1e-6` were specified before
comparison; existing model tolerances and fixtures remain unchanged.

Provenance and k-diffusion's MIT notice remain in THIRD_PARTY_NOTICES.md and
docs/licenses/k-diffusion-MIT.txt alongside the project GPLv3 licence.
