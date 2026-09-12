# Frozen DPM++ 2M source laboratory

The collector extracts and executes only `sample_dpmpp_2m` from ComfyUI
commit `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. It reads the source with
`git show`, preserves its arithmetic and records the source hash. It does
not import the ComfyUI application, modify the checkout or read C# outputs.

Run in the existing isolated PyTorch `2.10.0+cpu` environment:

```text
python -I -B labs/dpmpp-2m-source/reference.py --source <upstream-checkout> --output <new-json>
```

Output creation refuses to overwrite an existing file. Six sigma sequences
use Float32 shape `[2,4,1,1]` and the analytical function
`x * 0.25 + sigma * 0.125`, broadcast per batch. All model-call inputs and
the final values are recorded. Nonfinite values are represented explicitly
as an invalid case, not nonstandard JSON NaN. The .NET test must diagnose
that case while preserving valid repeated-sigma cases.

The fixed new absolute/relative bounds are `1e-6`, chosen before comparison.
No existing model tolerance or historical fixture is changed. The collector
verifies ODE behavior; the independent pretrained native workflow establishes
execution only. No Python interpreter is part of the application or .NET tests.
GPLv3 and the retained k-diffusion MIT notice apply as documented in
THIRD_PARTY_NOTICES.md.
