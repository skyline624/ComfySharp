# Frozen SD scheduler laboratory

This development-only collector executes selected definitions from the frozen
ComfyUI commit `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a` through `git show`.
It does not modify the upstream checkout or import the ComfyUI application.
The model sampling table, nine scheduler functions and `KSampler.set_steps`
provide the results; no ComfySharp output is used as a reference.

The recorded environment is Python 3.12, PyTorch `2.10.0+cpu`, NumPy `2.2.6`
and SciPy `1.18.0`, one intra-op and inter-op thread. In this campaign SciPy
was loaded from an existing compatible Python 3.12 environment via an explicit
optional path, after the isolated environment's NumPy/PyTorch had been loaded.
No package was installed, copied or modified. For an environment already
containing those dependencies, omit `--scipy-path`:

```text
python -I -B labs/sd-schedulers-source/reference.py --source <upstream-checkout> --output <new-json>
```

The script records dependency versions and exact upstream file hashes and
refuses to overwrite the output. It collects 71 cases, including variable
lengths, partial denoise, beta duplicate removal and DDIM requests above the
table length. Nonfinite source cases are recorded explicitly, without JSON
NaN values. The .NET tests diagnose those cases instead of accepting a
fabricated finite schedule. New absolute/relative bounds `1e-6` were fixed
before comparison; no historical fixture or model tolerance was relaxed.

This is a scheduler reference, not a model-family or GPU numerical acceptance.
Neither Python nor SciPy is part of the application or distributed .NET tests.
GPLv3 and applicable k-diffusion notices are retained in THIRD_PARTY_NOTICES.md.
