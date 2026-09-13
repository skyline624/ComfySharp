# Windows CPU control laboratory

Runs the frozen training source collector and existing .NET tests in fresh
processes on one Windows x64 host. The source lab uses the pinned CPU requirements
in `labs/clip-source/requirements-win-x64.txt` and Python 3.12.10. It remains
separate from the application and distributed .NET tests.

First build `ComfySharp.Native` from `native/ComfySharp.NativeGenerator` against
the matching CPU SDK, then build `tests/ComfySharp.Inference.Tests` in Release
with `NativeRuntime=win-x64` and `NativeGeneratorBridge` pointing to that binary.
The workflow contains the complete pinned setup. A stale bridge lacking the
identity exports is an error, not inferred dispatch evidence.

```text
<lab-python> -I -B labs/windows-runtime/run.py --source <frozen-ComfyUI-checkout> --output <new-evidence-directory>
```

The destination must not exist. The script does not resume or overwrite an
earlier experiment. Each native process has a ten-minute timeout, and the
workflow retains artifacts on failure. Do not restart an experiment because a
tool observation timed out; inspect its running process first.

The profiles control ATen, then optionally MKL/oneDNN. These are requested
settings, with only the effective ATen capability read from libtorch. Complete
training traces preserve failed historical assertions; same-host comparisons
and the unobserved ordinary suite have separate verdicts. Any failed, missing
or unexecuted evidence prevents success. No pretrained weights, model downloads,
fixture updates or tolerance changes occur.

See `docs/qualification/windows-runtime-controls.md` for the motivating runner
comparison and the limitations of this experiment.
