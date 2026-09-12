# Independent LoRA resume source collection

CI run 34725080075 passes the 14 resume tests on Windows but fails the four
all-parameter hash assertions on Linux and macOS. The distributed resume fixture
was collected on Windows. This laboratory investigates whether the frozen source
itself has platform-dependent initialization, without consulting C# outputs.

The collector, original source blobs, recipes and CPU dependency locks are hashed
in `protocol.json` before collection. `resume.py` executes the same frozen factory
and four cases; only its runtime check now admits the official macOS CPU version
string. Windows output still has SHA256
`47bb97f556dbfa0b3e6613e2e2be8613dcc643d3e9d906add15ee58e94268086`.
No comparison threshold, application code or expected fixture is changed by
collection. The separate GitHub workflow publishes JSON and a provenance manifest
for each OS. Python remains confined to this source laboratory.

```text
python -I -B labs/lora-resume-source/collect.py --source <frozen-source-checkout> --output <new-evidence-directory>
```

Matching source artifacts must be reviewed before use in .NET tests. A source
collection does not prove compatibility with real models, training workflows,
CUDA or MPS.
