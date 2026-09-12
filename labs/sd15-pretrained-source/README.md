# SD1.5 pretrained source comparison

This opt-in development laboratory executes frozen ComfyUI declarations at
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a` with the real SD1.5 EMA checkpoint.
It reuses the existing CPU Python/PyTorch 2.10 environment and verified source
snapshot. It never reads product activations during reference collection and
never downloads or copies model files. Python remains outside the application
and distributed .NET tests.

`protocol.json` fixes the checkpoint hash, source hashes, stock model dimensions,
tokenizer resources, prompt, seed, size, schedule and CPU thread count. It fixes
the diagnostic comparison criterion before collection: `3e-5 + 3e-5*abs(source)`.
Noise, sigmas and the initial latent require exact bytes. Existing accepted
profiles and fixtures are never replaced by this collector.

The source constructors use the same resident CPU infrastructure adapters as
the earlier laboratory. Actual CLIP, U-Net, VAE, EPS, CFG, Euler, Karras and latent
conversion bodies execute from hash-checked ComfyUI files. The model adapter
uses stock widths and loads every active parameter directly from the shared
safetensors file; no synthetic parameter filler is called. A projection matrix
absent from the checkpoint is identity only for the source's unused projected
pool calculation. SD1 conditioning uses the hidden output and unprojected pool.
The fixed ASCII sentence has an independent BPE input recipe using verified
vocabulary and merges. This does not qualify general tokenization or weighting.

Use the already provisioned environment from `labs/clip-source` and its platform
lock. Source snapshot paths are the same as the reduced pipeline laboratory.

```text
python -I -B labs/sd15-pretrained-source/reference.py --source <verified-snapshot> --checkpoint <shared-safetensors> --tokenizer <verified-tokenizer-directory> --output <new-directory>
```

The output contains eight little-endian Float32 captures and `result.json`.
Source bodies, loaded parameter identities, laboratory file hashes and runtime
versions are recorded. The model file is hashed before and after execution;
parameter storage and source files are checked unchanged. The output directory
must be absent. Failed runs may leave incomplete captures without a success
report; retain them separately and use a new directory after correcting a failure.

Run the .NET `sd15-generate` command with `--trace-dir <new-directory>`, saving
stdout as `result.json` beside that `traces` directory. Use the parameters in the
protocol and the same checkpoint. Then compare:

```text
python -I -B labs/sd15-pretrained-source/test_compare.py
python -I -B labs/sd15-pretrained-source/compare.py --source <reference-directory> --product <product-directory> --output <new-report.json>
```

The comparator checks settings, dimensions, dtypes, hashes, finiteness and all
eight unique captures. It retains every discrepancy and exits 1 on a numerical
failure. Comparator unit fixtures are tiny synthetic arrays; they are not model
qualification evidence. A passing real run covers only this resident CPU case,
not the full ComfyUI server, alternative workflows, GPU paths or model family.
