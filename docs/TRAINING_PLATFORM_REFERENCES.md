# Training references matched to the CPU platform

Training comparisons now select the independently collected source corpus for
Windows x64, Linux x64 or macOS ARM64, following the same platform boundary used
by the existing SD component references. This changes reference selection, not
the application math, source recipes, thresholds or asserted contracts.

The [source run 34711337834](https://github.com/skyline624/ComfySharp/actions/runs/34711337834)
succeeded on all three platforms at collector commit
`cd4d0348642984393aed211cc1f8d781eea7b5be`. The prospectively pinned protocol verifies
the source Git blobs, collector/helper bytes and unchanged hashed CPU environments.
Every artifact is identified by its manifest and raw-content SHA-256. Windows CI
reproduced the three original Windows files byte for byte; those files remain
unchanged and continue to be used on Windows.

The source audit found identical recipes, explicit inputs, target order, selections
and RNG-state hashes across platforms. ComfyUI itself produced platform-dependent
Float32 noise values and up-factor hashes. In the collected batches, the largest
noise/sigma difference from Windows was about `2.38e-7` on Linux and `1.23e-6` on
macOS. All 282 initial up-factor hashes differed from Windows on both platforms;
the first matches the hash previously observed by their .NET CI tests. These are
observations of the pinned native builds, not proof of a specific compiler/kernel
cause or a guarantee across all future CPU hardware.

`TrainingReferenceCorpus` verifies the immutable per-platform manifest and raw
artifact hashes before returning JSON. Linux/macOS data are stored as gzip
resources (under 1 MB total including all three manifests) to reduce disk use;
hashes are checked after decompression. The distributed tests use .NET only.
No runtime source generation, Python dependency or .NET-derived expected value
is introduced. Additional tests preserve recipe inputs and thresholds across the
three corpora.

Exact noise, selection, RNG-state and initializer checks remain exact. Denoising
and gradient comparisons retain `3e-5` absolute plus `3e-5` relative bounds.
Any disagreement against the corresponding platform's original source must still
fail and be investigated. The [qualification record](qualification/training-platform-references.json)
distinguishes successful source collection from actual .NET comparison results.
Complete training nodes, GPU source parity, real image datasets and model-family
qualification remain separate requirements.

The [.NET run 34712065610](https://github.com/skyline624/ComfySharp/actions/runs/34712065610)
at `8580892c8250639b8042c2062169dc725aad2622` completed with failures. macOS passed
all adapter, dataset and denoising steps (8, 12 and 96 tests), and its 860-test
inference suite; the separate stock CLIP L/G comparison still failed. Linux passed
exact initializer/RNG checks but failed two numerical output/gradient comparisons
in the adapter step; later dataset/denoising steps were not executed. Windows
passed 859/860 inference tests, with one final global tensor-count assertion
reporting 1 instead of 2. These failures remain open with unchanged assertions;
the qualification record preserves exact observations and avoids treating a
successful local rerun as their resolution.
