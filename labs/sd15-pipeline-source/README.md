# Reduced SD1.5 pipeline source laboratory

Prospective CPU/F32 source collection for the four cases in `protocol.json`.
The published [comparison profile](../../docs/qualification/sd15-pipeline-native210-cpu-f32-v1.md)
is pinned to its canonical Git bytes from commit `3c00a4bf64555e49228ccd1274594144d820e542`.
This draft has not imported Torch or executed a model. Publication and independent
review must precede native execution. It has no checkpoint, stock, arbitrary
prompt, seed, size, batch, scheduler or guidance-policy option.

The collector executes frozen ComfyUI declarations at
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, using the pinned existing helpers and
Python 3.12.10 / PyTorch 2.10.0 CPU / NumPy 2.2.6 / einops 0.8.1 locks. Python
remains outside the product and .NET test runtime. No network or model files are
read by the collector; the workflow fetches only hash-pinned source blobs.

The 971 selected parameters are 37 CLIP, 686 U-Net and 248 classical VAE weights.
All use the same indexed integer input recipe and native aligned F32 destinations.
The CLIP stock collector's unrelated fill recipe is never used. Before filling,
each actual source model schema must match its prospectively pinned name/shape
digest. All parameter bits/layouts are checked again before cases and after runs.

The exact CLIPTextModel includes its projection; the original source executes
that auxiliary projection even though SD1 selects the unprojected pooled output.
The product conditionally computes it. A count-only source hook attests six
auxiliary projection calls per case. Neither pooled result conditions the U-Net.
This corpus cannot qualify absent projection. The CLIP wrapper's initialization
is replaced only by explicit resident CPU attributes and the actual transformer;
the original encode/forward/token-weight methods remain unchanged. No unused
logit_scale parameter is constructed by this initialization adapter.

The VAE uses exact classes and the existing `_wrapper` adapter with original
normalization lambdas, resident batch-one CPU orchestration and no OOM/tiling
fallback. Its model is initialized on meta, allocated once on CPU, then fully
filled; initialization produces no source weights used in inference. U-Net,
EPS, CFG, latent scaling and Euler source bodies remain unchanged. Separate CFG
uses a short resident-condition adapter, as in the existing Euler laboratory.

Inputs are the independently established source token chunks plus twelve small
noise/zero/sigma records. Text weights preserve their binary64 bits. The noise
recipe is a deterministic test input, not Gaussian or a seed-compatibility claim.
Smax is copied from the existing, rehashed three-OS source schedule, not computed
by C# or guessed from a decimal. Tokenization expected records are not pipeline
outputs; no earlier image, CLIP activation or U-Net prediction is an oracle here.

Each case runs off/on/off. Only the middle run captures the eight stage boundaries
and four records per Euler step, totaling 64 records across the four cases. The
final diffusion latent, raw VAE latent and image must repeat exactly. Before-update
sigma/sigmaHat have exact scalar shape/dtype and input-schedule bits at the exact
index; activation tolerance does not apply to them. There are 45 U-Net calls,
24 CLIP encodings and 12 VAE decodings overall. No extra VAE decode is invoked to
obtain a diagnostic raw output.

The protocol's `3e-5 + 3e-5*abs(reference)` comparator is prospective. This source
collector does not apply C# comparisons or declare qualification. Future
comparators must retain failures and compare the unclamped intermediate latents,
not only the final image. Capture hashing may materialize serialization copies;
those copies never feed model operations, and original strides/alignment are
recorded before serialization.

After publication under `labs/sd15-pipeline-source`, validate using the pinned
interpreter without site packages:

```text
python -I -S -B labs/sd15-pipeline-source/reference.py --repository-root <repository> --source-directory <snapshot> --workflow <workflow-file> --output <new-absolute-directory> --validate-only
```

Validation uses stdlib only and creates no output directory. It checks protocol,
all source/AST/config identities, normalization AST, helper/lock canonical
admission, source donor identities, token chunks, Smax and all twenty input hashes.
The same call without `--validate-only` requires the pinned installed CPU venv,
a supported target and a published/tracked collector at the expected lab path.
Draft locations are rejected before importing any numeric runtime.

Output must be new and separate from the repository, snapshot, lab and Python
runtimes. `pipeline.json` and its protocol copy precede a final manifest committed
atomically without replacement. Failure writes `incomplete.json` with stage/type
and the actual manifest-commit state, without a private traceback. No collection
failure before that final commit creates a success manifest. Raw bytes of scripts, protocol,
source blobs, donors, helpers and locks must remain identical before/after;
CRLF-to-LF admission of helpers/locks never excuses a mutation during execution.
The captured initial file snapshot must match the identities admitted by preflight.
All helper execution records are rechecked against the approved source declarations;
these records attest compiled declarations, not invocation of every method in a class.
Native input shapes, strides, offsets, alignment and hashes remain identical after
each repetition. Captured values never replace a live input to a model.

Actual loaded Torch/CPU/OpenMP-related library maps are enumerated and hashed by
platform, separately from wheel package inventory. Missing required CPU mappings
fail collection. Requested dispatch and observed Torch capability are distinct.
Only basenames/size/hash enter public evidence; local paths are removed from
runtime configuration strings. Same version does not imply same binary or CPU
math path. No force-dispatch override is permitted by this profile.

The sibling workflow draft is intended for
`.github/workflows/sd15-pipeline-source-laboratory.yml`. It runs three independent
OS jobs, restores only the existing hashed CPU locks in isolated venvs, validates
before collection, retains incomplete evidence and checks tracked inputs remain
unchanged. It performs no .NET build, product execution, fixture import or upload
of source/model packages.
