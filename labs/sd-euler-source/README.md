# Prospective Euler trajectory source laboratory

This isolated source laboratory defines eight short Euler trajectories before collecting any native result. It does not read C# implementation, accepted expected values or a checkpoint. It changes no existing generator, fixture, dependency lock or numerical profile. Its profile is a **new prospective scope**, `sd-euler-native210-cpu-f32-v1`, retaining the existing numerical bound without claiming that the new composition already meets it.

The committed `protocol.json` fixes all cases, input names/shapes/SHA256, literal Float32 sigma schedules, CFG options, prediction kinds, model configuration, parameter recipe, upstream file hashes, audited helper hashes, dependency locks and runtime. `reference.py` pins the protocol hash. Parameter names/shapes come from the actual source model's 686 named parameters, never a C# schema. The existing integer/power-of-two recipe assigns all parameters without RNG. The protocol's input hashes were calculated using Python stdlib only before any native collection.

## Exact source and explicit adapters

The backend is ComfyUI commit `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`. Every selected declaration executes as its original verified AST, including decorators:

- [`comfy/k_diffusion/sampling.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py#L190): `sample_euler` (canonical lines190–212), `to_d` (63–65).
- [`comfy/k_diffusion/utils.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/utils.py#L21): `append_dims` (21–29).
- `comfy/model_sampling.py`: `reshape_sigma`, EPS, V_PREDICTION, ModelSamplingDiscrete; `comfy/samplers.py`: `sampling_function`, `cfg_function`; `comfy/conds.py`: CONDRegular, CONDCrossAttn.
- The unchanged hash-pinned `labs/sd-source/unet.py` constructs the real reduced full-topology U-Net, exact timestep embedding and basic attention AST. Its no-init PyTorch operations and unsupported-extension sentinels retain their audited semantics.

`trange` is replaced by `range` only to remove progress output. The `sampler_calc_cond_batch_function` extension point supplies an explicit resident full-image condition adapter: exact EPS/V preprocessing, discrete timestep, actual U-Net `_forward`, exact denoised conversion, and source condition concatenation methods. This adapter performs Separate calls or ConcatenateCompatible calls as requested, with separate fallback. Missing negative conditioning yields a zero prediction, as in the audited whole-image CFG composition. The actual source `sampling_function` decides the near-one optimization and invokes the actual `cfg_function`; neither is rewritten. Regional conditioning, weighting masks, model patchers and extensions are not simulated.

The sampler receives an already initialized diffusion-space latent, explicit sigmas, and contexts with the correct batch. `s_churn=0` throughout. There is no seed, initial-noise generation, VAE scaling, checkpoint detection, scheduler-name resolution, image preprocessing or public tensor callback. Source Euler uses sigma expanded to batch, computes its original derivative and tensor `dt`, and performs the terminal arithmetic without an algebraic shortcut.

## Prospective corpus and contract

All cases use baseChannels32/context16/full four-level U-Net, CPU/F32 and one intra/inter-op thread. The corpus has one single-step trajectory, one three-step B2 trajectory, two three-step B2 cases with context lengths2/3 under the two policies, near-one CFG with negative absent, near-one with negative present, near-one with optimization disabled, and one three-step B2 velocity-prediction case. Near-one scale is `1.0000000005`; the other cases use3.5. These are primitive/composition cases, not model-family qualifications.

All inputs are borrowed and checked unchanged. Each case runs **off/on/off**, recording three final hashes and requiring bit equality. The middle repetition records the true source callback just before the Euler update: `index`, `x`, `denoised`, scalar `sigma` and `sigmaHat`. A callback copies records to host and can change allocations/timing; the two unobserved repetitions explicitly test its neutrality for this run. No equivalence to an unrelated process's allocation history is claimed. Actual model parameters are rehashed after each case.

`euler.json` contains profile/target/source AST evidence, common config/sourceConfig/parameters, and eight cases. Each case contains its options, `latent`, `sigmas`, `positive`, optional `negative` (JSON null if absent), final `output`, pre-step `steps`, repeatHashes and unchanged-input/parameter flags. Every tensor record has dtype, shape, little-endian F32 SHA256, values, strides and aligned64. Model call count and sigma batch shape are explicit. Cases are independent; neither CFG policy is used as the reference for the other.

The prospective finite-corpus comparison is `abs(actual-expected) <= 3e-5 + 3e-5*abs(expected)`, with exact discrete metadata and hashes. Unexpected nonfinite values or repeat changes fail collection. Collection itself does not compare a C# result or grant acceptance. Any later comparison must verify provenance/input/parameter identities and retain numerical failures without replacing the profile or the outputs to make tests pass.

## Isolation and collection

The workflow alone downloads the eight pinned source blobs. The script downloads nothing and imports no .NET project. Output must be an absent absolute directory outside the repository, source snapshot and Python installations; existing evidence cannot be overwritten. Sources, helpers, protocol/scripts and all three canonical LF locks are verified before and after calculation. Native imports occur only after input verification and the exact Python/package preconditions.

Three separate jobs use the existing hash-locked PyTorch2.10 CPU environments on Windows x64, Linux x64 and macOS arm64, with automatic CPU dispatch recorded explicitly. Forced ATEN_CPU_CAPABILITY is rejected. The manifest records source and AST hashes, protocol/scripts/lock hashes, Python/PyTorch commit, thread settings, native build/parallel configuration and hashes of wheel-package libraries. This last list is explicitly a package inventory, **not proof of actual loaded-library mappings**. Runtime changes or OS drift do not authorize a larger tolerance.

Before publication, only syntax and stdlib `--validate-only` may run locally. A validate-only run writes no outputs and imports no native dependencies. The first native collection follows review and publication. The workflow triggers only for this new laboratory or its own file, and uploads separate artifacts; it never copies data into accepted test fixtures.

Each job keeps one reduced U-Net resident. The fixed case set executes57 guided evaluations, representing93 U-Net calls including the separate CFG branches, across24 sampler invocations. Only19 pre-step callbacks are captured in the middle repetitions. This is a bounded source collection, not a stock-model run. A local checkout with CRLF lock files deliberately fails raw-byte preflight; the workflow checks out canonical LF bytes and never normalizes a lock during verification.

```sh
python -I -B labs/sd-euler-source/reference.py \
  --source-directory /absolute/frozen-source \
  --output /absolute/absent-euler-evidence --validate-only
```

No stock/pretrained qualification, image quality claim, RNG parity, GPU path, churn, inpainting, VAE or end-to-end text-to-image workflow is established by this laboratory.
