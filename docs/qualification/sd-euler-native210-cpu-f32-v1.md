# Guided Euler trajectory comparison profile

This prospective profile covers eight short, no-churn Euler trajectories over an
already initialized diffusion latent. Its protocol was published at ComfySharp
commit `96fec72dd928276fc3f0ddc93a17406ea8249073` before collecting any numerical
output. The [protocol](../../labs/sd-euler-source/protocol.json) has SHA-256
`7868368a38fdf895fbe8698d4ab5bbbea1352a11df6dd7b3bd6c1e776c55cd05`.
It is a new composition profile; existing component and CLIP references remain
unchanged. A passing source collection alone does not qualify the C# sampler.

## Frozen scope

The source is ComfyUI `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`.
The separate [laboratory](../../labs/sd-euler-source/README.md) executes the actual
verified `sample_euler`, `to_d`, `append_dims`, `sampling_function` and CFG ASTs,
with the real reduced U-Net and exact EPS/V conversion. Its resident full-image
condition adapter uses the source extension point. It does not reproduce the
regional-conditioning engine, masks, model patches or extensions.

Runtime: CPU Float32, PyTorch/libtorch 2.10, one intra-op and one inter-op thread.
Source uses the unchanged locked CPU environment, Python 3.12.10, NumPy 2.2.6 and
einops 0.8.1. Distributed .NET tests consume JSON and native libraries only.
Each target has its own immutable source manifest and payload. This does not
promise identical outputs across operating systems, CPUs or native distributions.

The model has base channels 32, context width 16, four fixed heads, convolutional
spatial projections and 686 synthetic learned tensors. The protocol fixes all
input shapes and hashes before collection. Weight names and shapes come from
the source model; their deterministic recipe uses no RNG or pretrained weights.

| Case | Batch | Intervals | Prediction | Guidance |
|---|---:|---:|---|---|
| one-step | 1 | 1 | Epsilon | Separate, scale 3.5 |
| three-step-batch | 2 | 3 | Epsilon | Separate, scale 3.5 |
| unequal-separate | 2 | 3 | Epsilon | Context lengths 2/3, separate |
| unequal-concatenated | 2 | 3 | Epsilon | Context lengths 2/3, compatible concatenation |
| near-one-null | 1 | 2 | Epsilon | Scale 1.0000000005, negative absent |
| near-one-present | 1 | 2 | Epsilon | Same scale, negative present, optimization enabled |
| near-one-disabled | 1 | 2 | Epsilon | Same scale, optimization disabled |
| velocity-three-step | 2 | 3 | Velocity | Context lengths 2/3, separate |

Velocity here exercises prediction conversion on the explicit reduced graph;
it does not qualify SD2 or any other model family. Separate and concatenated
policies have independent expected results. Neither is substituted for the other.

## Comparisons and failure rules

All finite Float32 outputs and pre-update latent/denoised states must satisfy
`abs(actual - expected) <= 3e-5 + 3e-5 * abs(expected)` per element. Unexpected
nonfinite values fail. Discrete metadata, parameter/input identities, step
indices, sigma values, shapes and call counts are exact. Serialized Float32
payloads are rehashed before use; decimal JSON is interpreted as Float32.

Each case runs three complete trajectories with observer sequence off/on/off.
The central run captures every pre-update state. All three output hashes must
agree within each implementation, including when the observer is enabled.
Inputs and parameters must remain unchanged. These short checks do not prove
the absence of a persistent native memory leak.

No expected value is calculated by the C# implementation. A failed comparison
remains a failure; diagnostic traces are evidence for investigating it. Missing
fixtures, wrong manifests or unsupported runtimes must fail rather than skip.
Changing a tolerance or regenerating an accepted output requires a separate,
explicitly reviewed profile change, not an automatic response to a red test.

## Product boundary and remaining requirements

`SdEulerSampler` retains its denoiser for the complete operation and returns an
independently owned tensor. Its explicit schedule must be finite, nonincreasing,
positive before its final zero and contain at least two entries. Plateau sigmas
still evaluate the model. This is a restricted complete trajectory contract,
not a rule imposed on all ComfyUI samplers or partial schedules.

The component does not initialize noise, choose a scheduler, apply VAE scales,
encode text, load a checkpoint or decode an image. Noise/churn, stochastic
samplers, partial denoising, GPU execution, stock dimensions, pretrained weights,
full workflows and resource qualification remain separate migration work.
None of these synthetic trajectories closes a model-family or V1 release gate.
