"""Independent whole-image CFG policy evidence using the actual frozen U-Net.

Compare separate calls with source-compatible context concatenation. This module
does not read .NET code/results and does not assume the two policies are numerically
interchangeable. The fixed profile bound is reported, never relaxed.
"""
import logging
import math

from common import fill_parameters, load_symbols, require, synthetic_tensor, tensor_record
from unet import source_configuration, source_model_type


def _comparison(first, second):
    left = first.detach().contiguous().reshape(-1).tolist()
    right = second.detach().contiguous().reshape(-1).tolist()
    require(len(left) == len(right), "Policy output sizes differ.")
    errors = []
    violations = []
    for index, (a, b) in enumerate(zip(left, right)):
        if not math.isfinite(a) or not math.isfinite(b):
            same = (math.isnan(a) and math.isnan(b)) or a == b
            error = 0.0 if same else math.inf
        else:
            error = abs(a - b)
            same = error <= 3e-5 + 3e-5 * abs(a)
        errors.append(error)
        if not same:
            violations.append({"index": index, "separate": a, "concatenateCompatible": b,
                               "absoluteError": error, "bound": 3e-5 + 3e-5 * abs(a)})
    # These finite synthetic cases do not need JSON-specific nonfinite encoding.
    require(all(math.isfinite(error) for error in errors), "Unexpected nonfinite CFG diagnostic.")
    return {"absoluteTolerance": 3e-5, "relativeTolerance": 3e-5,
            "elements": len(left), "maximumAbsoluteError": max(errors),
            "outsideBound": len(violations), "firstViolation": violations[0] if violations else None}


def generate(source, evidence):
    import torch
    import numpy as np

    require(torch.__version__.split("+")[0] == "2.10.0" and torch.version.cuda is None,
            "Guidance evidence requires PyTorch 2.10.0 CPU.")
    require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
            "Guidance evidence requires one intra/inter-op thread.")
    require(not torch.is_grad_enabled(), "The reference runner must establish no-grad.")
    namespace = {"torch": torch, "np": np, "math": math, "logging": logging}
    load_symbols(source, "comfy/ldm/modules/diffusionmodules/util.py",
                 "fb58652a35521fc23bdcb75d91adace8e4cc79e2d5b13af1617a38d0c0f7142e",
                 ["make_beta_schedule"], namespace, evidence)
    load_symbols(source, "comfy/model_sampling.py",
                 "173346f1975f6ddedee08505915ed97d152c3951747ebd213be0434f8d7011bd",
                 ["reshape_sigma", "EPS", "ModelSamplingDiscrete"], namespace, evidence)
    load_symbols(source, "comfy/samplers.py",
                 "f2c264ca9d394612f828e3ffe167c856a278a10e1711b269f0ba65dccb66393f",
                 ["cfg_function"], namespace, evidence)
    load_symbols(source, "comfy/conds.py",
                 "72058e9a22c972a9c875819e59d432d30d367fd2f7092ee6c6c45e5a60c959b0",
                 ["CONDRegular", "CONDCrossAttn"], namespace, evidence)
    model_type, operations = source_model_type(source, evidence)
    config = source_configuration(False)
    model = model_type(**config, dtype=torch.float32, device="meta", operations=operations)
    model.to_empty(device="cpu")
    model.requires_grad_(False)
    model.eval()
    parameters = fill_parameters(model)
    require(len(parameters) == 686, "Unexpected full plain U-Net parameter schema.")
    schedule = namespace["ModelSamplingDiscrete"]()
    predictor = namespace["EPS"]()
    predictor.sigma_data = 1.0
    regular = namespace["CONDRegular"]
    cross_attention = namespace["CONDCrossAttn"]
    cfg = namespace["cfg_function"]
    scale = 3.5

    def denoise(latent, sigma, context):
        scaled = predictor.calculate_input(sigma, latent)
        timestep = schedule.timestep(sigma).float().reshape(-1)
        prediction = model._forward(scaled, timesteps=timestep, context=context, y=None,
                                    control=None, transformer_options={})
        return predictor.calculate_denoised(sigma, prediction, latent)

    def separate(latent, sigma, positive, negative):
        conditional = denoise(latent, sigma, positive)
        unconditional = denoise(latent, sigma, negative)
        return cfg(None, conditional, unconditional, scale, latent, sigma)

    cases = []
    for positive_length, negative_length in ((2, 3), (3, 3), (2, 10)):
        latent = synthetic_tensor("guidance.latent", (2, 4, 4, 5))
        sigma = torch.tensor([0.25, 1.5], dtype=torch.float32, device="cpu")
        positive = synthetic_tensor("guidance.positive", (2, positive_length, 16))
        negative = synthetic_tensor("guidance.negative", (2, negative_length, 16))
        original = [value.clone() for value in (latent, sigma, positive, negative)]
        positive_cond, negative_cond = cross_attention(positive), cross_attention(negative)
        eligible = positive_cond.can_concat(negative_cond)
        context = positive_cond.concat([negative_cond]) if eligible else None
        repeated = []
        outputs = None
        comparison = None
        for iteration in range(3):
            first = separate(latent, sigma, positive, negative)
            if eligible:
                latents = regular(latent).concat([regular(latent)])
                sigmas = regular(sigma).concat([regular(sigma)])
                prediction = denoise(latents, sigmas, context)
                conditional, unconditional = prediction.chunk(2)
                second = cfg(None, conditional, unconditional, scale, latent, sigma)
            else:
                second = separate(latent, sigma, positive, negative)
            records = {"separate": tensor_record(first), "concatenateCompatible": tensor_record(second)}
            hashes = {name: record["sha256"] for name, record in records.items()}
            if iteration == 0:
                outputs = records
                comparison = _comparison(first, second)
            else:
                require(hashes == repeated[0], "Repeated source CFG policies differ.")
            repeated.append(hashes)
        require(all(torch.equal(before, after) for before, after in zip(original, (latent, sigma, positive, negative))),
                "Source CFG composition mutated a caller input.")
        cases.append({"id": f"guidance-{positive_length}-{negative_length}", "scale": scale,
                      "latent": tensor_record(latent), "sigma": tensor_record(sigma),
                      "positive": tensor_record(positive), "negative": tensor_record(negative),
                      "concatEligible": eligible, "commonLength": int(context.shape[1]) if eligible else 0,
                      "outputs": outputs, "repeatHashes": repeated, "policyComparison": comparison})
    return {"config": {"baseChannels": 32, "contextSize": 16, "headMode": "fixedCount",
                       "headParameter": 4, "useLinearProjection": False},
            "sourceConfig": config, "parameters": parameters, "predictionKind": "epsilon",
            "cases": cases,
            "composition": "exact EPS input/denoised + ModelSamplingDiscrete timestep + unchanged UNetModel._forward + cfg_function",
            "concatenation": "exact CONDCrossAttn.can_concat/concat and CONDRegular.concat; whole-sequence repetition; separate-call fallback",
            "adapters": "same resident CPU/F32, no-init operations and unsupported-extension sentinels as unet.source_model_type",
            "qualification": "policy-specific source outputs; inter-policy equivalence is measured, not assumed"}
