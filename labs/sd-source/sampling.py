"""Execute frozen schedule/prediction/CFG/latent-format declarations, without product code."""
import math
from common import load_symbols, tensor_record, synthetic_tensor


def generate(source, evidence):
    import torch
    import numpy as np
    namespace = {"torch": torch, "math": math, "np": np}
    load_symbols(source, "comfy/ldm/modules/diffusionmodules/util.py",
                 "fb58652a35521fc23bdcb75d91adace8e4cc79e2d5b13af1617a38d0c0f7142e",
                 ["make_beta_schedule"], namespace, evidence)
    load_symbols(source, "comfy/model_sampling.py",
                 "173346f1975f6ddedee08505915ed97d152c3951747ebd213be0434f8d7011bd",
                 ["reshape_sigma", "EPS", "V_PREDICTION", "ModelSamplingDiscrete"], namespace, evidence)
    load_symbols(source, "comfy/samplers.py",
                 "f2c264ca9d394612f828e3ffe167c856a278a10e1711b269f0ba65dccb66393f",
                 ["cfg_function"], namespace, evidence)
    load_symbols(source, "comfy/latent_formats.py",
                 "8b782699d1c58626fc3e7c8ac31608e1311ac7bf6c8d22e44e291391dc2734b4",
                 ["LatentFormat", "SD15", "SDXL"], namespace, evidence)
    schedule = namespace["ModelSamplingDiscrete"]()
    times = torch.tensor([-100, 0, 0.125, 1, 12.75, 127.5, 499.5, 998.9, 999, 2000], dtype=torch.float32)
    sigma_query = torch.tensor([0, 1e-8, 0.029167532, 0.1, 0.5, 1, 3.14159, 9.99, 14.614641, 20, 1e8], dtype=torch.float32)
    percent = [-1, 0, 0.000001, 0.1, 0.333333333, 0.5, 0.9, 0.999999, 1, 2]
    document = {"schedule": {
        "sigmas": tensor_record(schedule.sigmas), "logSigmas": tensor_record(schedule.log_sigmas),
        "timestepInputs": tensor_record(sigma_query), "timesteps": tensor_record(schedule.timestep(sigma_query)),
        "sigmaInputs": tensor_record(times), "interpolatedSigmas": tensor_record(schedule.sigma(times)),
        "percentInputs": percent, "percentSigmas": [schedule.percent_to_sigma(p) for p in percent]}, "cases": []}
    for mode in ("epsilon", "velocity"):
        predictor = namespace["EPS" if mode == "epsilon" else "V_PREDICTION"]()
        predictor.sigma_data = 1.0
        for sigma in (torch.tensor(0.0), torch.tensor([0.65]), torch.tensor([0.1, 14.0])):
            latent = synthetic_tensor("sampling/latent", (2, 4, 2, 3))
            prediction = synthetic_tensor("sampling/prediction", latent.shape)
            case = {"id": mode + "/" + str(list(sigma.shape)) + "/" + str(sigma.reshape(-1)[0].item()),
                    "predictionKind": mode, "sigma": tensor_record(sigma),
                    "latent": tensor_record(latent), "prediction": tensor_record(prediction),
                    "scaledInput": tensor_record(predictor.calculate_input(sigma, latent)),
                    "denoised": tensor_record(predictor.calculate_denoised(sigma, prediction, latent)),
                    "noiseScaled": tensor_record(predictor.noise_scaling(sigma, prediction, latent)),
                    "maximumNoiseScaled": tensor_record(predictor.noise_scaling(sigma, prediction, latent, True))}
            document["cases"].append(case)
    cond = synthetic_tensor("sampling/positive", (2, 4, 2, 3))
    uncond = synthetic_tensor("sampling/negative", cond.shape)
    document["guidance"] = {"conditional": tensor_record(cond), "unconditional": tensor_record(uncond), "cases": []}
    for scale in (-3.5, 0.0, 0.5, 1.0 - 5e-10, 1.0, 1.0 + 5e-10, 1.0 + 2e-9, 7.5):
        for disable in (False, True):
            omitted = math.isclose(scale, 1.0) and not disable
            negative = torch.zeros_like(uncond) if omitted else uncond
            result = namespace["cfg_function"](None, cond, negative, scale, None, None)
            document["guidance"]["cases"].append({"scale": scale, "disableOptimization": disable, "omitted": omitted,
                                                    "output": tensor_record(result)})
    raw_latent = synthetic_tensor("sampling/raw-latent", (1, 4, 2, 3))
    document["latentFormats"] = {"input": tensor_record(raw_latent), "cases": []}
    for kind in ("SD15", "SDXL"):
        format = namespace[kind]()
        diffusion = format.process_in(raw_latent)
        document["latentFormats"]["cases"].append({"kind": kind, "scale": format.scale_factor,
            "diffusion": tensor_record(diffusion), "roundTrip": tensor_record(format.process_out(diffusion))})
    return document
