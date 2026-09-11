"""Classical image VAE evidence from verified frozen upstream declarations.

This module belongs only to the opt-in CPU reference laboratory. It never reads
product implementations, accepted fixtures, checkpoints, or network resources.
"""
import ast
from contextlib import contextmanager, nullcontext
import hashlib
import logging
import math
from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union

from common import COMMIT, fill_parameters, load_symbols, require, synthetic_tensor, tensor_record


MODEL = "comfy/ldm/modules/diffusionmodules/model.py"
MODEL_SHA = "a8e10d29e778007f5133bd870ed74fac478629727c08b81ec39390db3870d867"
AUTOENCODER = "comfy/ldm/models/autoencoder.py"
AUTOENCODER_SHA = "24b5a9126539733ef15f36e0d723abde8a8fb3319184083970b385ab918404f0"
DISTRIBUTIONS = "comfy/ldm/modules/distributions/distributions.py"
DISTRIBUTIONS_SHA = "3ae3040d1e43ecfff4ea9daa7807fa345e28cf93ae5e34d29436d49bb11df772"
SD = "comfy/sd.py"
SD_SHA = "ef373b88ced93955a7147dcd3323ecac9cd5316f0ae4623adfb9b6b128c37ba7"


def _raise(error):
    # The laboratory does not exercise an OOM retry or silently choose tiling.
    raise error


def _wrapper(source, evidence, namespace, model):
    """Execute the complete source VAE class, bypassing checkpoint/offload init.

    Only runtime/device infrastructure is supplied here. encode/decode/crop and
    the initial classical process_input/process_output lambdas remain exact AST.
    """
    import torch
    load_symbols(source, SD, SD_SHA, ["VAE"], namespace, evidence)
    wrapper = namespace["VAE"].__new__(namespace["VAE"])
    wrapper.first_stage_model = model
    wrapper.device = wrapper.output_device = torch.device("cpu")
    wrapper.vae_dtype = torch.float32
    wrapper.latent_dim = 2
    wrapper.downscale_ratio = wrapper.upscale_ratio = 8
    wrapper.latent_channels = 4
    wrapper.output_channels = 3
    wrapper.crop_input = True
    wrapper.pad_channel_value = None
    wrapper.disable_offload = True
    wrapper.not_video = False
    wrapper.extra_1d_channel = None
    wrapper.handles_tiling = False
    wrapper.format_encoded = None
    # A unit memory estimate and explicit large budget select one whole batch.
    # They affect orchestration only, never attention or tensor arithmetic.
    wrapper.memory_used_encode = wrapper.memory_used_decode = lambda shape, dtype: 1
    wrapper.patcher = SimpleNamespace(get_free_memory=lambda device: 2 ** 40)

    raw = (source / SD).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SD_SHA, "Frozen VAE wrapper changed during load.")
    cls = next(n for n in ast.parse(raw.decode("utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "VAE")
    initializer = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    # Only direct assignments in the initializer: family-specific branches are
    # excluded structurally, without rewriting either classical lambda body.
    assignments = [n for n in initializer.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
                   and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "self"
                   and n.targets[0].attr in ("process_input", "process_output")]
    require(len(assignments) == 2 and all(isinstance(n.value, ast.Lambda) for n in assignments),
            "Classical VAE normalization declarations differ.")
    selected = ast.Module(body=assignments, type_ignores=[])
    evidence.append({"source": SD, "url": f"https://github.com/comfy-org/ComfyUI/blob/{COMMIT}/{SD}",
                     "fileSha256": SD_SHA, "symbols": ["VAE.__init__.process_input", "VAE.__init__.process_output"],
                     "astSha256": hashlib.sha256(ast.dump(selected, include_attributes=False).encode()).hexdigest()})
    exec(compile(selected, SD, "exec"), {**namespace, "self": wrapper})
    return wrapper


def generate(source, evidence):
    import numpy as np
    import torch

    require(torch.__version__ in ("2.10.0", "2.10.0+cpu") and torch.version.cuda is None,
            "This evidence requires PyTorch 2.10.0 CPU.")
    require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
            "The source runner must configure one intra/inter-op thread.")
    require(not torch.is_grad_enabled(), "The source runner must establish no-grad.")
    management = SimpleNamespace(
        xformers_enabled_vae=lambda: False,
        pytorch_attention_enabled_vae=lambda: False,
        get_free_memory=lambda device: 2 ** 40,
        raise_non_oom=_raise,
        soft_empty_cache=lambda *args: None,
        cuda_device_context=lambda device: nullcontext(),
        load_models_gpu=lambda *args, **kwargs: None,
        intermediate_dtype=lambda: torch.float32,
    )
    comfy = SimpleNamespace(ops=SimpleNamespace(disable_weight_init=torch.nn),
                            model_management=management,
                            model_prefetch=SimpleNamespace(pause_malloc_graph=nullcontext))
    namespace = {"torch": torch, "nn": torch.nn, "ops": torch.nn, "np": np,
                 "math": math, "logging": logging, "comfy": comfy,
                 "model_management": management, "contextmanager": contextmanager,
                 "Any": Any, "Dict": Dict, "Tuple": Tuple, "Union": Union}
    load_symbols(source, MODEL, MODEL_SHA,
                 ["torch_cat_if_needed", "nonlinearity", "Normalize", "CarriedConv3d",
                  "conv_carry_causal_3d", "interpolate_up", "Upsample", "Downsample",
                  "ResnetBlock", "slice_attention", "normal_attention", "vae_attention",
                  "AttnBlock", "make_attn", "Encoder", "Decoder"], namespace, evidence)
    load_symbols(source, DISTRIBUTIONS, DISTRIBUTIONS_SHA,
                 ["DiagonalGaussianDistribution"], namespace, evidence)

    def instantiate(config):
        targets = {
            "comfy.ldm.modules.diffusionmodules.model.Encoder": namespace["Encoder"],
            "comfy.ldm.modules.diffusionmodules.model.Decoder": namespace["Decoder"],
            "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer": namespace["DiagonalGaussianRegularizer"],
        }
        require(config["target"] in targets, "Unapproved source constructor.")
        return targets[config["target"]](**config.get("params", {}))

    namespace["instantiate_from_config"] = instantiate
    load_symbols(source, AUTOENCODER, AUTOENCODER_SHA,
                 ["DiagonalGaussianRegularizer", "AbstractAutoencoder", "AutoencodingEngine",
                  "AutoencodingEngineLegacy", "AutoencoderKL"], namespace, evidence)
    ddconfig = {"double_z": True, "z_channels": 4, "resolution": 256, "in_channels": 3,
                "out_ch": 3, "ch": 32, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2,
                "attn_resolutions": [], "dropout": 0.0}
    model = namespace["AutoencoderKL"](embed_dim=4, ddconfig=ddconfig).cpu().float().eval()
    parameters = fill_parameters(model)
    require(len(parameters) == 248, "Unexpected classical VAE parameter schema.")
    require(not model.regularization.sample and model.max_batch_size is None and model.bn is None,
            "The classical mean/whole-batch route must remain explicit.")
    for module in model.modules():
        if isinstance(module, namespace["AttnBlock"]):
            require(module.optimized_attention is namespace["normal_attention"],
                    "The source must select exact normal_attention.")
    wrapper = _wrapper(source, evidence, namespace, model)
    cases = []

    def record_case(identifier, kind, shape, operation, capture_moments=False):
        value = synthetic_tensor(identifier + ".input", shape)
        if kind == "imageEncode":
            value = value * 2.0  # Explicit input values outside [0,1], saved below.
        original = value.clone()
        outputs = {}
        repeated = []
        for repeat in range(3):
            captured = {}
            handle = None
            if capture_moments:
                def capture(module, inputs, result):
                    captured["moments"] = result.detach().clone()
                handle = model.quant_conv.register_forward_hook(capture)
            try:
                result = operation(value)
            finally:
                if handle is not None:
                    handle.remove()
            captured["latent" if kind in ("encode", "imageEncode") else "image"] = result
            records = {name: tensor_record(tensor) for name, tensor in captured.items()}
            hashes = {name: tensor["sha256"] for name, tensor in records.items()}
            if repeat == 0:
                outputs = records
            else:
                require(hashes == repeated[0], "Source VAE forwards are not repeatable.")
            repeated.append(hashes)
        require(torch.equal(value, original), "Source VAE mutated a caller input.")
        cases.append({"id": identifier, "kind": kind, "input": tensor_record(value),
                      "outputs": outputs, "repeatHashes": repeated})

    record_case("vae-encode-odd-b1", "encode", (1, 3, 17, 25), model.encode, True)
    record_case("vae-encode-odd-b2", "encode", (2, 3, 9, 17), model.encode, True)
    record_case("vae-decode-b1", "decode", (1, 4, 2, 3), model.decode)
    record_case("vae-decode-b2", "decode", (2, 4, 1, 2), model.decode)
    record_case("vae-image-encode-rgba", "imageEncode", (1, 17, 25, 4), wrapper.encode)
    record_case("vae-image-encode-extra-channels", "imageEncode", (2, 19, 26, 5), wrapper.encode)
    record_case("vae-image-decode-b1", "imageDecode", (1, 4, 2, 3), wrapper.decode)
    record_case("vae-image-decode-b2", "imageDecode", (2, 4, 1, 2), wrapper.decode)
    return {"config": {"baseChannels": 32}, "sourceConfig": ddconfig, "parameters": parameters,
            "adapters": [
                "torch.nn operations replace disable_weight_init; every parameter is overwritten before execution",
                "instantiate_from_config accepts only the three exact source constructors listed in this module",
                "CPU, F32, normal_attention with a fixed 2**40-byte budget selecting one unsliced attention block",
                "VAE checkpoint/offload initialization is bypassed; exact source encode/decode/crop and initial image lambdas execute",
                "device/prefetch contexts and model residency are no-ops for an already resident CPU model",
                "unit memory estimates and large free-memory budget select whole-batch wrapper processing; all native exceptions propagate",
            ], "cases": cases}
