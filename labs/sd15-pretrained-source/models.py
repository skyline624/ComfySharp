"""Stock model initialization adapter; frozen source forward bodies are unchanged.
Derived from the existing reduced laboratory constructor, without synthetic filling.
CLIP has 12 layers, U-Net base320/context768/8heads, VAE base128.
All parameters are subsequently filled from the real checkpoint by reference.py.
"""
from contextlib import contextmanager, nullcontext
import hashlib, json, logging, math, numbers, os, types
from typing import Any, Dict, Tuple, Union

def bind(helper):
    global import_helper, selected_module, load_source, require, ast_hash
    import_helper=helper.import_helper
    selected_module=helper.selected_module
    load_source=helper.load_source
    require=helper.require
    ast_hash=helper.ast_hash

def build_models(repo, source, protocol, blobs, evidence):
    common = import_helper(repo, "labs/sd-source/common.py", "common")
    clip_helper = import_helper(repo, "labs/clip-source/reference.py", "pipeline_clip_helper")
    modules = {name: selected_module(blobs[name], spec["symbols"]) for name, spec in clip_helper.SOURCES.items() if "symbols" in spec}
    torch, np, operations, clip_type = clip_helper.initialize_runtime(modules)
    # initialize_runtime is the reviewed adapter only. Its unrelated stock fill/generator is never called.
    for name, module in modules.items():
        evidence.append({"file": name, "fileSha256": next(s["sha256"] for s in protocol["sourceFiles"] if s["file"] == name),
                         "symbols": clip_helper.SOURCES[name]["symbols"], "astSha256": ast_hash(module)})
    clip = clip_type(dict(protocol["configs"]["clip"]), torch.float32, "meta", operations)
    clip.to_empty(device="cpu"); clip.requires_grad_(False); clip.eval()
    management = types.SimpleNamespace(intermediate_device=lambda: torch.device("cpu"))
    clip_namespace = {"torch": torch, "numbers": numbers, "logging": logging, "os": os, "json": json,
        "comfy": types.SimpleNamespace(clip_model=types.SimpleNamespace(CLIPTextModel=clip_type), ops=operations),
        "model_management": management}
    load_source(blobs, protocol, "comfy/sd1_clip.py",
        ["gen_empty_tokens", "ClipTokenWeightEncoder", "SDClipModel", "SD1CheckpointClipModel"], clip_namespace, evidence)
    # Bypass file/model-management initialization only; actual encode/forward/weighting ASTs remain unchanged.
    clip_wrapper = clip_namespace["SD1CheckpointClipModel"].__new__(clip_namespace["SD1CheckpointClipModel"])
    torch.nn.Module.__init__(clip_wrapper)
    clip_wrapper.transformer = clip
    for name, value in {"num_layers": 12, "max_length": 77, "layer": "last", "layer_idx": None,
            "special_tokens": {"start": 49406, "end": 49407, "pad": 49407}, "operations": operations,
            "enable_attention_masks": False, "zero_out_masked": False, "layer_norm_hidden_state": True,
            "return_projected_pooled": False, "return_attention_masks": False, "execution_device": None}.items():
        setattr(clip_wrapper, name, value)
    clip_wrapper.eval()

    unet_helper = import_helper(repo, "labs/sd-source/unet.py", "pipeline_unet_helper")
    unet_type, unet_operations = unet_helper.source_model_type(source, evidence)
    unet = unet_type(**protocol["configs"]["unet"], dtype=torch.float32, device="meta", operations=unet_operations)
    unet.to_empty(device="cpu"); unet.requires_grad_(False); unet.eval()

    vae_helper = import_helper(repo, "labs/sd-source/vae.py", "pipeline_vae_helper")
    management = types.SimpleNamespace(xformers_enabled_vae=lambda: False, pytorch_attention_enabled_vae=lambda: False,
        get_free_memory=lambda device: 2 ** 40, raise_non_oom=vae_helper._raise, soft_empty_cache=lambda *args: None,
        cuda_device_context=lambda device: nullcontext(), load_models_gpu=lambda *args, **kwargs: None,
        intermediate_dtype=lambda: torch.float32)
    comfy = types.SimpleNamespace(ops=types.SimpleNamespace(disable_weight_init=torch.nn), model_management=management,
                                  model_prefetch=types.SimpleNamespace(pause_malloc_graph=nullcontext))
    namespace = {"torch": torch, "nn": torch.nn, "ops": torch.nn, "np": np, "math": math, "logging": logging,
        "comfy": comfy, "model_management": management, "contextmanager": contextmanager,
        "Any": Any, "Dict": Dict, "Tuple": Tuple, "Union": Union}
    model_symbols = next(s["symbols"] for s in protocol["sourceFiles"] if s["file"] == vae_helper.MODEL)
    load_source(blobs, protocol, vae_helper.MODEL, model_symbols, namespace, evidence)
    load_source(blobs, protocol, vae_helper.DISTRIBUTIONS, ["DiagonalGaussianDistribution"], namespace, evidence)
    def instantiate(config):
        allowed = {"comfy.ldm.modules.diffusionmodules.model.Encoder": namespace["Encoder"],
            "comfy.ldm.modules.diffusionmodules.model.Decoder": namespace["Decoder"],
            "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer": namespace["DiagonalGaussianRegularizer"]}
        require(config["target"] in allowed, "Unapproved VAE constructor.")
        return allowed[config["target"]](**config.get("params", {}))
    namespace["instantiate_from_config"] = instantiate
    auto_symbols = next(s["symbols"] for s in protocol["sourceFiles"] if s["file"] == vae_helper.AUTOENCODER)
    load_source(blobs, protocol, vae_helper.AUTOENCODER, auto_symbols, namespace, evidence)
    with torch.device("meta"):
        vae = namespace["AutoencoderKL"](embed_dim=4, ddconfig=dict(protocol["configs"]["vae"]))
    vae.to_empty(device="cpu"); vae.requires_grad_(False); vae.eval()
    require(not vae.regularization.sample and vae.max_batch_size is None and vae.bn is None, "Unexpected VAE route.")
    for module in vae.modules():
        if isinstance(module, namespace["AttnBlock"]):
            require(module.optimized_attention is namespace["normal_attention"], "VAE attention differs.")
    vae_wrapper = vae_helper._wrapper(source, evidence, namespace, vae)
    models = {"clip": clip, "unet": unet, "vae": vae}
    return torch, np, common, models, clip_wrapper, vae_wrapper
