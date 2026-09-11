"""Reduced full-topology SD1/SD2 references from verified upstream ASTs only.

The .NET graph, weight schema and output tensors are never imported or consulted.
"""
from abc import abstractmethod
import functools
import logging
import math
from pathlib import Path
import types

from common import load_symbols, fill_parameters, synthetic_tensor, tensor_record, require


def source_model_type(source: Path, evidence: list):
    import torch
    from einops import rearrange, repeat

    class Linear(torch.nn.Linear):
        def reset_parameters(self):
            pass

    class Conv2d(torch.nn.Conv2d):
        def reset_parameters(self):
            pass

    class GroupNorm(torch.nn.GroupNorm):
        def reset_parameters(self):
            pass

    class LayerNorm(torch.nn.LayerNorm):
        def reset_parameters(self):
            pass

    def conv_nd(dimensions, *args, **kwargs):
        require(dimensions == 2, "The U-Net source profile only permits two-dimensional convolution.")
        return Conv2d(*args, **kwargs)

    class UnsupportedVideoResBlock(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise ValueError("Temporal residual blocks are outside this U-Net source profile.")

    class UnsupportedSpatialVideoTransformer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise ValueError("Temporal attention is outside this U-Net source profile.")

    class UnsupportedCheckpointFunction:
        @staticmethod
        def apply(*args, **kwargs):
            raise ValueError("Gradient checkpointing is outside this frozen inference profile.")

    operations = types.SimpleNamespace(Linear=Linear, Conv2d=Conv2d, GroupNorm=GroupNorm,
                                      LayerNorm=LayerNorm, conv_nd=conv_nd)
    namespace = {"torch": torch, "th": torch, "nn": torch.nn, "F": torch.nn.functional,
                 "einsum": torch.einsum, "math": math, "functools": functools,
                 "logging": logging, "abstractmethod": abstractmethod,
                 "rearrange": rearrange, "repeat": repeat, "ops": operations,
                 "comfy": types.SimpleNamespace(ops=operations),
                 "args": types.SimpleNamespace(dont_upcast_attention=False),
                 "FORCE_UPCAST_ATTENTION_DTYPE": None,
                 "VideoResBlock": UnsupportedVideoResBlock,
                 "SpatialVideoTransformer": UnsupportedSpatialVideoTransformer,
                 "CheckpointFunction": UnsupportedCheckpointFunction}
    load_symbols(source, "comfy/ldm/modules/diffusionmodules/util.py",
                 "fb58652a35521fc23bdcb75d91adace8e4cc79e2d5b13af1617a38d0c0f7142e",
                 ["checkpoint", "timestep_embedding", "avg_pool_nd"], namespace, evidence)
    load_symbols(source, "comfy/ldm/modules/attention.py",
                 "9cafaafaf93ff53e8cbefb5e4a204014019985df2da8c1996bf40f1235fc2960",
                 ["get_attn_precision", "exists", "default", "_heads_from_dim", "_reshape_qkv_to_heads",
                  "GEGLU", "FeedForward", "AttentionTensorContainer", "wrap_attn", "attention_basic",
                  "CrossAttention", "BasicTransformerBlock", "SpatialTransformer"], namespace, evidence)
    namespace["optimized_attention"] = namespace["attention_basic"]
    namespace["optimized_attention_masked"] = namespace["attention_basic"]
    load_symbols(source, "comfy/ldm/modules/diffusionmodules/openaimodel.py",
                 "9d27fb036cab8a262ef3d866a643f7fdc40994022616f1b8be14b7d919f57f96",
                 ["TimestepBlock", "forward_timestep_embed", "TimestepEmbedSequential", "Upsample",
                  "Downsample", "ResBlock", "apply_control", "UNetModel"], namespace, evidence)
    return namespace["UNetModel"], operations


def source_configuration(linear_projection):
    # These are the explicit plain SD15/SD21 constructor options, with only diagnostic
    # widths/context/head settings reduced. Keep mutable depth lists fresh per model.
    return {"image_size": 32, "in_channels": 4, "model_channels": 32, "out_channels": 4,
            "num_res_blocks": [2, 2, 2, 2], "dropout": 0.0, "channel_mult": [1, 2, 4, 4],
            "conv_resample": True, "dims": 2, "num_classes": None, "use_checkpoint": False,
            "num_heads": -1 if linear_projection else 4,
            "num_head_channels": 8 if linear_projection else -1,
            "use_scale_shift_norm": False, "resblock_updown": False,
            "use_spatial_transformer": True, "legacy": False, "context_dim": 16,
            "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
            "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            "transformer_depth_middle": 1, "use_linear_in_transformer": linear_projection,
            "use_temporal_resblock": False, "use_temporal_attention": False,
            "disable_middle_self_attn": False, "heatmap_head": False}


def generate(source: Path, evidence: list):
    import torch
    require(torch.__version__.split("+")[0] == "2.10.0" and torch.version.cuda is None,
            "U-Net source references require the declared PyTorch 2.10 CPU runtime.")
    require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
            "U-Net source references require one intra/inter-op thread.")
    model_type, operations = source_model_type(source, evidence)
    document = {"models": {}, "cases": [], "attention": "frozen attention_basic, explicitly selected",
                "adapter": {
                    "operations": "PyTorch Linear/Conv2d/GroupNorm/LayerNorm; reset_parameters disabled; dims=2 conv_nd only",
                    "initialization": "meta construction, CPU to_empty, all parameters assigned by the shared input recipe; no RNG",
                    "entryPoint": "unchanged UNetModel._forward; only external wrapper dispatch is bypassed",
                    "extensions": "no classes/control/patches/temporal/codebook/heatmap; unsupported sentinels reject activation",
                    "intermediates": "source module hooks on first repetition; no mathematical replacement",
                    "prediction": "raw four-channel output; no sigma conversion, CFG or latent scaling"}}
    shapes = [
        ("square", 1, 8, 8, 3, [0.125]),
        ("odd-rectangle", 1, 9, 11, 5, [17.25]),
        ("batch-distinct-time", 2, 8, 10, 7, [17.25, 731.5]),
        ("batch-shared-time-odd", 2, 7, 9, 4, [99.75]),
    ]
    # forward_timestep_embed iterates a block's children directly: the Sequential
    # container's forward hook is not called. Its final child gives that block's
    # unchanged output, including each upsample's next-skip target size.
    boundaries = {"time_embed": "timeEmbedding", "input_blocks.2.1": "down0",
                  "input_blocks.5.1": "down1", "input_blocks.8.1": "down2", "input_blocks.11.0": "down3",
                  "middle_block.2": "middle", "output_blocks.2.1": "up0",
                  "output_blocks.5.2": "up1", "output_blocks.8.2": "up2"}
    with torch.no_grad():
        for linear_projection in (False, True):
            identifier = "sd2-reduced" if linear_projection else "sd15-reduced"
            configuration = source_configuration(linear_projection)
            model = model_type(**configuration, dtype=torch.float32, device="meta", operations=operations)
            model.to_empty(device="cpu")
            model.requires_grad_(False)
            model.eval()
            parameters = fill_parameters(model)
            require(len(parameters) == 686, "Source-derived U-Net parameter count differs from the full plain topology.")
            document["models"][identifier] = {
                "config": {"baseChannels": 32, "contextSize": 16,
                           "headMode": "fixedSize" if linear_projection else "fixedCount",
                           "headParameter": 8 if linear_projection else 4, "useLinearProjection": linear_projection},
                "sourceConfig": configuration, "parameters": parameters}
            for label, batch, height, width, length, times in shapes:
                case_id = identifier + "/" + label
                latent = synthetic_tensor(case_id + "/latent", (batch, 4, height, width))
                context = synthetic_tensor(case_id + "/context", (batch, length, 16))
                timesteps = torch.tensor(times, dtype=torch.float32, device="cpu")
                captures = {}
                hooks = []
                for module_name, output_name in boundaries.items():
                    hooks.append(model.get_submodule(module_name).register_forward_hook(
                        lambda _module, _args, value, name=output_name: captures.__setitem__(name, tensor_record(value))))
                try:
                    output = model._forward(latent, timesteps=timesteps, context=context, y=None, control=None,
                                            transformer_options={})
                    expected = tensor_record(output)
                finally:
                    for hook in hooks:
                        hook.remove()
                hashes = [expected["sha256"]]
                for _ in range(2):
                    output = model._forward(latent, timesteps=timesteps, context=context, y=None, control=None,
                                            transformer_options={})
                    hashes.append(tensor_record(output)["sha256"])
                require(len(set(hashes)) == 1, "Repeated U-Net source forwards differ.")
                require(expected["shape"] == [batch, 4, height, width], "Source U-Net failed to preserve the latent shape.")
                require(set(captures) == set(boundaries.values()), "A required source boundary was not captured.")
                document["cases"].append({"id": case_id, "model": identifier,
                    "latent": tensor_record(latent), "timesteps": tensor_record(timesteps),
                    "context": tensor_record(context), "output": expected,
                    "intermediates": captures, "repeatHashes": hashes})
                del latent, context, timesteps, output
            del model
    return document
