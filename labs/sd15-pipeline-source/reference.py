"""Prospective reduced pipeline source collector. No native imports during validation.

Draft execution is forbidden; publish and review the pinned laboratory first.
The product and its outputs are never imported or used as an oracle.
"""
import argparse
import ast
from contextlib import contextmanager, nullcontext
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import numbers
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import time
import types
from typing import Any, Dict, Tuple, Union

HERE = Path(__file__).resolve().parent
PROTOCOL_SHA256 = "f0e4f537414c6f8692c852832d040fe3fd5dc23fdc270bb351a16bbba6cd75e7"
PROFILE_SPEC = {"file": "docs/qualification/sd15-pipeline-native210-cpu-f32-v1.md",
                "gitCanonicalSha256": "d56988dd5dfeada0d5e0b7507c68cdc4d1dd53b659f0fcf45565ab9ecd727b32",
                "gitCanonicalBytes": 7128}
MAX_SOURCE_BYTES = 8 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def within(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Expected a relative input name.")
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "Input escapes its declared root.")
    return path


def bounded(path, maximum=MAX_SOURCE_BYTES):
    require(path.stat().st_size <= maximum, "Input exceeds its bound.")
    data = path.read_bytes()
    require(len(data) <= maximum, "Input grew beyond its bound.")
    return data


def selected_module(raw, names):
    nodes = []
    for node in ast.parse(raw.decode("utf-8")).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            nodes.append(node)
    require(len(nodes) == len(names), "Source AST selection is ambiguous or incomplete.")
    return ast.Module(body=nodes, type_ignores=[])


def ast_hash(module):
    return hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()


def canonical_admission(path, specification):
    raw = bounded(path)
    canonical = raw.replace(b"\r\n", b"\n")
    require(hashlib.sha256(canonical).hexdigest() == specification["gitCanonicalSha256"] and
            len(canonical) == specification["gitCanonicalBytes"], "Canonical helper/lock identity differs.")
    return {"rawSha256": hashlib.sha256(raw).hexdigest(), "rawBytes": len(raw),
            "canonicalSha256": specification["gitCanonicalSha256"], "canonicalBytes": len(canonical),
            "normalization": "literal CRLF-to-LF only", "crlfCount": raw.count(b"\r\n")}


def input_payloads(protocol):
    texts = {item["id"]: item for item in protocol["texts"]}
    require(len(texts) == 4, "Expected four unique public texts.")
    result = {}
    for entry in protocol["inputRecords"]:
        name = entry["id"]
        require(name not in result, "Duplicate input record.")
        if name.endswith("/ids") or name.endswith("/weights"):
            text_id, suffix = name.rsplit("/", 1)
            chunks = texts[text_id]["sourceChunks"]
            require(all(len(row) == 77 for row in chunks), "Noncanonical token chunk length.")
            triples = [triple for row in chunks for triple in row]
            if suffix == "ids":
                require(entry["dtype"] == "int64" and all(0 <= t[0] < 49408 for t in triples), "Invalid token IDs.")
                raw = b"".join(int(t[0]).to_bytes(8, "little", signed=True) for t in triples)
            else:
                require(entry["dtype"] == "float64", "Token weights must retain binary64 precision.")
                raw = b"".join(bytes.fromhex(t[1])[::-1] for t in triples)
                require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<d", raw)), "Invalid token weight.")
            require(entry["shape"] == [len(chunks), 77], "Token shape differs.")
        elif name.endswith("/noise"):
            require(entry["shape"] == [1, 4, 4, 5] and entry["dtype"] == "float32", "Noise shape/dtype differs.")
            seed = int.from_bytes(hashlib.sha256(entry["recipeName"].encode()).digest()[:4], "little")
            require(seed == entry["seed32"] and entry["parameter"] is False, "Noise recipe differs.")
            raw = b"".join(struct.pack("<f", ((((i * 1664525 + seed) & 0xffffffff) >> 16) - 32768) * 2.0 ** -15)
                           for i in range(80))
        elif name.endswith("/emptyLatent"):
            require(entry["shape"] == [1, 4, 4, 5] and entry["dtype"] == "float32", "Zero latent shape/dtype differs.")
            raw = bytes(320)
        elif name.endswith("/sigmas"):
            raw = b"".join(int(value, 16).to_bytes(4, "little") for value in entry["float32BitsHex"])
            require(entry["dtype"] == "float32" and entry["shape"] == [len(raw) // 4], "Sigma shape/dtype differs.")
            require(list(struct.unpack("<" + str(len(raw) // 4) + "f", raw)) == entry["values"], "Sigma bits/values differ.")
        else:
            raise ValueError("Unsupported input record.")
        require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"], "Input hash differs.")
        result[name] = raw
    require(len(result) == 20, "Expected twenty input records.")
    return result


def preflight(repo, source, output, workflow):
    repo, source = repo.resolve(strict=True), source.resolve(strict=True)
    require(repo.is_dir() and source.is_dir() and output.is_absolute(), "Explicit roots and absolute output required.")
    output = output.resolve()
    for protected in (repo, source, HERE, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output), "Output overlaps protected input.")
    require(not output.exists(), "Output must be absent; evidence is never overwritten.")
    protocol_path = HERE / "protocol.json"
    raw = bounded(protocol_path, 256 * 1024)
    require(hashlib.sha256(raw).hexdigest() == PROTOCOL_SHA256, "Prospective protocol hash differs.")
    protocol = json.loads(raw)
    require(protocol["parameterCount"] == 971 and len(protocol["cases"]) == 4, "Unapproved protocol size.")
    require(protocol["configs"]["clipProjectionPresent"] and not protocol["configs"]["projectionUsedForConditioning"], "Projection contract differs.")
    inputs = input_payloads(protocol)
    for case in protocol["cases"]:
        require(case["predictionKind"] == "epsilon" and case["policy"] == "separate" and
                case["negativeContextIsNull"] is False, "Unapproved pipeline policy.")
        values = next(r["values"] for r in protocol["inputRecords"] if r["id"] == case["inputs"]["sigmas"])
        require(len(values) >= 2 and values[-1] == 0 and all(math.isfinite(v) and v > 0 for v in values[:-1])
                and all(a >= b for a, b in zip(values, values[1:])), "Invalid sigma sequence.")
        maximum = protocol["sigmaMaximum"]["value"]
        require(case["maximumDenoise"] == (math.isclose(maximum, values[0], rel_tol=1e-5) or values[0] > maximum), "Maximum-denoise rule differs.")
        require(case["scaleFloat64BitsHex"] == struct.pack(">d", case["scale"]).hex(), "CFG scalar bits differ.")
    paths = {"laboratory/" + name: HERE / name for name in ("reference.py", "protocol.json", "README.md")}
    workflow = workflow.resolve(strict=True)
    require(workflow.is_file() and (workflow.is_relative_to(repo) or workflow.is_relative_to(HERE)), "Workflow must belong to the declared repository or draft.")
    paths["workflow"] = workflow
    source_bytes = {}
    for spec in protocol["sourceFiles"]:
        path = within(source, spec["file"])
        data = bounded(path)
        require(len(data) == spec["bytes"] and hashlib.sha256(data).hexdigest() == spec["sha256"], "Frozen source bytes differ.")
        require(ast_hash(selected_module(data, spec["symbols"])) == spec["astSha256"], "Frozen source AST differs.")
        paths["source/" + spec["file"]] = path
        source_bytes[spec["file"]] = data
    config = protocol["clipConfigSource"]
    config_path = within(source, config["file"])
    require(digest(config_path) == config["sha256"], "Source CLIP config differs.")
    actual_config = json.loads(bounded(config_path)); actual_config.update(config["overrides"])
    require(actual_config == protocol["configs"]["clip"], "Reduced source config differs.")
    paths["source/" + config["file"]] = config_path
    vae_class = next(n for n in ast.parse(source_bytes["comfy/sd.py"].decode()).body if isinstance(n, ast.ClassDef) and n.name == "VAE")
    vae_init = next(n for n in vae_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignments = [n for n in vae_init.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and
        isinstance(n.targets[0], ast.Attribute) and isinstance(n.targets[0].value, ast.Name) and
        n.targets[0].value.id == "self" and n.targets[0].attr in ("process_input", "process_output")]
    require(len(assignments) == 2 and ast_hash(ast.Module(body=assignments, type_ignores=[])) ==
            protocol["vaeNormalizationAst"]["astSha256"], "Classical VAE normalization AST differs.")
    attestations = {}
    for spec in protocol["helpers"] + protocol["locks"] + [PROFILE_SPEC]:
        path = within(repo, spec["file"])
        attestations[spec["file"]] = canonical_admission(path, spec)
        paths["repository/" + spec["file"]] = path
    donors = [protocol["tokenizerProvenance"]] + protocol["sigmaMaximum"]["donors"] + [
        protocol["sourceParameterSchemaDonors"][component] for component in ("unet", "vae")]
    for donor in donors:
        path = within(repo, donor["file"])
        require(path.stat().st_size == donor["bytes"] and digest(path) == donor["sha256"], "Existing source donor identity differs.")
        paths["donor/" + donor["file"]] = path
    # The only consumed donor values are tokenizer inputs and existing source schedule values.
    token_donor = json.loads(bounded(within(repo, protocol["tokenizerProvenance"]["file"]), 32 * 1024 * 1024))
    for entry in protocol["texts"]:
        matches = [c for c in token_donor["cases"] if c["id"] == entry["id"]]
        require(len(matches) == 1, "Tokenizer donor selection differs.")
        donor = matches[0]
        require(donor["text"] == entry["text"] and donor["chunks"] == entry["sourceChunks"] and donor["profile"] == "sd1-l", "Tokenizer source input differs.")
    for donor in protocol["sigmaMaximum"]["donors"]:
        table = json.loads(bounded(within(repo, donor["file"]), 32 * 1024 * 1024))["schedule"]["sigmas"]
        data = struct.pack("<1000f", *table["values"])
        require(hashlib.sha256(data).hexdigest() == donor["tableSha256"] == table["sha256"] and
                data[-4:].hex() == protocol["sigmaMaximum"]["littleEndianHex"], "Existing source sigma endpoint differs.")
    return repo, source, output, protocol, inputs, paths, attestations, source_bytes


def snapshot(paths):
    return {name: {"bytes": path.stat().st_size, "sha256": digest(path)} for name, path in sorted(paths.items())}


def verify_admitted_snapshot(protocol, before, attestations):
    require(before["laboratory/protocol.json"]["sha256"] == PROTOCOL_SHA256, "Protocol changed after admission.")
    for spec in protocol["sourceFiles"] + [protocol["clipConfigSource"]]:
        require(before["source/" + spec["file"]]["sha256"] == spec["sha256"], "Source changed after admission.")
    for name, spec in attestations.items():
        require(before["repository/" + name] == {"bytes": spec["rawBytes"], "sha256": spec["rawSha256"]},
                "Helper or lock changed after canonical admission.")
    donors = [protocol["tokenizerProvenance"]] + protocol["sigmaMaximum"]["donors"] + [
        protocol["sourceParameterSchemaDonors"][component] for component in ("unet", "vae")]
    for spec in donors:
        require(before["donor/" + spec["file"]] == {"bytes": spec["bytes"], "sha256": spec["sha256"]},
                "Donor changed after admission.")


def verify_execution_evidence(evidence, protocol, blobs):
    normalized = []
    for item in evidence:
        name = item.get("file", item.get("source"))
        spec = next(s for s in protocol["sourceFiles"] if s["file"] == name)
        names = item["symbols"]
        if names == protocol["vaeNormalizationAst"]["symbols"]:
            require(name == "comfy/sd.py", "Normalization source differs.")
            expected = protocol["vaeNormalizationAst"]["astSha256"]
        else:
            require(set(names).issubset(spec["symbols"]), "Helper executed unapproved source declarations.")
            expected = ast_hash(selected_module(blobs[name], names))
        require(item["fileSha256"] == spec["sha256"] and item["astSha256"] == expected,
                "Helper execution identity differs from the approved source AST.")
        normalized.append({"file": name, "fileSha256": spec["sha256"], "symbols": names, "astSha256": expected})
    require({item["file"] for item in normalized} == {s["file"] for s in protocol["sourceFiles"]},
            "Some approved source files were not loaded.")
    for spec in protocol["sourceFiles"]:
        loaded = {name for item in normalized if item["file"] == spec["file"] for name in item["symbols"]
                  if not name.startswith("VAE.__init__.")}
        require(loaded == set(spec["symbols"]), "Source declaration coverage differs from the prospective protocol.")
    return normalized


def import_helper(repo, relative, module_name):
    path = within(repo, relative)
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, "Helper cannot be imported.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_source(source_bytes, protocol, relative, names, namespace, evidence):
    spec = next(s for s in protocol["sourceFiles"] if s["file"] == relative)
    require(set(names).issubset(spec["symbols"]), "Unapproved source declaration requested.")
    module = selected_module(source_bytes[relative], names)
    evidence.append({"file": relative, "fileSha256": spec["sha256"], "symbols": names, "astSha256": ast_hash(module)})
    exec(compile(module, relative, "exec"), namespace)


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
    for name, value in {"num_layers": 2, "max_length": 77, "layer": "last", "layer_idx": None,
            "special_tokens": {"start": 49406, "end": 49407, "pad": 49407}, "operations": operations,
            "enable_attention_masks": False, "zero_out_masked": False, "layer_norm_hidden_state": True,
            "return_projected_pooled": False, "return_attention_masks": False, "execution_device": None}.items():
        setattr(clip_wrapper, name, value)
    clip_wrapper.eval()

    unet_helper = import_helper(repo, "labs/sd-source/unet.py", "pipeline_unet_helper")
    require(unet_helper.source_configuration(False) == protocol["configs"]["unet"], "Pinned U-Net helper config differs.")
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
    for component, model in models.items():
        schema = [{"name": name, "shape": list(p.shape)} for name, p in sorted(model.named_parameters())]
        expected = protocol["parameterSchemas"][component]
        require(len(schema) == expected["tensorCount"] and hashlib.sha256(compact(schema)).hexdigest() == expected["nameShapeSchemaSha256"], "Source parameter schema differs.")
        recipe_records = common.fill_parameters(model)
        actual = parameter_records(model)
        require([{k: row[k] for k in ("name", "shape", "sha256")} for row in actual] == recipe_records, "Native parameter storage differs from recipe.")
    return torch, np, common, models, clip_wrapper, vae_wrapper


def parameter_records(model):
    result = []
    for name, value in model.named_parameters():
        require(str(value.dtype) == "torch.float32" and value.device.type == "cpu" and not value.requires_grad and
                value.is_contiguous() and value.data_ptr() % 64 == 0, "Noncanonical source parameter storage.")
        result.append({"name": name, "shape": list(value.shape), "stride": list(value.stride()), "aligned64": True,
                       "sha256": hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()})
    return result


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def input_identity(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
            "device": value.device.type, "stride": list(value.stride()), "storageOffset": value.storage_offset(),
            "aligned64": value.data_ptr() % 64 == 0, "sha256": tensor_hash(value)}


def record(value, common):
    require(str(value.dtype) == "torch.float32" and value.device.type == "cpu" and
            not value.requires_grad and value.isfinite().all().item(), "Finite inference CPU/F32 capture required.")
    return {**common.tensor_record(value), "stride": list(value.stride()), "aligned64": value.data_ptr() % 64 == 0,
            "storageOffset": value.storage_offset()}


def verify_step_scalar(value, index, input_record, raw_sigmas):
    require(isinstance(index, int) and 0 <= index < input_record["shape"][0] - 1, "Euler index differs.")
    require(value.shape == () and str(value.dtype) == "torch.float32" and value.device.type == "cpu", "Euler scalar shape/dtype differs.")
    require(value.detach().numpy().tobytes() == raw_sigmas[index * 4:(index + 1) * 4], "Euler sigma/sigmaHat bits differ from the input schedule.")


def collect(repo, source, protocol, payloads, blobs):
    evidence = []
    torch, np, common, models, clip, vae = build_models(repo, source, protocol, blobs, evidence)
    namespace = {"torch": torch, "np": np, "math": math, "logging": logging}
    def load(relative, names, target=namespace):
        load_source(blobs, protocol, relative, names, target, evidence)
    load("comfy/ldm/modules/diffusionmodules/util.py", ["make_beta_schedule"])
    load("comfy/model_sampling.py", ["reshape_sigma", "EPS", "ModelSamplingDiscrete"])
    load("comfy/samplers.py", ["cfg_function", "sampling_function", "Sampler"])
    load("comfy/conds.py", ["CONDRegular", "CONDCrossAttn"])
    utility = {}; load("comfy/k_diffusion/utils.py", ["append_dims"], utility)
    namespace["utils"] = types.SimpleNamespace(append_dims=utility["append_dims"])
    namespace["trange"] = lambda count, disable=None: range(count)
    load("comfy/k_diffusion/sampling.py", ["to_d", "sample_euler"])
    load("comfy/latent_formats.py", ["LatentFormat", "SD15"])
    schedule = namespace["ModelSamplingDiscrete"]()
    require(tensor_hash(schedule.sigmas) == protocol["sigmaMaximum"]["donors"][0]["tableSha256"], "Actual source schedule differs from the pinned input donor.")
    predictor = namespace["EPS"](); predictor.sigma_data = 1.0
    latent_format = namespace["SD15"](scale_factor=protocol["pipeline"]["latentScale"])
    sampler_contract = namespace["Sampler"]()
    model_wrap = types.SimpleNamespace(inner_model=types.SimpleNamespace(model_sampling=schedule))
    entries = {r["id"]: r for r in protocol["inputRecords"]}
    text_cases = {t["id"]: t for t in protocol["texts"]}
    parameters = {name: parameter_records(model) for name, model in models.items()}
    require(sum(len(v) for v in parameters.values()) == 971, "Selected bank count differs.")
    cases = []
    projection_calls = [0]
    def projection_counter(module, args, output):
        projection_calls[0] += 1
    handle = models["clip"].text_projection.register_forward_hook(projection_counter)
    try:
        with torch.no_grad():
            for case in protocol["cases"]:
                native_inputs = {}
                for key in ("noise", "emptyLatent", "sigmas"):
                    id = case["inputs"][key]
                    native_inputs[key] = torch.frombuffer(bytearray(payloads[id]), dtype=torch.float32).clone().reshape(entries[id]["shape"])
                    require(tensor_hash(native_inputs[key]) == entries[id]["sha256"] and native_inputs[key].data_ptr() % 64 == 0, "Materialized input differs.")
                native_input_before = {key: input_identity(value) for key, value in native_inputs.items()}
                tokens = {}
                for polarity in ("positive", "negative"):
                    text = text_cases[case[polarity + "Text"]]
                    tokens[polarity] = [[(t[0], struct.unpack(">d", bytes.fromhex(t[1]))[0]) for t in row] for row in text["sourceChunks"]]
                token_snapshot = compact(tokens)
                captures = {}; steps = []; repeats = []; calls = []
                before_case = {name: parameter_records(model) for name, model in models.items()}
                require(before_case == parameters, "Parameters changed before a case.")
                projection_before = projection_calls[0]
                for iteration in range(3):
                    active = iteration == 1
                    def capture(name, value):
                        if active:
                            require(name not in captures, "Duplicate pipeline boundary.")
                            captures[name] = record(value, common)
                    positive, positive_pooled = clip.encode_token_weights(tokens["positive"])
                    negative, negative_pooled = clip.encode_token_weights(tokens["negative"])
                    require(list(positive.shape) == case["positiveContextShape"] and list(negative.shape) == case["negativeContextShape"], "CLIP context shape differs.")
                    capture("positive.hidden", positive); capture("positive.pooled", positive_pooled)
                    capture("negative.hidden", negative); capture("negative.pooled", negative_pooled)
                    sigmas = native_inputs["sigmas"]
                    require(sampler_contract.max_denoise(model_wrap, sigmas) == case["maximumDenoise"], "Actual max-denoise source decision differs.")
                    initial = predictor.noise_scaling(sigmas[0], native_inputs["noise"], native_inputs["emptyLatent"], case["maximumDenoise"])
                    capture("initialDiffusionLatent", initial)
                    call_count = [0]
                    def denoise(latent, sigma, context):
                        call_count[0] += 1
                        scaled = predictor.calculate_input(sigma, latent)
                        timestep = schedule.timestep(sigma).float().reshape(-1)
                        raw = models["unet"]._forward(scaled, timesteps=timestep, context=context, y=None, control=None, transformer_options={})
                        return predictor.calculate_denoised(sigma, raw, latent)
                    def calc(args):
                        cond, uncond = args["conds"]; latent, sigma = args["input"], args["sigma"]
                        return [denoise(latent, sigma, cond), torch.zeros_like(latent) if uncond is None else denoise(latent, sigma, uncond)]
                    options = {"disable_cfg1_optimization": False, "sampler_calc_cond_batch_function": calc}
                    def guided(latent, sigma):
                        require(list(sigma.shape) == [1], "Euler model sigma batch differs.")
                        return namespace["sampling_function"](None, latent, sigma, negative, positive, case["scale"], model_options=options)
                    def callback(step):
                        index = step["i"]
                        require(index == len(steps), "Euler callback index/order differs.")
                        sigma_entry = entries[case["inputs"]["sigmas"]]
                        for key in ("sigma", "sigma_hat"):
                            verify_step_scalar(step[key], index, sigma_entry, payloads[sigma_entry["id"]])
                        steps.append({"index": index, "x": record(step["x"], common), "denoised": record(step["denoised"], common),
                                      "sigma": record(step["sigma"], common), "sigmaHat": record(step["sigma_hat"], common)})
                    final = namespace["sample_euler"](guided, initial, sigmas, callback=callback if active else None, disable=True, s_churn=0.0)
                    final = predictor.inverse_noise_scaling(sigmas[-1], final)
                    capture("finalDiffusionLatent", final)
                    raw_latent = latent_format.process_out(final)
                    capture("rawVaeLatent", raw_latent)
                    image = vae.decode(raw_latent)
                    capture("image", image)
                    require(list(image.shape) == case["imageShape"] and image.isfinite().all().item(), "Image shape/finiteness differs.")
                    require(call_count[0] == case["unetCallsPerRun"], "U-Net call count differs.")
                    calls.append(call_count[0])
                    repeats.append({"finalDiffusionLatent": tensor_hash(final), "rawVaeLatent": tensor_hash(raw_latent), "image": tensor_hash(image)})
                    require({key: input_identity(value) for key, value in native_inputs.items()} == native_input_before,
                            "Source changed a borrowed native input's bits or layout.")
                    require(compact(tokens) == token_snapshot, "Source changed token inputs.")
                    require({name: parameter_records(model) for name, model in models.items()} == parameters, "Source changed parameter bits or layout.")
                    del positive, negative, positive_pooled, negative_pooled, initial, final, raw_latent, image
                require(repeats[0] == repeats[1] == repeats[2], "Off/on/off trajectory outputs differ.")
                require(len(captures) == 8 and len(steps) == case["steps"], "Pipeline captures incomplete.")
                require(projection_calls[0] - projection_before == 6, "Exact source auxiliary projection was not executed twice per repetition.")
                cases.append({"id": case["id"], "captures": captures, "steps": steps, "repeatHashes": repeats,
                    "unetCallsPerRun": calls, "clipEncodeCalls": 6, "vaeDecodeCalls": 3, "sourceAuxiliaryProjectionCalls": 6,
                    "inputsUnchanged": True, "parametersUnchanged": True, "parameterHashesBefore": before_case,
                    "nativeInputBefore": native_input_before,
                    "nativeInputAfter": {key: input_identity(value) for key, value in native_inputs.items()},
                    "parameterHashesAfter": {name: parameter_records(model) for name, model in models.items()}})
    finally:
        handle.remove()
    require(sum(len(c["steps"]) for c in cases) == 8 and sum(sum(c["unetCallsPerRun"]) for c in cases) == 45,
            "Aggregate source call/step count differs.")
    require(sum(len(c["captures"]) + len(c["steps"]) * 4 for c in cases) == 64, "Expected sixty-four capture records.")
    evidence = verify_execution_evidence(evidence, protocol, blobs)
    return torch, {"parameters": parameters, "inputRecords": protocol["inputRecords"], "texts": protocol["texts"],
                   "sourcesExecuted": evidence, "cases": cases, "sourceAuxiliaryProjectionCalls": projection_calls[0],
                   "observerSequence": ["off", "on", "off"], "syntheticWeights": True, "pretrainedWeightsUsed": False}


def loaded_libraries():
    # Native introspection is deliberately below preflight/validate-only, never imported by validation.
    paths = set()
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                paths.add(Path(fields[5]))
    elif sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel.GetCurrentProcess()
        enum = psapi.EnumProcessModules
        enum.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        enum.restype = wintypes.BOOL
        modules = (wintypes.HMODULE * 8192)(); needed = wintypes.DWORD()
        require(enum(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)) and needed.value <= ctypes.sizeof(modules), "Loaded module enumeration failed or overflowed.")
        get_name = psapi.GetModuleFileNameExW
        get_name.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        get_name.restype = wintypes.DWORD
        for i in range(needed.value // ctypes.sizeof(wintypes.HMODULE)):
            buffer = ctypes.create_unicode_buffer(32768)
            length = get_name(process, modules[i], buffer, len(buffer))
            require(0 < length < len(buffer), "Loaded module path could not be read.")
            paths.add(Path(buffer.value))
    elif sys.platform == "darwin":
        import ctypes
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        for index in range(dyld._dyld_image_count()):
            name = dyld._dyld_get_image_name(index)
            require(name is not None, "Loaded dyld image has no name.")
            paths.add(Path(os.fsdecode(name)))
    else:
        raise ValueError("Unsupported loaded-module platform.")
    selected = []
    for path in sorted(paths):
        name = path.name.lower()
        if not any(part in name for part in ("torch", "c10", "gomp", "iomp", "libomp", "mkl", "dnnl", "fbgemm")) and not name.startswith("_c."):
            continue
        require(path.is_file(), "A relevant mapped library cannot be hashed.")
        selected.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
    require(any("torch_cpu" in p["name"].lower() for p in selected) and any("c10" in p["name"].lower() for p in selected), "Actual CPU library mappings are incomplete.")
    require(len({p["name"] for p in selected}) == len(selected), "Duplicate mapped library basename is ambiguous.")
    return selected


def sanitize(value, protected):
    for path in protected:
        value = value.replace(str(path), "<local>").replace(str(path).replace("\\", "/"), "<local>")
    value = re.sub(r"(?:CXX_COMPILER|CUDA_TOOLKIT_ROOT_DIR)=[^,\n]*", "build_path=<redacted>", value)
    return value


def runtime_record(torch, protected):
    wheel = Path(torch.__file__).resolve().parent / "lib"
    inventory = [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(wheel.iterdir())
                 if p.is_file() and (p.suffix.lower() in (".dll", ".dylib", ".so") or ".so." in p.name)]
    return {"python": platform.python_version(), "torch": torch.__version__, "torchGit": torch.version.git_version,
        "numpy": importlib.metadata.version("numpy"), "einops": importlib.metadata.version("einops"),
        "threads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads(),
        "requestedCapability": os.environ.get("ATEN_CPU_CAPABILITY", "auto"), "actualTorchCapability": torch.backends.cpu.get_cpu_capability(),
        "processorIdentifier": platform.processor() or "unavailable", "machine": platform.machine(),
        "build": sanitize(torch.__config__.show(), protected), "parallelInfo": sanitize(torch.__config__.parallel_info(), protected),
        "actualLoadedLibraries": loaded_libraries(), "wheelPackageInventory": inventory,
        "libraryEvidence": "actual process mappings and package inventory are separate observations"}


def write_new(path, document):
    with path.open("xb") as stream:
        stream.write(json_bytes(document))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    repo, source, output, protocol, inputs, paths, attestations, blobs = preflight(
        args.repository_root, args.source_directory, args.output, args.workflow)
    before = snapshot(paths)
    verify_admitted_snapshot(protocol, before, attestations)
    if args.validate_only:
        require(snapshot(paths) == before, "Inputs changed during validation.")
        print(json.dumps({"status": "validated", "nativeImported": False, "inputRecords": len(inputs), "parameterCount": 971,
                          "protocolSha256": PROTOCOL_SHA256, "outputsCreated": False}))
        return 0
    require(HERE == repo / "labs/sd15-pipeline-source", "Drafts cannot execute; publish/review the source laboratory first.")
    require(sys.version_info[:3] == (3, 12, 10), "Pinned Python 3.12.10 required.")
    for package, expected in (("torch", "2.10.0"), ("numpy", "2.2.6"), ("einops", "0.8.1")):
        actual = importlib.metadata.version(package)
        require(actual == expected or package == "torch" and actual == expected + "+cpu", "Pinned numeric dependency differs.")
    require(not os.environ.get("ATEN_CPU_CAPABILITY"), "Forced CPU dispatch is outside this profile.")
    target = {("Windows", "AMD64"): "win-x64", ("Linux", "x86_64"): "linux-x64", ("Darwin", "arm64"): "osx-arm64"}.get((platform.system(), platform.machine()))
    require(target is not None, "Unsupported CPU platform.")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    require(re.fullmatch("[0-9a-f]{40}", commit), "Collector Git identity unavailable.")
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", "HEAD", "--", "labs", ".github/workflows"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "labs/sd15-pipeline-source/reference.py"], check=True, stdout=subprocess.DEVNULL)
    output.mkdir(parents=True)
    stage = "runtime-and-models"
    manifest_committed = False
    start = time.perf_counter()
    try:
        torch, document = collect(repo, source, protocol, inputs, blobs)
        require(torch.version.cuda is None and torch.get_num_threads() == torch.get_num_interop_threads() == 1, "Actual CPU runtime configuration differs.")
        stage = "integrity"
        after = snapshot(paths)
        require(after == before, "Protected raw input bytes changed during collection.")
        native = runtime_record(torch, (repo, source, output, HERE, Path.home(), Path(sys.prefix), Path(sys.base_prefix)))
        require(snapshot(paths) == before, "Protected inputs changed during runtime attestation.")
        stage = "publish"
        document = {"profile": protocol["profile"], "backendCommit": protocol["backendCommit"], "collectorCommit": commit,
                    "target": target, "protocolSha256": PROTOCOL_SHA256, "runtime": native,
                    "qualification": "none; prospective reduced source collection", **document}
        write_new(output / "pipeline.json", document)
        write_new(output / "prospective-protocol.json", protocol)
        manifest = {"status": "completed", "profile": protocol["profile"], "target": target, "collectorCommit": commit,
            "backendCommit": protocol["backendCommit"], "protocolSha256": PROTOCOL_SHA256, "qualification": "none",
            "inputIntegrityBefore": before, "inputIntegrityAfter": after, "canonicalAdmission": attestations,
            "component": {"file": "pipeline.json", "bytes": (output / "pipeline.json").stat().st_size, "sha256": digest(output / "pipeline.json")},
            "elapsedSeconds": time.perf_counter() - start, "cases": 4, "captureRecords": 64, "selectedParameters": 971,
            "comparison": protocol["comparison"], "runtime": native}
        require(snapshot(paths) == before, "Inputs changed before final manifest commit.")
        write_new(output / "manifest.pending.json", manifest)
        require(not (output / "manifest.json").exists(), "Final manifest already exists.")
        # link creates the final name atomically and fails if it already exists,
        # including on POSIX where rename would otherwise replace a racing file.
        os.link(output / "manifest.pending.json", output / "manifest.json")
        manifest_committed = True
        (output / "manifest.pending.json").unlink()
        print(json.dumps({"status": "completed", "cases": 4, "captureRecords": 64, "qualification": "none"}))
        return 0
    except BaseException as error:
        write_new(output / "incomplete.json", {"status": "incomplete", "stage": stage,
            "failureType": type(error).__name__, "qualification": "none", "finalManifestCommitted": manifest_committed})
        print(json.dumps({"status": "incomplete", "stage": stage, "failureType": type(error).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        # No traceback/private absolute path belongs in uploaded validation logs.
        print(json.dumps({"status": "preflight-failed", "failureType": type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
