"""Isolated, opt-in source laboratory. Never imported by product or .NET tests.

Executes verified frozen upstream ASTs with synthetic parameters. Outputs are
observational traces, never acceptance fixtures. No network or model-file access.
"""
import argparse
import ast
import functools
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import types

COMMIT = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a"
INPUT_SHA = "edc3470a883c96f79e75d4c222b2ba8089ba4d4f78de0d5b941dc03175135f6c"
SOURCES = {
    "comfy/ldm/modules/attention.py": {
        "sha256": "9cafaafaf93ff53e8cbefb5e4a204014019985df2da8c1996bf40f1235fc2960",
        "symbols": ["get_attn_precision", "exists", "_heads_from_dim", "_reshape_qkv_to_heads",
                    "AttentionTensorContainer", "wrap_attn", "attention_basic"],
        "astSha256": "59f3a86ee4966e65e25db886e97c707ad72ff53af6919f99bd300cac96aa604f",
    },
    "comfy/clip_model.py": {
        "sha256": "de877f05881345974bcda584696537a5a425215043f88ce9b97864cecee8343d",
        "symbols": ["CLIPAttention", "ACTIVATIONS", "CLIPMLP", "CLIPLayer", "CLIPEncoder",
                    "CLIPEmbeddings", "CLIPTextModel_", "CLIPTextModel"],
        "astSha256": "5cdeb4ba168704f3868e01d6811fcaae796a81665a5c3faf3da550dd60599a03",
    },
    "comfy/sd1_clip_config.json": {
        "sha256": "c4dabd85f1abdd818c7a9f5dee1f6cf2b9cebc5ed3237ab2f2c4b00bdb02ce6c",
    },
    "comfy/clip_config_bigg.json": {
        "sha256": "c5092eb3e08a7bf9ac1163578629c03ecc6b4fc38c6c1af4f88892d8391fb7d5",
    },
}
CONFIG_FILES = {"l": "comfy/sd1_clip_config.json", "g": "comfy/clip_config_bigg.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_bytes(path, maximum):
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "Input exceeds the bounded laboratory read size.")
    return raw


def overlaps(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


def validate_paths(source, inputs, output):
    require(output.is_absolute(), "Output must be an explicit absolute directory.")
    source, inputs, output = source.resolve(strict=True), inputs.resolve(strict=True), output.resolve()
    repository = Path(__file__).resolve().parents[2]
    require(source.is_dir() and inputs.is_file(), "Sources must be a directory and inputs a file.")
    for protected in (repository, source, inputs.parent):
        require(not overlaps(output, protected), "Output must be separate from the repository, source snapshots and inputs.")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Output must be absent or an empty directory; existing evidence is never overwritten.")
    return source, inputs, output


def load_verified_inputs(source, inputs):
    require(sys.version_info[:2] == (3, 12), "Python 3.12 is required for the pinned AST serialization.")
    raw = checked_bytes(inputs, 32 * 1024 * 1024)
    require(hashlib.sha256(raw).hexdigest() == INPUT_SHA, "Stock input document SHA-256 mismatch.")
    document = json.loads(raw)
    require(document["backendCommit"] == COMMIT, "Stock source revision mismatch.")
    # Only copy inputs. Expected output arrays are never decoded, compared, or used.
    cases = {case["id"].removeprefix("stock/"): {
        key: case[key] for key in ("config", "tokens", "options")
    } for case in document["cases"]}
    del document, raw
    require(set(cases) == {"l", "g"}, "Expected exactly the two pinned stock inputs.")
    modules, evidence, blobs = {}, [], {}
    for relative, specification in SOURCES.items():
        path = (source / relative).resolve(strict=True)
        require(path.is_relative_to(source), "Source snapshot resolves outside its directory.")
        raw = checked_bytes(path, 8 * 1024 * 1024)
        require(hashlib.sha256(raw).hexdigest() == specification["sha256"], "Source snapshot SHA-256 mismatch: " + relative)
        blobs[relative] = raw
        entry = {"url": f"https://github.com/comfy-org/ComfyUI/blob/{COMMIT}/{relative}",
                 "fileSha256": specification["sha256"]}
        if "symbols" in specification:
            selected = []
            for node in ast.parse(raw.decode("utf-8")).body:
                name = getattr(node, "name", None)
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                if name in specification["symbols"]:
                    selected.append(node)
            require(len(selected) == len(specification["symbols"]), "AST symbol selection mismatch.")
            module = ast.Module(body=selected, type_ignores=[])
            digest = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
            require(digest == specification["astSha256"], "Selected AST SHA-256 mismatch: " + relative)
            modules[relative] = module
            entry.update(symbols=specification["symbols"], astSha256=digest)
        evidence.append(entry)
    for variant, case in cases.items():
        config = json.loads(blobs[CONFIG_FILES[variant]])
        require(config == case["config"], "Source config differs from stock input config.")
        case["config"] = config
        require(case["options"] == {"num_tokens": [5, 3], "intermediate_output": -2,
                                    "final_layer_norm_intermediate": False}, "Stock options mismatch.")
        require(len(case["tokens"]) == 2 and all(len(row) == 77 for row in case["tokens"]), "Stock token shape mismatch.")
    return cases, modules, evidence


def initialize_runtime(modules):
    # Check distribution metadata before importing/loading the native runtime.
    require(importlib.metadata.version("torch").split("+")[0] == "2.10.0", "This laboratory requires PyTorch 2.10.0; no alternate version is permitted.")
    import numpy as np
    import torch
    from einops import rearrange, repeat

    require(torch.__version__.split("+")[0] == "2.10.0" and torch.version.cuda is None,
            "PyTorch 2.10.0 CPU build is required.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    class Linear(torch.nn.Linear):
        def reset_parameters(self):
            pass

    class LayerNorm(torch.nn.LayerNorm):
        def reset_parameters(self):
            pass

    class Embedding(torch.nn.Embedding):
        def reset_parameters(self):
            pass

        def forward(self, input, out_dtype=None):
            return torch.nn.functional.embedding(input, self.weight.to(dtype=out_dtype or self.weight.dtype),
                                                  self.padding_idx, self.max_norm, self.norm_type,
                                                  self.scale_grad_by_freq, self.sparse)

    operations = types.SimpleNamespace(Linear=Linear, LayerNorm=LayerNorm, Embedding=Embedding,
                                      cast_to=lambda tensor, dtype=None, device=None: tensor.to(dtype=dtype, device=device))
    namespace = dict(torch=torch, einsum=torch.einsum, rearrange=rearrange, repeat=repeat,
                     functools=functools, args=types.SimpleNamespace(dont_upcast_attention=False),
                     FORCE_UPCAST_ATTENTION_DTYPE=None, comfy=types.SimpleNamespace(ops=operations))
    attention = "comfy/ldm/modules/attention.py"
    exec(compile(modules[attention], attention, "exec"), namespace)
    namespace["optimized_attention_for_device"] = lambda *args, **kwargs: namespace["attention_basic"]
    clip = "comfy/clip_model.py"
    exec(compile(modules[clip], clip, "exec"), namespace)
    return torch, np, operations, namespace["CLIPTextModel"]


def fill(model, torch, np):
    for name, parameter in list(model.named_parameters()):
        digest = hashlib.sha256(name.encode("ascii")).digest()
        seed = int.from_bytes(digest[:4], "little") % 65521
        stride = 1 + int.from_bytes(digest[4:8], "little") % 251
        shape = tuple(parameter.shape)
        count = math.prod(shape)
        values = np.empty(count, dtype=np.float32)
        # Chunked integer intermediates limit the peak during the large embedding bank.
        for start in range(0, count, 1024 * 1024):
            end = min(start + 1024 * 1024, count)
            chunk = ((np.arange(start, end, dtype=np.int64) * stride + seed) % 257 - 128).astype(np.float32)
            chunk /= np.float32(4096 if len(shape) > 1 else 8192)
            if name.endswith(".weight") and "layer_norm" in name:
                chunk += np.float32(1)
            values[start:end] = chunk
        owner, attribute = name.rsplit(".", 1)
        model.get_submodule(owner)._parameters[attribute] = torch.nn.Parameter(
            torch.from_numpy(values.reshape(shape)), requires_grad=False)
    return model.eval()


def native_evidence(torch):
    def safe(operation):
        try:
            return operation()
        except (AttributeError, RuntimeError):
            return "unavailable"

    def sanitized(text):
        # Config output can contain build-machine paths; preserve configuration but
        # omit compiler path settings and any actual current home/prefix/cwd paths.
        text = re.sub(r"(?:CXX_COMPILER|CUDA_TOOLKIT_ROOT_DIR)=[^,\n]*", "build_path=<redacted>", text)
        for path in (str(Path.home()), str(Path.cwd()), sys.prefix):
            if len(path) > 3:
                text = text.replace(path, "<local-path>").replace(path.replace("\\", "/"), "<local-path>")
        return text

    package = Path(torch.__file__).resolve().parent
    binaries = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and (path.name.endswith((".dll", ".so", ".dylib", ".pyd")) or ".so." in path.name):
            resolved = path.resolve()
            require(resolved.is_relative_to(package), "Native package binary resolves outside the package.")
            binaries.append({"file": path.relative_to(package).as_posix(), "sha256": digest_file(path), "bytes": path.stat().st_size})
    require(bool(binaries), "No native package binaries found for provenance.")
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": importlib.metadata.version("numpy"), "einops": importlib.metadata.version("einops"),
        "operatingSystem": platform.system().lower(), "processArchitecture": platform.machine(),
        "torchGitVersion": torch.version.git_version,
        "torchConfig": sanitized(torch.__config__.show()),
        "parallelConfig": sanitized(torch.__config__.parallel_info()),
        "cpuCapability": torch.backends.cpu.get_cpu_capability(),
        "mklAvailable": torch.backends.mkl.is_available(), "mkldnnAvailable": torch.backends.mkldnn.is_available(),
        "mkldnnEnabled": torch.backends.mkldnn.enabled, "mkldnnDeterministic": torch.backends.mkldnn.deterministic,
        "float32MatmulPrecision": torch.get_float32_matmul_precision(),
        "globalFp32Precision": safe(lambda: torch.backends.fp32_precision),
        "mkldnnMatmulFp32Precision": safe(lambda: torch.backends.mkldnn.matmul.fp32_precision),
        "cudaMatmulFp32Precision": safe(lambda: torch.backends.cuda.matmul.fp32_precision),
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
        "gradEnabled": torch.is_grad_enabled(), "threads": torch.get_num_threads(),
        "interopThreads": torch.get_num_interop_threads(), "defaultDtype": str(torch.get_default_dtype()),
        "nativeBinaries": binaries,
    }


def capture_profile(variant, case, output, torch, np, model_type, operations, sources, native):
    started = time.perf_counter()
    model = fill(model_type(case["config"], torch.float32, "meta", operations), torch, np)
    directory = output / variant
    directory.mkdir()
    manifest = {"schemaVersion": 1, "backendCommit": COMMIT, "profile": variant,
                "synthetic": True, "modelWeightsUsed": False, "observationalOnly": True,
                "diagnosticOnly": True, "acceptanceChanged": False, "inputCorpusSha256": INPUT_SHA,
                "config": case["config"], "configSha256": SOURCES[CONFIG_FILES[variant]]["sha256"],
                "tokens": case["tokens"], "originalOptions": case["options"],
                "options": dict(case["options"], intermediate_output="all"),
                "native": native, "sources": sources,
                "parameterRecipe": "SHA256 ASCII name; LE u32 seed%65521, stride=1+LE u32%251; ((index*stride+seed)%257-128)/(4096 matrix else 8192); layer_norm.weight += 1; F32.",
                "parameters": {}, "parameterShapes": {}, "tensors": {}}

    def array_and_payload(tensor):
        require(tensor.device.type == "cpu" and tensor.dtype == torch.float32, "Trace tensor is not CPU/F32.")
        array = tensor.detach().contiguous().numpy().astype("<f4", copy=False)
        return array, memoryview(array).cast("B")

    def tensor_hash(tensor):
        array, payload = array_and_payload(tensor)
        return hashlib.sha256(payload).hexdigest()

    def save(name, tensor):
        array, payload = array_and_payload(tensor)
        with (directory / (name + ".f32")).open("xb") as stream:
            stream.write(payload)
        manifest["tensors"][name] = {"file": name + ".f32", "shape": list(tensor.shape),
                                     "dtype": "float32", "byteOrder": "little",
                                     "bytes": payload.nbytes, "sha256": hashlib.sha256(payload).hexdigest(),
                                     "stride": list(tensor.stride()), "contiguous": tensor.is_contiguous()}

    for name, parameter in model.named_parameters():
        require(not parameter.requires_grad, "Synthetic parameter is not frozen.")
        manifest["parameters"][name] = tensor_hash(parameter)
        manifest["parameterShapes"][name] = list(parameter.shape)
    del parameter
    expected_count, expected_elements = (197, 123650304) if variant == "l" else (517, 694659840)
    require(len(manifest["parameters"]) == expected_count and
            sum(math.prod(shape) for shape in manifest["parameterShapes"].values()) == expected_elements,
            "Full-stock parameter structure mismatch.")
    manifest["buildAndHashSeconds"] = time.perf_counter() - started
    ids = torch.tensor(case["tokens"], dtype=torch.long, device="cpu")
    repeat_hashes = []
    for iteration in range(3):
        outputs = model(ids, **case["options"])
        hashes = {name: tensor_hash(tensor) for name, tensor in zip(("final", "intermediate", "projected", "pooled"), outputs)}
        require(not repeat_hashes or hashes == repeat_hashes[0], "Source stock forward is not repeatable.")
        if iteration == 0:
            save("intermediate", outputs[1])
        repeat_hashes.append(hashes)
        del outputs
    manifest["originalForwardHashes"] = repeat_hashes
    manifest["identicalOriginalForwards"] = 3

    hooks = []

    def output_hook(module, name):
        hooks.append(module.register_forward_hook(lambda _module, _args, value: save(name, value)))

    def input_hook(module, name):
        hooks.append(module.register_forward_pre_hook(lambda _module, args: save(name, args[0])))

    try:
        output_hook(model.text_model.embeddings, "embeddings.output")
        for index, block in enumerate(model.text_model.encoder.layers):
            prefix = f"layer.{index}."
            input_hook(block, prefix + "input")
            output_hook(block.layer_norm1, prefix + "norm1")
            for projection in ("q", "k", "v"):
                output_hook(getattr(block.self_attn, projection + "_proj"), prefix + projection)
            input_hook(block.self_attn.out_proj, prefix + "attention")
            output_hook(block.self_attn.out_proj, prefix + "out_proj")
            input_hook(block.layer_norm2, prefix + "residual1")
            output_hook(block.layer_norm2, prefix + "norm2")
            output_hook(block.mlp.fc1, prefix + "fc1")
            input_hook(block.mlp.fc2, prefix + "activation")
            output_hook(block.mlp.fc2, prefix + "fc2")
            output_hook(block, prefix + "output")
        output_hook(model.text_model.final_layer_norm, "final_layer_norm")
        outputs = model(ids, **manifest["options"])
        for name, tensor in zip(("final", "all", "projected", "pooled"), outputs):
            save(name, tensor)
            if name != "all":
                require(tensor_hash(tensor) == repeat_hashes[0][name], "Diagnostic all-layer capture changed a primary output.")
        require(tensor_hash(outputs[1][:, case["options"]["intermediate_output"]]) == repeat_hashes[0]["intermediate"],
                "All-layer capture disagrees with the original selected intermediate.")
        del outputs
    finally:
        for hook in hooks:
            hook.remove()
    manifest["totalSeconds"] = time.perf_counter() - started
    temporary = directory / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    print(json.dumps({"phase": "manifest-ready", "profile": variant, "parameters": expected_count,
                      "traces": len(manifest["tensors"]), "identicalOriginalForwards": 3}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=("l", "g"), default=["l", "g"])
    parser.add_argument("--validate-only", action="store_true", help="Verify pinned inputs/ASTs/paths without importing Torch or writing outputs.")
    args = parser.parse_args()
    require(len(args.profiles) == len(set(args.profiles)), "Duplicate profiles are not allowed.")
    source, inputs, output = validate_paths(args.source_directory, args.inputs, args.output)
    cases, modules, sources = load_verified_inputs(source, inputs)
    if args.validate_only:
        print(json.dumps({"phase": "validated-without-native-runtime", "backendCommit": COMMIT,
                          "inputSha256": INPUT_SHA, "sourceFiles": len(sources), "profiles": args.profiles}))
        return
    torch, np, operations, model_type = initialize_runtime(modules)
    native = native_evidence(torch)
    output.mkdir(parents=True, exist_ok=True)
    for variant in args.profiles:
        capture_profile(variant, cases[variant], output, torch, np, model_type, operations, sources, native)
        gc.collect()
    require(digest_file(inputs) == INPUT_SHA, "Input document changed during the diagnostic.")


if __name__ == "__main__":
    main()
