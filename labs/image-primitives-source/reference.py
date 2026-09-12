"""Prospective four-node IMAGE source collector. Preflight never imports Torch or executes source."""
import argparse
import base64
import contextlib
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import types

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
PROTOCOL_SHA256 = "26ce418d969ee583ab077880530d20b6c11530094d7bbb97956bc5feaf56c0b4"
PUBLIC_FILES = ("protocol.json", "reference.py", "README.md")
NODES = ("EmptyImage", "ImageInvert", "RepeatImageBatch", "ImageFromBatch")
ARGUMENTS = {"EmptyImage": ("width", "height", "batch_size", "color"), "ImageInvert": (),
             "RepeatImageBatch": ("amount",), "ImageFromBatch": ("batch_index", "length")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf8")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON key")
        value[key] = item
    return value


def file_record(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": digest(raw)}


def load_protocol():
    raw = (BASE / "protocol.json").read_bytes()
    require(digest(raw) == PROTOCOL_SHA256, "Protocol pin mismatch")
    protocol = json.loads(raw, object_pairs_hook=unique_object)
    require(protocol["schema"] == 1 and protocol["id"] == "image-primitives-cpu-f32-bits-v1", "Protocol identity")
    cases = protocol["cases"]
    require(len(cases) == 28 and len({c["id"] for c in cases}) == 28, "Case inventory")
    require(all(sum(c["node"] == name for c in cases) == 7 for name in NODES), "Case partition")
    for case in cases:
        require(re.fullmatch("[a-z0-9-]+", case["id"]) is not None, "Unsafe case ID")
        require(list(case["arguments"]) == list(ARGUMENTS[case["node"]]), "Argument order")
        require(all(type(v) is int for v in case["arguments"].values()), "Integer arguments")
        if case["node"] == "EmptyImage":
            a = case["arguments"]
            require("image" not in case and 1 <= a["width"] <= 16384 and 1 <= a["height"] <= 16384
                    and 1 <= a["batch_size"] <= 4096 and 0 <= a["color"] <= 0xffffff, "EmptyImage domain")
            require(a["width"] * a["height"] * a["batch_size"] * 3 <= 4096, "Output bound")
        else:
            image = case["image"]
            shape, strides, offset = image["shape"], image["stride"], image["storageOffset"]
            require(image["dtype"] == "float32" and image["device"] == "cpu", "Image type")
            require(len(shape) == len(strides) == 4 and all(type(n) is int and n > 0 for n in shape + strides)
                    and shape[3] in (3, 4) and type(offset) is int and offset >= 0, "Image layout")
            raw = base64.b64decode(image["storageLittleEndianBase64"], validate=True)
            require(len(raw) == 4 * image["storageElements"] and 0 < image["storageElements"] <= 1024
                    and digest(raw) == image["storageSha256"], "Image storage integrity")
            require(offset + sum((n - 1) * s for n, s in zip(shape, strides)) < image["storageElements"], "View exceeds storage")
            require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)), "Nonfinite storage")
            # No overlapping view is admitted, including otherwise legal as_strided layouts.
            offsets = {offset + b*strides[0] + h*strides[1] + w*strides[2] + c*strides[3]
                       for b in range(shape[0]) for h in range(shape[1])
                       for w in range(shape[2]) for c in range(shape[3])}
            require(len(offsets) == math.prod(shape), "Overlapping view")
            mutation_index = case["mutationIndex"]
            require(len(mutation_index) == 4 and all(type(i) is int and 0 <= i < n for i, n in zip(mutation_index, shape)), "Mutation index domain")
            if case["node"] == "RepeatImageBatch":
                require(1 <= case["arguments"]["amount"] <= 4096, "Amount domain")
                require(math.prod(shape) * case["arguments"]["amount"] <= 4096, "Repeat output bound")
            if case["node"] == "ImageFromBatch":
                require(-16384 <= case["arguments"]["batch_index"] <= 16384
                        and 1 <= case["arguments"]["length"] <= 4096, "Extraction argument domain")
                # Admission of the separate mutation observation, not an output-value oracle.
                index = case["arguments"]["batch_index"]
                if index < 0:
                    index += shape[0]
                index = max(0, min(shape[0] - 1, index))
                end = index + min(shape[0] - index, case["arguments"]["length"])
                require(index <= mutation_index[0] < end, "Mutation must address the selected slice")
    return protocol


def verify_helpers(protocol):
    records = {}
    for record in protocol["helperFiles"]:
        path = ROOT / record["path"]
        raw = path.read_bytes()
        require(len(raw) == record["bytes"] and digest(raw) == record["sha256"], "Helper raw identity")
        require(raw == git(ROOT, "show", record["commit"] + ":" + record["path"]), "Helper public origin")
        records[record["path"]] = file_record(path)
    return records


def load_helpers(protocol):
    verify_helpers(protocol)
    result = []
    for name, relative in (("ast", "labs/autogrow-source/reference.py"),
                           ("runtime", "labs/sd-stock-source/reference.py")):
        spec = importlib.util.spec_from_file_location("_image_driver_" + name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # Verified inert laboratory definitions, never upstream declarations.
        result.append(module)
    return result


def public_evidence(commit, execute):
    records = {}
    for name in PUBLIC_FILES:
        relative = "labs/image-primitives-source/" + name
        raw = (ROOT / relative).read_bytes()
        require(b"\r" not in raw, "Public laboratory must use canonical LF bytes")
        if execute:
            require(raw == git(ROOT, "show", commit + ":" + relative), "Publish exact laboratory before execution")
        records[relative] = {"bytes": len(raw), "sha256": digest(raw)}
    return records


def lock_evidence(protocol, target, runtime):
    record, = (r for r in protocol["dependencyLocks"] if r["target"] == target)
    raw = (ROOT / record["path"]).read_bytes()
    canonical_raw = git(ROOT, "show", record["commit"] + ":" + record["path"])
    require(len(canonical_raw) == record["canonicalBytes"] and digest(canonical_raw) == record["canonicalSha256"], "Lock Git pin")
    require(raw.replace(b"\r\n", b"\n") == canonical_raw, "Lock canonical content")
    return {"path": record["path"], "originCommit": record["commit"],
            **runtime.lock_attestation(raw, record["canonicalSha256"])}


@contextlib.contextmanager
def source_modules(helper, compiled, torch):
    installed = []

    def module(label, source, dependencies=None):
        name = "_image_reference_" + label
        require(name not in sys.modules, "Source module collision")
        value = types.ModuleType(name)
        value.__dict__.update(helper.stdlib_namespace())
        value.__dict__.update(dependencies or {})
        sys.modules[name] = value
        installed.append(name)
        exec(compiled[source], value.__dict__)  # The exact hash-verified AST, after admission only.
        return value

    try:
        internal = module("internal", "comfy_api/internal/__init__.py")
        graph = module("graph_utils", "comfy_execution/graph_utils.py")
        io = module("io", "comfy_api/latest/_io.py", {
            **{name: getattr(internal, name) for name in (
                "_ComfyNodeInternal", "_NodeOutputInternal", "classproperty", "copy_class",
                "first_real_override", "is_class", "prune_dict", "shallow_clone_class")},
            "ExecutionBlocker": graph.ExecutionBlocker, "torch": torch})
        management = module("management", "comfy/model_management.py", {
            "torch": torch, "args": types.SimpleNamespace(gpu_only=False, fp16_intermediates=False)})
        legacy = module("legacy", "nodes.py", {"torch": torch, "comfy": types.SimpleNamespace(model_management=management)})
        images = module("images", "comfy_extras/nodes_images.py", {"torch": torch, "IO": io, "MAX_RESOLUTION": legacy.MAX_RESOLUTION})
        yield io, legacy, images, graph.ExecutionBlocker
    finally:
        for name in reversed(installed):
            del sys.modules[name]


def tensor_record(tensor, torch):
    require(tensor.device.type == "cpu" and tensor.dtype == torch.float32 and tensor.dim() == 4, "Output type/rank")
    shape = list(tensor.shape)
    require(all(n > 0 for n in shape) and shape[-1] in (3, 4) and tensor.numel() <= 4096, "Tensor capture domain")
    require(not tensor.requires_grad, "Unexpected output autograd")
    # Detached CPU contiguous logical capture; source tensor metadata is measured before copying.
    record = {"shape": shape, "stride": list(tensor.stride()), "storageOffset": tensor.storage_offset(),
              "aligned64": tensor.data_ptr() % 64 == 0, "dtype": "float32", "device": "cpu", "requiresGrad": tensor.requires_grad}
    raw = tensor.detach().contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
    require(len(raw) == tensor.numel() * 4 and all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)), "Capture byte/finite check")
    record.update(bytes=len(raw), sha256=digest(raw), littleEndianBase64=base64.b64encode(raw).decode("ascii"))
    return record


def image_input(descriptor, torch):
    raw = base64.b64decode(descriptor["storageLittleEndianBase64"], validate=True)
    require(sys.byteorder == "little", "Input frombuffer requires little-endian host")
    # One copied native storage per fresh run. Source receives its actual strided view.
    storage = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
    require(storage.data_ptr() % 64 == 0, "Expected native CPU allocator alignment")
    require(digest(storage.numpy().astype("<f4", copy=False).tobytes()) == descriptor["storageSha256"], "Actual input storage mismatch")
    return storage.as_strided(descriptor["shape"], descriptor["stride"], descriptor["storageOffset"])


def run_once(helper, compiled, case, torch):
    with source_modules(helper, compiled, torch) as (io, legacy, images, blocker):
        name = case["node"]
        node = getattr(legacy if name in ("EmptyImage", "ImageInvert") else images, name)
        encode = lambda value: helper.encode(value, blocker)
        if name in ("RepeatImageBatch", "ImageFromBatch"):
            node.GET_SCHEMA()
            node.RELATIVE_PYTHON_MODULE = "comfy_extras.nodes_images"
            schema = {"kind": "source-v3-object-info", "objectInfo": encode(node.GET_NODE_INFO_V1()),
                      "inputTypes": encode(node.INPUT_TYPES())}
        else:
            schema = {"kind": "source-legacy-declarations-not-complete-object-info", "inputTypes": encode(node.INPUT_TYPES()),
                      "returnTypes": encode(node.RETURN_TYPES), "function": node.FUNCTION, "category": node.CATEGORY}
        inputs = copy.deepcopy(case["arguments"])
        image = None if name == "EmptyImage" else image_input(case["image"], torch)
        before = None if image is None else tensor_record(image, torch)
        if name == "EmptyImage":
            returned = node().generate(**inputs)
        elif name == "ImageInvert":
            returned = node().invert(image=image)
        else:
            returned_v3 = node.execute(image=image, **inputs)
            require(isinstance(returned_v3, io.NodeOutput) and returned_v3.ui is None
                    and returned_v3.expand is None and returned_v3.block_execution is None, "Unexpected V3 result")
            returned = returned_v3.result
        require(isinstance(returned, tuple) and len(returned) == 1, "Expected one IMAGE output")
        result = returned[0]
        output = tensor_record(result, torch)
        after_body = None if image is None else tensor_record(image, torch)
        require(before == after_body, "Source mutated its input")
        observation = {"status": "not_applicable_no_image_input"}
        if image is not None:
            # A separate post-body property check, never a source wrapper or changed source input.
            mutation_index = tuple(case["mutationIndex"])
            original_scalar = image[mutation_index].item()
            image[mutation_index].add_(0.5)
            mutated_scalar = image[mutation_index].item()
            mutated_input, output_after_mutation = tensor_record(image, torch), tensor_record(result, torch)
            input_changed = before["sha256"] != mutated_input["sha256"]
            unchanged = output["littleEndianBase64"] == output_after_mutation["littleEndianBase64"]
            observation = {"status": "observed_after_source_return", "mutation": "image[mutationIndex].add_(0.5)", "mutationIndex": list(mutation_index),
                           "scalarBeforeF32Hex": struct.pack("<f", original_scalar).hex(),
                           "scalarAfterF32Hex": struct.pack("<f", mutated_scalar).hex(),
                           "inputAfterMutation": mutated_input, "outputAfterMutation": output_after_mutation,
                           "verification": {"inputChanged": input_changed, "outputBytesUnchanged": unchanged}}
            require(input_changed and unchanged, "Copy independence property failed")
        return {"schema": schema, "arguments": inputs, "inputBefore": before, "inputAfterBody": after_body,
                "output": output, "sourceBodyInvocations": 1, "copyObservation": observation}


def final_provenance(protocol, target, runtime, helper, source, admitted, before, sources):
    require(git(ROOT, "rev-parse", "HEAD").decode().strip() == admitted, "Collector HEAD changed")
    after = {"laboratoryFiles": public_evidence(admitted, True), "helperFiles": verify_helpers(protocol),
             "dependencyLock": lock_evidence(protocol, target, runtime), "executable": file_record(Path(sys.executable).resolve())}
    _, after_sources = helper.load_source(source, protocol)
    require(after_sources == sources and after == before, "Source/helper/lock/executable changed during collection")
    return after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Frozen backend Git repository")
    parser.add_argument("--target", choices=("win-x64", "linux-x64", "osx-arm64"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    helper, runtime = load_helpers(protocol)
    compiled, sources = helper.load_source(args.source, protocol)
    admitted = git(ROOT, "rev-parse", "HEAD").decode().strip()
    before = {"laboratoryFiles": public_evidence(admitted, args.execute), "helperFiles": verify_helpers(protocol),
              "dependencyLock": lock_evidence(protocol, args.target, runtime), "executable": file_record(Path(sys.executable).resolve())}
    evidence = {"schema": 1, "protocolId": protocol["id"], "protocolSha256": PROTOCOL_SHA256,
                "collectorCommit": admitted, "backendCommit": protocol["backendCommit"], "sourceFiles": sources,
                "helperOrigins": protocol["helperFiles"], "profile": protocol["profile"], "environment": protocol["environment"],
                "target": args.target, "provenanceBefore": before, "executed": False, "caseCount": 28}
    if not args.execute:
        require(args.output is None, "--output requires --execute")
        print(json.dumps(evidence, indent=2))
        return 0
    require(platform.python_implementation() == "CPython" and platform.python_version() == "3.12.10"
            and not sys.flags.optimize, "CPython 3.12.10 with assertions required")
    require(not os.environ.get("ATEN_CPU_CAPABILITY"), "Prospective baseline requires unset ATEN_CPU_CAPABILITY")
    actual_target = {("Windows", "AMD64"): "win-x64", ("Linux", "x86_64"): "linux-x64",
                     ("Darwin", "arm64"): "osx-arm64"}.get((platform.system(), platform.machine()))
    require(actual_target == args.target, "Target platform mismatch")
    destination = helper.output_path(args.output, args.source)
    destination.mkdir(parents=True, exist_ok=False)
    runtime.write_json(destination / "collection.json", {**evidence, "status": "started_not_completed"})
    cases = []
    try:
        # Only after public-byte admission and creation of explicit incomplete collection evidence.
        import torch
        require(torch.__version__ == ("2.10.0" if args.target == "osx-arm64" else "2.10.0+cpu")
                and torch.version.cuda is None, "Declared CPU Torch lock version mismatch")
        require(importlib.metadata.version("numpy") == "2.2.6" and importlib.metadata.version("einops") == "0.8.1", "Declared closure version mismatch")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1 and not torch.is_grad_enabled(), "Execution mode mismatch")
        native = None
        for case in protocol["cases"]:
            records = []
            for repeat in range(3):
                records.append(run_once(helper, compiled, case, torch))
                if native is None:
                    native = copy.deepcopy(runtime.runtime_identity(torch, [args.source, destination]))
                    native["buildSha256"] = digest(native.pop("build").encode("utf8"))
                    native["nativeLibraries"]["capturePoint"] = "after_first_image_body_once_per_process"
                gc.collect()
            hashes = [digest(canonical(record)) for record in records]
            require(len(set(hashes)) == 1, "Independent source repeats differ")
            captured = {"id": case["id"], "node": case["node"], "repeats": records, "repeatSha256": hashes, "bitIdentical": True}
            cases.append(captured)
            runtime.write_json(destination / (case["id"] + ".partial.json"), {"status": "partial_not_final_reference", **captured})
        after = final_provenance(protocol, args.target, runtime, helper, args.source, admitted, before, sources)
        document = {**evidence, "executed": True, "status": "completed", "provenanceAfter": after,
                    "native": native, "cases": cases, "sourceBodyInvocations": 84, "modelCompatibility": "not_assessed",
                    "limitations": ["No mapper, HTTP validation, weights, image codec or complete workflow is assessed.",
                                    "Post-body mutation observes copy independence separately from source calculation.",
                                    "Native module hashes are once-per-process observations, not per-operator dispatch evidence.",
                                    "Installed package versions and pinned lock content do not attest every installed package file."]}
        runtime.write_json(destination / "reference.json", document)
        raw = (destination / "reference.json").read_bytes()
        print(json.dumps({"status": "completed", "cases": len(cases), "sourceBodyInvocations": 84, "bytes": len(raw), "sha256": digest(raw)}))
        return 0
    except BaseException as error:
        runtime.write_json(destination / "incomplete.json", {"status": "incomplete", "errorType": type(error).__name__, "completedCases": len(cases)})
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted"}), file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(json.dumps({"status": "failed", "errorType": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)
