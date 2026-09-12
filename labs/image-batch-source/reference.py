"""Prospective ImageBatch source laboratory. Preflight is inert; execution requires the exact published files."""
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
ROOT = next(p for p in BASE.parents if (p / "ComfySharp.slnx").is_file())
PUBLIC_DIRECTORY = "labs/image-batch-source"
PROTOCOL_SHA256 = "6d3f1e09482713beb02497063dd2e3721032fd8a24ff486e5deed9eb23461962"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate property")
        result[key] = value
    return result


def protocol():
    raw = (BASE / "protocol.json").read_bytes()
    require(digest(raw) == PROTOCOL_SHA256, "Protocol pin mismatch")
    p = json.loads(raw, object_pairs_hook=unique_object)
    require(p["id"] == "image-batch-cpu-f32-v1" and p["schema"] == 1, "Protocol identity")
    cases = p["cases"]
    require(len(cases) == len({c["id"] for c in cases}) == 19, "Case inventory")
    require({k: sum(c["classification"] == k for c in cases) for k in ("exact-no-resize", "bilinear", "source-error")}
            == {"exact-no-resize": 4, "bilinear": 14, "source-error": 1}, "Partition")
    for case in cases:
        require(case["node"] == "ImageBatch" and re.fullmatch("[a-z0-9-]+", case["id"]), "Case ID")
        for key in ("image1", "image2"):
            image = case[key]
            shape, stride, offset = image["shape"], image["stride"], image["storageOffset"]
            require(image["dtype"] == "float32" and image["device"] == "cpu", "Input type")
            require(len(shape) == len(stride) == 4 and all(type(n) is int and n > 0 for n in shape + stride)
                    and shape[-1] in (3, 4) and type(offset) is int and offset >= 0, "Input layout")
            raw = base64.b64decode(image["storageLittleEndianBase64"], validate=True)
            require(len(raw) == 4 * image["storageElements"] and 0 < image["storageElements"] <= 2048
                    and digest(raw) == image["storageSha256"], "Input storage bytes")
            require(all(math.isfinite(v[0]) and abs(v[0]) <= 32 for v in struct.iter_unpack("<f", raw)), "Input value bound")
            offsets = {offset + b*stride[0] + h*stride[1] + w*stride[2] + c*stride[3]
                       for b in range(shape[0]) for h in range(shape[1]) for w in range(shape[2]) for c in range(shape[3])}
            require(len(offsets) == math.prod(shape) and max(offsets) < image["storageElements"], "Overlapping/out-of-storage view")
            require(case["mutationIndices"][key] == [0, 0, 0, 0], "Mutation scalar")
        s1, s2 = case["image1"]["shape"], case["image2"]["shape"]
        require((s1[0] + s2[0]) * s1[1] * s1[2] * max(s1[3], s2[3]) <= 4096, "Output bound")
        require((s1[1:3] == s2[1:3]) == (case["classification"] == "exact-no-resize"), "Resize classification")
        require(case["permittedSourceErrors"] == ([{"stage": "body", "type": "RuntimeError"}]
                if case["classification"] == "source-error" else []), "Error admission")
    return p


def dependencies(p):
    # Verify every helper/origin before importing the first inert driver.
    for record in p["helperFiles"]:
        raw = (ROOT / record["path"]).read_bytes()
        require(len(raw) == record["bytes"] and digest(raw) == record["sha256"], "Helper identity")
        require(raw == git(ROOT, "show", record["commit"] + ":" + record["path"]), "Helper public origin")
    spec = importlib.util.spec_from_file_location("_image_batch_image_driver", ROOT / "labs/image-primitives-source/reference.py")
    image = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image)  # Inert verified driver only.
    ast_helper, runtime = image.load_helpers(p)
    return image, ast_helper, runtime


def public_evidence(admitted, execute):
    if execute:
        require(BASE == (ROOT / PUBLIC_DIRECTORY).resolve(), "Ignored draft cannot execute; promote and publish first")
    records = {}
    for name in ("protocol.json", "reference.py", "README.md"):
        raw = (BASE / name).read_bytes()
        require(b"\r" not in raw, "Canonical LF public bytes required")
        relative = PUBLIC_DIRECTORY + "/" + name
        if execute:
            require(raw == git(ROOT, "show", admitted + ":" + relative), "Publish exact laboratory before execution")
        records[relative] = {"bytes": len(raw), "sha256": digest(raw)}
    return records


def provenance(p, source, target, admitted, execute, image, helper, runtime):
    compiled, sources = helper.load_source(source, p)  # Compile only, no source exec.
    return compiled, {"collectorCommit": git(ROOT, "rev-parse", "HEAD").decode().strip(),
                      "laboratoryFiles": public_evidence(admitted, execute), "sourceFiles": sources,
                      "helperFiles": image.verify_helpers(p), "dependencyLock": image.lock_evidence(p, target, runtime),
                      "executable": image.file_record(Path(sys.executable).resolve())}


@contextlib.contextmanager
def source_modules(compiled, torch):
    installed = []
    def module(name, path, dependencies):
        require(name not in sys.modules, "Source module collision")
        value = types.ModuleType(name)
        value.__dict__.update(dependencies)
        sys.modules[name] = value
        installed.append(name)
        exec(compiled[path], value.__dict__)  # Original AST declarations only, after admission.
        return value
    try:
        utils = module("_image_batch_utils", "comfy/utils.py", {"torch": torch})
        nodes = module("_image_batch_nodes", "nodes.py", {"torch": torch, "comfy": types.SimpleNamespace(utils=utils)})
        yield nodes.ImageBatch, utils.common_upscale
    finally:
        for name in reversed(installed):
            del sys.modules[name]


def layout(tensor):
    # Metadata-only observation also permits the empty crop of the classified error.
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "storageOffset": tensor.storage_offset(),
            "dtype": str(tensor.dtype), "device": str(tensor.device)}


def run_once(case, compiled, torch, image, observe):
    with source_modules(compiled, torch) as (node, upscale):
        schema = {"kind": "source-legacy-declarations-not-complete-object-info", "inputTypes": node.INPUT_TYPES(),
                  "returnTypes": node.RETURN_TYPES, "function": node.FUNCTION, "category": node.CATEGORY,
                  "deprecated": node.DEPRECATED, "searchAliases": node.SEARCH_ALIASES}
        inputs = {key: image.image_input(case[key], torch) for key in ("image1", "image2")}
        before = {key: image.tensor_record(value, torch) for key, value in inputs.items()}
        observation, observation_errors = [], []
        def profile(frame, event, arg):
            if frame.f_code is upscale.__code__ and event == "return":
                try:
                    local = frame.f_locals
                    observation.append({"capture": "actual_common_upscale_return", "exceptionalReturn": arg is None,
                        "x": local["x"], "y": local["y"], "oldWidth": local["old_width"], "oldHeight": local["old_height"],
                        "oldAspect": local["old_aspect"], "newAspect": local["new_aspect"],
                        "requestedWidth": local["width"], "requestedHeight": local["height"],
                        "method": local["upscale_method"], "crop": local["crop"],
                        "samples": layout(local["samples"]), "cropped": layout(local["s"]),
                        "output": None if arg is None else layout(arg)})
                except Exception as error:
                    observation_errors.append(type(error).__name__)
                    raise RuntimeError("Observation failed") from error
        require(sys.getprofile() is None, "External profiler would confound observation")
        returned = None
        try:
            if observe:
                sys.setprofile(profile)
            try:
                returned = node().batch(inputs["image1"], inputs["image2"])
            finally:
                if observe:
                    sys.setprofile(None)
            require(case["classification"] != "source-error", "Prospective source error unexpectedly returned")
            require(type(returned) is tuple and len(returned) == 1, "Expected one IMAGE output")
            record = {"status": "returned", "output": image.tensor_record(returned[0], torch)}
        except RuntimeError as error:
            require({"stage": "body", "type": type(error).__name__} in case["permittedSourceErrors"], "Unadmitted source failure")
            record = {"status": "raised", "stage": "body", "errorType": type(error).__name__}
        require(not observation_errors, "Observer errors must never be accepted as source errors")
        after = {key: image.tensor_record(value, torch) for key, value in inputs.items()}
        require(before == after, "Source input mutation")
        record.update(schema=schema, inputBefore=before, inputAfterBody=after, sourceBodyInvocations=1)
        if returned is not None:
            output = returned[0]
            mutations = []
            for key in ("image1", "image2"):
                index = tuple(case["mutationIndices"][key])
                old = inputs[key][index].item()
                inputs[key][index].add_(0.5)
                changed = image.tensor_record(inputs[key], torch)
                output_after = image.tensor_record(output, torch)
                require(before[key]["sha256"] != changed["sha256"], "Mutation must change input")
                require(output_after == record["output"], "Output storage independence failed")
                mutations.append({"input": key, "index": list(index), "scalarBeforeF32Hex": struct.pack("<f", old).hex(),
                    "scalarAfterF32Hex": struct.pack("<f", inputs[key][index].item()).hex(),
                    "inputsAfterMutation": {k: image.tensor_record(t, torch) for k, t in inputs.items()},
                    "outputAfterMutation": output_after, "verification": {"inputChanged": True, "outputUnchanged": True}})
            record["copyObservation"] = {"status": "separate_post_body_mutations", "sequence": mutations}
        else:
            record["copyObservation"] = {"status": "not_applicable_source_error_no_output"}
        if observe:
            require(len(observation) == (0 if case["classification"] == "exact-no-resize" else 1), "Unexpected observed resize count")
            if case["classification"] == "source-error":
                require(record["status"] == "raised" and observation[0]["exceptionalReturn"] is True
                        and any(n == 0 for n in observation[0]["cropped"]["shape"][-2:]),
                        "Expected failure at the observed empty spatial crop")
            record["observation"] = observation
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", choices=("win-x64", "linux-x64", "osx-arm64"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    p = protocol()
    image, helper, runtime = dependencies(p)
    admitted = git(ROOT, "rev-parse", "HEAD").decode().strip()
    compiled, before = provenance(p, args.source, args.target, admitted, args.execute, image, helper, runtime)
    document = {"schema": 1, "protocolId": p["id"], "protocolSha256": PROTOCOL_SHA256,
                "backendCommit": p["backendCommit"], "collectorCommit": admitted, "target": args.target,
                "profile": p["profile"], "environment": p["environment"], "helperOrigins": p["helperFiles"],
                "provenanceBefore": before, "executed": False, "caseCount": 19}
    if not args.execute:
        require(args.output is None, "--output requires --execute")
        print(json.dumps(document, indent=2))
        return 0
    require(platform.python_implementation() == "CPython" and platform.python_version() == "3.12.10"
            and not sys.flags.optimize and not os.environ.get("ATEN_CPU_CAPABILITY"), "Execution environment mismatch")
    target = {("Windows", "AMD64"): "win-x64", ("Linux", "x86_64"): "linux-x64", ("Darwin", "arm64"): "osx-arm64"}.get((platform.system(), platform.machine()))
    require(target == args.target, "Target mismatch")
    output = helper.output_path(args.output, args.source)
    output.mkdir(parents=True, exist_ok=False)
    runtime.write_json(output / "collection.json", {**document, "status": "started_not_completed"})
    cases = []
    try:
        import torch
        require(torch.__version__ == ("2.10.0" if target == "osx-arm64" else "2.10.0+cpu") and torch.version.cuda is None, "CPU Torch version mismatch")
        require(importlib.metadata.version("numpy") == "2.2.6" and importlib.metadata.version("einops") == "0.8.1", "Closure version mismatch")
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1 and not torch.is_grad_enabled(), "Execution mode mismatch")
        native = None
        for case in p["cases"]:
            records = []
            for observe in (False, True, False):
                records.append(run_once(case, compiled, torch, image, observe))
                if native is None:
                    native = copy.deepcopy(runtime.runtime_identity(torch, [args.source, output]))
                    native["buildSha256"] = digest(native.pop("build").encode())
                    native["nativeLibraries"]["capturePoint"] = "after_first_image_batch_body_once_per_process"
                gc.collect()
            neutral = [{k: v for k, v in record.items() if k != "observation"} for record in records]
            hashes = [digest(image.canonical(record)) for record in neutral]
            require(len(set(hashes)) == 1, "Observation or fresh repeat changed source result")
            captured = {"id": case["id"], "classification": case["classification"], "repeats": records,
                        "neutralRepeatSha256": hashes, "bitIdentical": True}
            cases.append(captured)
            runtime.write_json(output / (case["id"] + ".partial.json"), {"status": "partial_not_final_reference", **captured})
        _, after = provenance(p, args.source, args.target, admitted, True, image, helper, runtime)
        require(before == after, "Provenance changed during collection")
        document.update(executed=True, status="completed", provenanceAfter=after, native=native, cases=cases,
                        sourceBodyInvocations=57, modelCompatibility="not_assessed",
                        limitations=["Source bodies and separate observations only, not mapper/HTTP or product qualification.",
                                     "Bilinear tolerance is prospective and unapplied here; no product outputs are read.",
                                     "Native libraries observed once; package versions and locks do not attest every installed file."])
        runtime.write_json(output / "reference.json", document)
        raw = (output / "reference.json").read_bytes()
        print(json.dumps({"status": "completed", "cases": 19, "sourceBodyInvocations": 57, "bytes": len(raw), "sha256": digest(raw)}))
        return 0
    except BaseException as error:
        runtime.write_json(output / "incomplete.json", {"status": "incomplete", "errorType": type(error).__name__, "completedCases": len(cases)})
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
