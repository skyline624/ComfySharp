"""Prospective source-only prompt-value laboratory; default preflight executes no source."""
import argparse
import ast
import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import math
import pathlib
import platform
import struct
import subprocess
import sys
import traceback
import types
import typing

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent.parent
PROTOCOL_SHA256 = "dbd4038623e3313005cbb29393da978c0d8c9eb49a96860fb765ff217b6fe2ee"
PUBLIC_FILES = ("protocol.json", "reference.py", "README.md")
MAX_OUTPUT_BYTES = 16 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON property")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Non-finite JSON is outside the protocol")


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def identity(raw):
    return {"bytes": len(raw), "sha256": digest(raw)}


def selected_node(tree, name):
    if name == "@future-annotations":
        found = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    elif name == "@setup-dynamic":
        found = [n for n in tree.body if isinstance(n, ast.If) and "DYNAMIC_INPUT_LOOKUP" in ast.unparse(n.test)]
    else:
        domain = ast.walk(tree) if name == "execution_block_cb" else tree.body
        found = [n for n in domain if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        if not found:
            found = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign)) and any(
                isinstance(t, ast.Name) and t.id == name for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    if len(found) != 1:
        raise ValueError("Missing or ambiguous pinned declaration: " + name)
    return found[0]


def compile_sources(source_root, commit, records):
    compiled = {}
    for record in records:
        raw = git(source_root, "show", commit + ":" + record["path"])
        if identity(raw) != {k: record[k] for k in ("bytes", "sha256")}:
            raise ValueError("Pinned backend blob mismatch: " + record["path"])
        text = raw.decode("utf-8")
        tree = ast.parse(text, filename=record["path"])
        nodes = []
        for declaration in record["declarations"]:
            node = selected_node(tree, declaration["name"])
            if (node.lineno != declaration["line"] or
                    digest(ast.dump(node, include_attributes=False).encode()) != declaration["astSha256"] or
                    digest(ast.get_source_segment(text, node).encode()) != declaration["segmentSha256"]):
                raise ValueError("Pinned AST mismatch: " + declaration["name"])
            if any(isinstance(n, (ast.Import, ast.ImportFrom)) and not
                   (isinstance(n, ast.ImportFrom) and n.module == "__future__") for n in ast.walk(node)):
                raise ValueError("Selected source imports an external module")
            nodes.append(node)
        compiled[record["path"]] = compile(ast.Module(body=nodes, type_ignores=[]),
                                           record["path"], "exec", dont_inherit=True)
    return compiled


def prepare(source_root):
    """Only read/hash/parse/compile. No exec, upstream import or constructor."""
    raw = (BASE / "protocol.json").read_bytes()
    if digest(raw) != PROTOCOL_SHA256 or b"\r" in raw:
        raise ValueError("Published LF protocol identity mismatch")
    protocol = parse(raw)
    if (protocol["schema"] != 1 or protocol["id"] != "prompt-values-boundaries-v1" or
            len(protocol["cases"]) != 28 or len({c["id"] for c in protocol["cases"]}) != 28 or
            sum(c["secondValidationPass"] for c in protocol["cases"]) != 3):
        raise ValueError("Unexpected prospective case set")
    helpers = {}
    for item in protocol["sourceReuse"]["files"]:
        local = (ROOT / item["path"]).read_bytes()
        published = git(ROOT, "show", protocol["sourceReuse"]["commit"] + ":" + item["path"])
        if identity(local) != {k: item[k] for k in ("bytes", "sha256")} or local != published:
            raise ValueError("Pinned helper differs from published bytes: " + item["path"])
        helpers[item["path"]] = local
    inherited = parse(helpers["labs/autogrow-source/protocol.json"])
    if inherited["backendCommit"] != protocol["backendCommit"]:
        raise ValueError("Inherited backend identity mismatch")
    original = compile_sources(source_root, protocol["backendCommit"], inherited["sources"])
    additional = compile_sources(source_root, protocol["backendCommit"], protocol["additionalSources"])
    helper_code = compile(helpers["labs/autogrow-source/reference.py"], "labs/autogrow-source/reference.py", "exec", dont_inherit=True)
    evidence = {"protocol": identity(raw), "helpers": protocol["sourceReuse"],
                "baseSources": inherited["sources"], "additionalSources": protocol["additionalSources"]}
    return protocol, original, additional, helper_code, evidence


def typed(value):
    """Unambiguous public encoding, preserving runtime scalar types and float bits."""
    if value is None:
        return {"kind": "null"}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "decimal": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Non-finite observation is outside this protocol")
        return {"kind": "float64", "bitsLE": struct.pack("<d", value).hex(), "repr": repr(value)}
    if isinstance(value, str):
        return {"kind": "string", "value": str(value)}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("Non-string dictionary key is outside JSON prompt profile")
        return {"kind": "dict", "items": [[str(k), typed(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list", "items": [typed(v) for v in value]}
    raise TypeError("Unqualified observation type: " + type(value).__name__)


def public_validation(value):
    """Keep exact values; explicitly omit traceback text, which is not compared."""
    if isinstance(value, dict):
        return {k: {"tracebackOmitted": True, "frameCount": len(v)} if k == "traceback" and isinstance(v, list)
                else public_validation(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(public_validation(v) for v in value)
    if isinstance(value, list):
        return [public_validation(v) for v in value]
    return value


@contextlib.contextmanager
def namespaces(helper, original, additional):
    """All source exec occurs here, only after main's publication/version/output guards."""
    installed = []
    with helper.source_modules(original) as (io, execution, create_list, blocker):
        def module(label, code, bindings):
            name = "_prompt_values_source_" + label
            if name in sys.modules:
                raise RuntimeError("Source laboratory namespace already exists")
            mod = types.ModuleType(name)
            mod.__dict__.update(bindings)
            sys.modules[name] = mod
            installed.append(name)
            exec(code, mod.__dict__)
            return mod
        try:
            exec(additional["comfy_api/latest/_io.py"], io.__dict__)
            primitives = module("primitives", additional["comfy_extras/nodes_primitive.py"], {"io": io, "sys": sys})
            legacy_io = sys.modules["_autogrow_reference_legacy"].IO
            preview = module("preview", additional["comfy_extras/nodes_preview_any.py"], {"IO": legacy_io, "json": json})
            validation = module("validation", additional["comfy_execution/validation.py"], {"IO": io})
            registry = {"CreateList": create_list, "PrimitiveString": primitives.String,
                        "PrimitiveInt": primitives.Int, "PrimitiveFloat": primitives.Float,
                        "PrimitiveBoolean": primitives.Boolean, "PreviewAny": preview.PreviewAny}
            first_override = sys.modules["_autogrow_reference_internal"].first_real_override
            execution.__dict__.update(nodes=types.SimpleNamespace(NODE_CLASS_MAPPINGS=registry),
                                      first_real_override=first_override,
                                      validate_node_input=validation.validate_node_input,
                                      sys=sys, traceback=traceback, logging=logging, Union=typing.Union)
            exec(additional["execution.py"], execution.__dict__)
            for name, node in registry.items():
                if node is not preview.PreviewAny:
                    node.GET_SCHEMA()
                    node.RELATIVE_PYTHON_MODULE = "comfy_extras.nodes_toolkit" if node is create_list else "comfy_extras.nodes_primitive"
                    if first_override(node, "validate_inputs") is not None:
                        raise RuntimeError("Custom validation is outside this protocol")
                elif getattr(node, "VALIDATE_INPUTS", None) is not None:
                    raise RuntimeError("Custom validation is outside this protocol")
            # PreviewAny's real main has a torch reference; it is deliberately never invoked.
            if "torch" in preview.__dict__:
                raise RuntimeError("No Torch binding is permitted")
            forbidden = {getattr(node, node.FUNCTION).__func__.__code__ if hasattr(getattr(node, node.FUNCTION), "__func__")
                         else getattr(node, node.FUNCTION).__code__ for node in registry.values()}
            if sys.getprofile() is not None:
                raise RuntimeError("External profiler is not permitted")
            forbidden_attempts = []
            def forbid_node_execution(frame, event, arg):
                if event == "call" and frame.f_code in forbidden:
                    forbidden_attempts.append(frame.f_code.co_name)
                    raise RuntimeError("Node execution is forbidden in this validation laboratory")
            sys.setprofile(forbid_node_execution)
            try:
                yield execution, registry
            finally:
                sys.setprofile(None)
                if forbidden_attempts:
                    raise RuntimeError("A forbidden node call was attempted, even if source caught its exception")
        finally:
            for name in reversed(installed):
                del sys.modules[name]


def prompt_for(protocol, case):
    prompt = copy.deepcopy(protocol["infrastructure"]["producerPrompt"])
    inputs = {"inputs.input0": "anchor"} if case["optionalAnchor"] else {}
    if case["present"]:
        inputs[case["inputName"]] = copy.deepcopy(case["value"])
    prompt["subject"] = {"class_type": case["classType"], "inputs": inputs}
    prompt["preview"] = {"class_type": protocol["infrastructure"]["outputClass"],
                         "inputs": copy.deepcopy(protocol["infrastructure"]["outputInput"])}
    return prompt


def acquire(execution, node, prompt, protocol, permitted_errors=()):
    inputs = copy.deepcopy(prompt["subject"]["inputs"])
    original_inputs = typed(inputs)
    cache = copy.deepcopy(protocol["infrastructure"]["cache"])
    original_cache = typed(cache)
    reads = []
    class FixtureCache:
        def get_cache(self, producer, consumer):
            reads.append([producer, consumer])
            return types.SimpleNamespace(outputs=cache[producer]) if producer in cache else None
    try:
        flat, missing, v3 = execution.get_input_data(inputs, node, "subject", execution_list=FixtureCache())
        result = {"status": "returned", "flat": typed(flat), "flatOrder": list(flat),
                  "missing": typed(missing), "v3Keys": list(v3)}
    except (TypeError, IndexError, KeyError, ValueError) as error:
        # These are observed source outcomes, never converted into acceptance or skipped cases.
        if type(error).__name__ not in permitted_errors:
            raise
        result = {"status": "raised", "errorType": type(error).__name__, "errorMessage": str(error)}
    result.update(cacheReads=typed(reads), inputsAfter=typed(inputs), cacheAfter=typed(cache))
    if typed(inputs) != original_inputs or typed(cache) != original_cache:
        raise AssertionError("Acquisition changed a fixture")
    return result


def validation_exception_types(value):
    if isinstance(value, dict):
        if "exception_type" in value:
            yield value["exception_type"]
        for item in value.values():
            yield from validation_exception_types(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from validation_exception_types(item)


async def validate(execution, prompt, permitted_exceptions):
    before = typed(prompt)
    returned = await execution.validate_prompt("fixture-prompt", prompt, ["preview"])
    if not isinstance(returned, tuple) or len(returned) != 4 or type(returned[0]) is not bool:
        raise TypeError("Unexpected exact source validate_prompt contract")
    if any(name not in permitted_exceptions for name in validation_exception_types(returned)):
        raise RuntimeError("Unexpected source validation exception; no dependency fallback is permitted")
    return {"status": "returned", "accepted": returned[0],
            "result": typed(public_validation(returned)), "promptBefore": before, "promptAfter": typed(prompt)}


async def run_case(helper, original, additional, protocol, case):
    with namespaces(helper, original, additional) as (execution, registry):
        original_prompt = prompt_for(protocol, case)
        pristine = typed(original_prompt)
        node = registry[case["classType"]]
        valid_inputs = node.INPUT_TYPES()
        if issubclass(node, execution._ComfyNodeInternal):
            valid_inputs, _, _ = execution._io.get_finalized_class_inputs(
                valid_inputs, copy.deepcopy(original_prompt["subject"]["inputs"]))
        info = execution.get_input_info(node, case["inputName"], valid_inputs)
        record = {"id": case["id"], "inputInfo": typed(info),
                  "recognitionOriginal": {"inputPresent": case["present"], "predicateInvoked": case["present"]}}
        if case["present"]:
            record["recognitionOriginal"].update(value=typed(case["value"]), isLink=execution.is_link(case["value"]))
        record["acquisitionOriginal"] = acquire(execution, node, copy.deepcopy(original_prompt), protocol,
                                                case["permittedAcquisitionErrors"])
        normalized = copy.deepcopy(original_prompt)
        record["validation"] = await validate(execution, normalized, case["permittedValidationExceptions"])
        record["acquisitionAfterValidation"] = acquire(execution, node, normalized, protocol) if record["validation"]["accepted"] else {
            "status": "not-run", "reason": "source-validation-rejected"}
        if case["secondValidationPass"]:
            # This is the sole intentional reuse of mutated data, labeled separately.
            second = copy.deepcopy(normalized)
            record["secondValidation"] = await validate(execution, second, case["permittedValidationExceptions"])
            record["acquisitionAfterSecondValidation"] = acquire(execution, node, second, protocol) if record["secondValidation"]["accepted"] else {
                "status": "not-run", "reason": "source-second-validation-rejected"}
        if typed(original_prompt) != pristine:
            raise AssertionError("Original protocol prompt was mutated")
        record["originalPromptUnchanged"] = True
        return record


async def collect(helper, original, additional, protocol):
    results = []
    for case in protocol["cases"]:
        repetitions = [await run_case(helper, original, additional, protocol, case) for _ in range(3)]
        hashes = [digest(canonical(r)) for r in repetitions]
        if len(set(hashes)) != 1:
            raise AssertionError("Fresh source observations were not repeatable: " + case["id"])
        record = repetitions[0]
        record["repeats"] = {"count": 3, "freshSourceNamespaces": True, "sha256": hashes, "bitIdentical": True}
        results.append(record)
    return results


def output_path(value, source):
    if value is None or not value.is_absolute() or value.exists():
        raise ValueError("Execution requires a new absolute output directory")
    path = value.resolve()
    for protected in (ROOT, source, pathlib.Path(sys.base_prefix)):
        protected = protected.resolve()
        if path.is_relative_to(protected) or protected.is_relative_to(path):
            raise ValueError("Output overlaps a protected repository or interpreter directory")
    return path


def public_files(commit, execute):
    records = {}
    for name in PUBLIC_FILES:
        relative = "labs/prompt-values-source/" + name
        raw = (ROOT / relative).read_bytes()
        if b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("Laboratory files must be UTF8 LF without BOM")
        raw.decode("utf-8")
        if execute and raw != git(ROOT, "show", commit + ":" + relative):
            raise ValueError("Publish exact new laboratory bytes before execution")
        records[relative] = identity(raw)
    return records


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from strings(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-commit")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    protocol, original, additional, helper_code, provenance = prepare(args.source)
    admitted = git(ROOT, "rev-parse", "HEAD").decode().strip()
    public = public_files(admitted, args.execute)
    evidence = {"schema": 1, "protocolId": protocol["id"], "executed": False,
                "cases": len(protocol["cases"]), "provenance": provenance, "laboratoryFiles": public}
    if not args.execute:
        if args.output is not None or args.expected_commit is not None:
            parser.error("--output and --expected-commit require --execute")
        print(json.dumps(evidence, indent=2, allow_nan=False))
        return
    if args.expected_commit != admitted:
        raise ValueError("Execution requires the explicit published HEAD commit")
    if platform.python_version() != protocol["python"] or sys.flags.optimize:
        raise ValueError("Exact prospective Python version and enabled assertions are required")
    if sys.getprofile() is not None or any(n == "torch" or n.startswith("torch.") for n in sys.modules):
        raise ValueError("External profiler or native Torch import is not permitted")
    destination = output_path(args.output, args.source)
    destination.mkdir(parents=True, exist_ok=False)
    helper_name = "_prompt_values_pinned_helper"
    if helper_name in sys.modules:
        raise RuntimeError("Helper namespace already exists")
    helper = types.ModuleType(helper_name)
    helper.__file__ = str(ROOT / "labs/autogrow-source/reference.py")
    sys.modules[helper_name] = helper
    try:
        exec(helper_code, helper.__dict__)
        results = asyncio.run(collect(helper, original, additional, protocol))
        if git(ROOT, "rev-parse", "HEAD").decode().strip() != admitted:
            raise ValueError("HEAD changed during collection")
        if public_files(admitted, True) != public:
            raise ValueError("New laboratory bytes changed during collection")
        _, _, _, _, after = prepare(args.source)
        if after != provenance:
            raise ValueError("Source or pinned helper provenance changed during collection")
        evidence.update(executed=True, collectorCommit=admitted, backendCommit=protocol["backendCommit"],
                        python=platform.python_version(), platform=platform.system(),
                        sourceNodeExecution=False, referenceOutputs=results,
                        limits=[protocol["scope"], protocol["comparison"], protocol["sanitization"],
                                "Cache and registry are explicit fixture infrastructure; no node execution or server ran.",
                                "Rejected validation and non-run dependent acquisition remain visible, not accepted parity.",
                                "All observed values are source results; no C# output is consumed."])
        raw = (json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(raw) > MAX_OUTPUT_BYTES:
            raise ValueError("Observation payload exceeds the fixed bound")
        protected = [str(p.resolve()) for p in (ROOT, args.source, destination, pathlib.Path(sys.base_prefix))]
        if any(path in text or path.replace("\\", "/") in text
               for text in strings(evidence) for path in protected):
            raise ValueError("Private path found in public observations")
        with (destination / "reference.json").open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"cases": len(results), **identity(raw)}))
    except Exception as error:
        marker = {"executed": False, "complete": False, "errorType": type(error).__name__,
                  "message": "Collection incomplete; no qualification artifact is accepted."}
        with (destination / "incomplete.json").open("xb") as stream:
            stream.write((json.dumps(marker, indent=2) + "\n").encode("utf-8"))
        raise
    finally:
        del sys.modules[helper_name]


if __name__ == "__main__":
    main()
