"""Prospective real StringFormat collector. Default preflight executes no source."""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import types

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
PUBLIC_FILES = ("protocol.json", "reference.py", "README.md")
PROTOCOL_SHA256 = "2eae9a7d3719a225110e032e4b12233300c7ae168a6659d140d87778a34aaefd"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON property: " + key)
        result[key] = value
    return result


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_protocol():
    raw = (BASE / "protocol.json").read_bytes()
    if digest(raw) != PROTOCOL_SHA256:
        raise ValueError("Prospective protocol identity mismatch")
    value = json.loads(raw, object_pairs_hook=unique_object)
    if value["schema"] != 1 or value["id"] != "string-format-text-v1":
        raise ValueError("Unexpected protocol")
    if len(value["cases"]) != 44 or len({c["id"] for c in value["cases"]}) != 44:
        raise ValueError("Expected 44 unique prospective cases")
    if {kind: sum(c["portClassification"] == kind for c in value["cases"])
            for kind in ("comparable", "source-error", "unsupported-profile")} != {
            "comparable": 24, "source-error": 12, "unsupported-profile": 8}:
        raise ValueError("Prospective partition mismatch")
    return value


def verify_helpers(protocol):
    for record in protocol["helperFiles"]:
        raw = (ROOT / record["path"]).read_bytes()
        if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
            raise ValueError("Published helper identity mismatch: " + record["path"])
        if raw != git(ROOT, "show", record["commit"] + ":" + record["path"]):
            raise ValueError("Published helper commit mismatch: " + record["path"])


def load_helpers(protocol):
    # All six raw files are verified before importing any driver definitions.
    verify_helpers(protocol)
    spec = importlib.util.spec_from_file_location(
        "_string_format_names_driver", ROOT / "labs/autogrow-names-source/reference.py")
    names = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(names)  # Inert driver definitions; no source declarations.
    old = names.load_helper(names.load_protocol())
    return names, old


async def run_case(names, old, compiled, case, observe):
    result = {"id": case["id"], "portClassification": case["portClassification"],
              "bodyOrigin": "source.StringFormat.execute -> builtin str.format"}
    with names.source_modules(old, compiled) as (io, execution, node, blocker_type):
        encode = lambda value: old.encode(value, blocker_type)
        stage = "schema"
        prompt = copy.deepcopy(case["promptInputs"])
        cache = old.decode_cache(copy.deepcopy(case["cache"]), blocker_type)
        before_prompt, before_cache = canonical(encode(prompt)), canonical(encode(cache))
        reports, cache_reads, body_calls, nested_returns = [], [], [], []
        execution.server = types.SimpleNamespace(client_id="fixture-client")
        execution.server.send_sync = lambda event, payload, client: reports.append(
            {"event": event, "payload": payload, "client": client})
        execution.prompt_id, execution.unique_id = "fixture-prompt", case["id"]
        execution.class_type, execution.executed = "StringFormat", ["fixture-upstream"]

        class FixtureCache:
            def get_cache(self, producer, consumer):
                cache_reads.append([producer, consumer])
                return types.SimpleNamespace(outputs=cache[producer]) if producer in cache else None

        try:
            node.GET_SCHEMA()  # Source loader ordering precedes module attribution.
            node.RELATIVE_PYTHON_MODULE = "comfy_extras.nodes_string"
            result["objectInfo"] = encode(node.GET_NODE_INFO_V1())
            input_types = node.INPUT_TYPES()
            result["inputTypes"] = encode(input_types)
            finalized, hidden, v3 = io.get_finalized_class_inputs(input_types, prompt)
            result.update(finalized=encode(finalized), finalizedOrder={k: list(v) for k, v in finalized.items()},
                          hidden=encode(hidden), v3Data=encode(v3), dynamicPathOrder=list(v3.get("dynamic_paths", {})))
            stage = "acquisition"
            flat, missing, actual_v3 = execution.get_input_data(
                prompt, node, case["id"], execution_list=FixtureCache())
            result.update(flatInputs=encode(flat), flatOrder=list(flat), missingKeys=encode(missing),
                          acquiredV3Data=encode(actual_v3), cacheReads=cache_reads)
            stage = "mapper"
            body_code = node.execute.__func__.__code__
            binder_code = io.build_nested_inputs.__code__

            def profile(frame, event, arg):
                try:
                    if frame.f_code is body_code and event == "call":
                        values = frame.f_locals["values"]
                        body_calls.append({"values": encode(values), "valueOrder": list(values),
                                           "f_string": frame.f_locals["f_string"]})
                    elif frame.f_code is binder_code and event == "return":
                        nested_returns.append({"arguments": encode(arg), "rootOrder": list(arg),
                                               "valueOrder": list(arg["values"])})
                except Exception as error:
                    raise RuntimeError("Source observation failed") from error

            if sys.getprofile() is not None:
                raise RuntimeError("An external profiler would confound observations")
            if observe:
                sys.setprofile(profile)
            try:
                mapped = await execution._async_map_node_over_list(
                    "fixture-prompt", case["id"], node, flat, node.FUNCTION,
                    allow_interrupt=False, execution_block_cb=execution.execution_block_cb, v3_data=actual_v3)
                returned = await execution.resolve_map_node_over_list_results(mapped)
            finally:
                if observe:
                    sys.setprofile(None)
            stage = "merge"
            outputs, ui, expanded = execution.get_output_from_returns(returned, node)
            result.update(status="returned", outputs=encode(outputs), ui=encode(ui), hasSubgraph=expanded)
        except (AssertionError, ValueError, TypeError, IndexError, KeyError, OverflowError) as error:
            if {"stage": stage, "type": type(error).__name__} not in case["permittedSourceErrors"]:
                raise
            result.update(status="raised", stage=stage, errorType=type(error).__name__, errorMessage=str(error))
        if case["portClassification"] == "source-error" and result["status"] != "raised":
            raise AssertionError("Prospectively erroneous case unexpectedly returned: " + case["id"])
        result.update(reports=reports, inputsAfter=encode(prompt), cacheAfter=encode(cache))
        if canonical(result["inputsAfter"]) != before_prompt or canonical(result["cacheAfter"]) != before_cache:
            raise AssertionError("Source input/cache changed")
        if observe:
            result["observation"] = {
                "method": "sys.setprofile real StringFormat.execute call and build_nested_inputs return code objects",
                "bodyCalls": body_calls, "nestedInputReturns": nested_returns}
        return result


async def run_all(names, old, compiled, protocol):
    records = []
    for case in protocol["cases"]:
        before = await run_case(names, old, compiled, case, False)
        observed = await run_case(names, old, compiled, case, True)
        after = await run_case(names, old, compiled, case, False)
        neutral = {k: v for k, v in observed.items() if k != "observation"}
        hashes = [digest(canonical(value)) for value in (before, neutral, after)]
        if len(set(hashes)) != 1:
            raise AssertionError("Observer or repeat changed source result: " + case["id"])
        observed["repeats"] = {"sequence": ["off", "on", "off"], "sha256": hashes, "bitIdentical": True}
        records.append(observed)
    return records


def public_evidence(admitted, execute):
    result = {}
    for name in PUBLIC_FILES:
        relative = "labs/string-format-source/" + name
        raw = (BASE / name).read_bytes()
        result[relative] = {"bytes": len(raw), "sha256": digest(raw)}
        if execute and raw != git(ROOT, "show", admitted + ":" + relative):
            raise ValueError("Publish the exact laboratory before execution: " + relative)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    names, old = load_helpers(protocol)
    compiled, sources = old.load_source(args.source, protocol)  # AST compile, never exec.
    admitted = git(ROOT, "rev-parse", "HEAD").decode().strip()
    public = public_evidence(admitted, args.execute)
    evidence = {"schema": 1, "protocolId": protocol["id"], "protocolSha256": PROTOCOL_SHA256,
                "backendCommit": protocol["backendCommit"], "sourceFiles": sources,
                "laboratoryFiles": public, "helperFiles": protocol["helperFiles"],
                "cases": len(protocol["cases"]), "executed": False, "profile": protocol["profile"]}
    if not args.execute:
        if args.output is not None:
            parser.error("--output is only valid with --execute")
        print(json.dumps(evidence, indent=2))
        return
    if platform.python_version() != protocol["python"] or sys.flags.optimize:
        raise ValueError("Execution requires prospective Python version and enabled assertions")
    destination = old.output_path(args.output, args.source)
    destination.mkdir(parents=True, exist_ok=False)
    results = asyncio.run(run_all(names, old, compiled, protocol))
    if git(ROOT, "rev-parse", "HEAD").decode().strip() != admitted:
        raise ValueError("Collector HEAD changed during execution")
    if public_evidence(admitted, True) != public:
        raise ValueError("Published files changed during execution")
    verify_helpers(protocol)
    _, after_sources = old.load_source(args.source, protocol)
    if after_sources != sources:
        raise ValueError("Source identity changed during execution")
    evidence.update(executed=True, collectorCommit=admitted, python=platform.python_version(),
                    platform=platform.system(), sourceNodeExecution=True, referenceOutputs=results,
                    limitations=[protocol["scope"], protocol["comparison"], protocol["fixtureEncoding"],
                                 "No C# execution/expected output, HTTP validation or native model operation.",
                                 "Unsupported-profile source outputs are not successful port expectations."])
    raw = json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with (destination / "reference.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"cases": len(results), "bytes": len(raw), "sha256": digest(raw)}))


if __name__ == "__main__":
    main()
