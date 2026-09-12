"""Prospective CaseConverter and exhaustive upper/capitalize/title digests. Preflight executes no source."""
import argparse
import asyncio
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import types
import unicodedata

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
PUBLIC_FILES = ("protocol.json", "reference.py", "README.md")
PROTOCOL_SHA256 = "5ccc7810c4c99d427ee1dea167b26bcc638187bb4e0ae0baa2d6cda8291a58de"
ARGUMENTS = {"CaseConverter": ("string", "mode")}
CASING_RECIPES = (
    {"id": "upper_single", "operation": "upper", "prefix": "", "suffix": ""},
    {"id": "capitalize_single", "operation": "capitalize", "prefix": "", "suffix": ""},
    {"id": "capitalize_a_cp_sigma", "operation": "capitalize", "prefix": "A", "suffix": "Σ"},
    {"id": "title_a_cp_sigma", "operation": "title", "prefix": "A", "suffix": "Σ"},
    {"id": "title_a_sigma_cp_a", "operation": "title", "prefix": "AΣ", "suffix": "A"},
)


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


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_protocol():
    raw = (BASE / "protocol.json").read_bytes()
    if digest(raw) != PROTOCOL_SHA256:
        raise ValueError("Prospective protocol identity mismatch")
    value = json.loads(raw, object_pairs_hook=unique_object)
    if value["schema"] != 1 or value["id"] != "case-converter-python312-v1":
        raise ValueError("Unexpected protocol")
    cases = value["cases"]
    if len(cases) != 40 or len({c["id"] for c in cases}) != 40:
        raise ValueError("Expected 40 unique cases")
    if {k: sum(c["portClassification"] == k for c in cases) for k in
            ("comparable", "source-error", "source-only-invalid-mode")} != {
            "comparable": 28, "source-error": 8, "source-only-invalid-mode": 4}:
        raise ValueError("Unexpected prospective partition")
    for case in cases:
        if (case["node"] not in ARGUMENTS or set(case["promptInputs"]) != set(ARGUMENTS[case["node"]])
                or list(case["promptInputs"]) != case["promptInputOrder"]):
            raise ValueError("Unexpected node or argument order")
    lower = value["helperCasing"]
    if len(lower["planes"]) != 17 or len(lower["modes"]) != 5 or lower["repeats"] != 3:
        raise ValueError("Unexpected exhaustive plan")
    if (lower["modes"] != list(CASING_RECIPES) or lower["digestCount"] != 85
            or lower["inputApplicationsPerPass"] != 5560320 or lower["totalValidScalars"] != 1112064):
        raise ValueError("Unexpected casing recipes or totals")
    for plane, record in enumerate(lower["planes"]):
        if record != dict(plane=plane, startInclusive=plane << 16,
                          endInclusive=((plane + 1) << 16) - 1, scalarCount=63488 if plane == 0 else 65536):
            raise ValueError("Unexpected scalar plane")
    if sum(r["scalarCount"] for r in lower["planes"]) != 1112064:
        raise ValueError("Unexpected valid-scalar count")
    return value


def verify_helpers(protocol):
    for record in protocol["helperFiles"]:
        raw = (ROOT / record["path"]).read_bytes()
        if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
            raise ValueError("Published helper identity mismatch: " + record["path"])
        if raw != git(ROOT, "show", record["commit"] + ":" + record["path"]):
            raise ValueError("Published helper commit mismatch: " + record["path"])


def load_helper(protocol):
    verify_helpers(protocol)  # All six origin/helper files checked before importing the driver.
    spec = importlib.util.spec_from_file_location(
        "_case_converter_prefix_driver", ROOT / "labs/autogrow-source/reference.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)  # Inert driver definitions, not source declarations.
    return helper


@contextlib.contextmanager
def source_modules(helper, compiled):
    # Reached only after publication admission. Every source declaration is unchanged.
    with helper.source_modules(compiled) as (io, execution, _, blocker):
        name = "_case_converter_source_nodes"
        if name in sys.modules:
            raise ValueError("Laboratory module collision")
        module = types.ModuleType(name)
        module.__dict__["io"] = io
        sys.modules[name] = module
        try:
            exec(compiled["comfy_extras/nodes_string.py"], module.__dict__)
            yield io, execution, module, blocker
        finally:
            del sys.modules[name]


async def run_case(helper, compiled, case, observe):
    result = {"id": case["id"], "node": case["node"], "portClassification": case["portClassification"],
              "bodyOrigin": "frozen " + case["node"] + ".execute -> CPython builtin casing or identity fallback"}
    with source_modules(helper, compiled) as (io, execution, module, blocker_type):
        node = getattr(module, case["node"])
        encode = lambda value: helper.encode(value, blocker_type)
        prompt = copy.deepcopy(case["promptInputs"])
        cache = helper.decode_cache(copy.deepcopy(case["cache"]), blocker_type)
        before_prompt, before_cache = canonical(encode(prompt)), canonical(encode(cache))
        reports, cache_reads, body_calls, nested_returns = [], [], [], []
        execution.server = types.SimpleNamespace(client_id="fixture-client")
        execution.server.send_sync = lambda event, payload, client: reports.append(
            {"event": event, "payload": payload, "client": client})
        execution.prompt_id, execution.unique_id = "fixture-prompt", case["id"]
        execution.class_type, execution.executed = case["node"], ["fixture-upstream"]

        class FixtureCache:
            def get_cache(self, producer, consumer):
                cache_reads.append([producer, consumer])
                return types.SimpleNamespace(outputs=cache[producer]) if producer in cache else None

        stage = "schema"
        try:
            node.GET_SCHEMA()  # Real loader ordering: initial schema before module attribution.
            node.RELATIVE_PYTHON_MODULE = "comfy_extras.nodes_string"
            result["objectInfo"] = encode(node.GET_NODE_INFO_V1())
            input_types = node.INPUT_TYPES()
            result["inputTypes"] = encode(input_types)
            finalized, hidden, v3 = io.get_finalized_class_inputs(input_types, prompt)
            result.update(finalized=encode(finalized), finalizedOrder={k: list(v) for k, v in finalized.items()},
                          hidden=encode(hidden), v3Data=encode(v3))
            stage = "acquisition"
            flat, missing, actual_v3 = execution.get_input_data(prompt, node, case["id"], execution_list=FixtureCache())
            result.update(flatInputs=encode(flat), flatOrder=list(flat), missingKeys=encode(missing),
                          acquiredV3Data=encode(actual_v3), cacheReads=cache_reads)
            stage = "mapper"
            body_code, binder_code = node.execute.__func__.__code__, io.build_nested_inputs.__code__

            def profile(frame, event, arg):
                try:
                    if frame.f_code is body_code and event == "call":
                        body_calls.append({"arguments": {k: encode(frame.f_locals[k]) for k in ARGUMENTS[case["node"]]},
                                           "argumentOrder": list(ARGUMENTS[case["node"]])})
                    elif frame.f_code is binder_code and event == "return":
                        nested_returns.append({"arguments": encode(arg), "rootOrder": list(arg)})
                except Exception as error:
                    raise RuntimeError("Source observation failed") from error

            if sys.getprofile() is not None:
                raise RuntimeError("An external profiler would confound observation")
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
        except (AssertionError, ValueError, TypeError, IndexError, KeyError, OverflowError, AttributeError) as error:
            if {"stage": stage, "type": type(error).__name__} not in case["permittedSourceErrors"]:
                raise
            result.update(status="raised", stage=stage, errorType=type(error).__name__, errorMessage=str(error))
        if case["portClassification"] == "source-error" and result["status"] != "raised":
            raise AssertionError("Prospectively erroneous case unexpectedly returned")
        result.update(reports=reports, inputsAfter=encode(prompt), cacheAfter=encode(cache))
        if canonical(result["inputsAfter"]) != before_prompt or canonical(result["cacheAfter"]) != before_cache:
            raise AssertionError("Source input/cache mutation")
        if observe:
            result["observation"] = {"method": "sys.setprofile actual body call and actual binder return code objects",
                                     "bodyCalls": body_calls, "nestedInputReturns": nested_returns}
        return result


async def run_all(helper, compiled, protocol):
    records = []
    for case in protocol["cases"]:
        before = await run_case(helper, compiled, case, False)
        observed = await run_case(helper, compiled, case, True)
        after = await run_case(helper, compiled, case, False)
        neutral = {k: v for k, v in observed.items() if k != "observation"}
        hashes = [digest(canonical(v)) for v in (before, neutral, after)]
        if len(set(hashes)) != 1:
            raise AssertionError("Observation/repeat changed the source result")
        observed["repeats"] = {"sequence": ["off", "on", "off"], "sha256": hashes, "bitIdentical": True}
        records.append(observed)
    return records


def framed(cp, text):
    raw = text.encode("utf-8", "strict")
    return struct.pack("<II", cp, len(raw)) + raw


def collect_casing(protocol):
    # Independent builtin-only oracle. Never reads product tables or their extractor.
    if sys.getprofile() is not None:
        raise RuntimeError("Exhaustive builtin collection requires no external profiler")
    records = []
    plan = protocol["helperCasing"]
    for plane in plan["planes"]:
        for mode in plan["modes"]:
            repeats = []
            for _ in range(plan["repeats"]):
                inputs, outputs = hashlib.sha256(), hashlib.sha256()
                count = input_bytes = output_bytes = 0
                for cp in range(plane["startInclusive"], plane["endInclusive"] + 1):
                    if 0xd800 <= cp <= 0xdfff:
                        continue
                    text = mode["prefix"] + chr(cp) + mode["suffix"]
                    if mode["operation"] == "upper":
                        converted = text.upper()
                    elif mode["operation"] == "capitalize":
                        converted = text.capitalize()
                    elif mode["operation"] == "title":
                        converted = text.title()
                    else:
                        raise AssertionError("Unexpected prospective builtin operation")
                    source, lowered = framed(cp, text), framed(cp, converted)
                    inputs.update(source); outputs.update(lowered)
                    count += 1; input_bytes += len(source); output_bytes += len(lowered)
                if count != plane["scalarCount"]:
                    raise AssertionError("Scalar count mismatch")
                repeats.append(dict(count=count, inputFramedBytes=input_bytes, outputFramedBytes=output_bytes,
                                    inputSha256=inputs.hexdigest(), sha256=outputs.hexdigest()))
            if repeats[0] != repeats[1] or repeats[1] != repeats[2]:
                raise AssertionError("Builtin casing repeats differ")
            records.append(dict(plane=plane["plane"], mode=mode["id"], **repeats[0],
                                repeatSha256=[r["sha256"] for r in repeats], repeatsIdentical=True))
    if len(records) != 85:
        raise AssertionError("Expected 85 casing digests")
    return {"id": plan["id"], "framing": plan["framing"], "records": records,
            "totalValidScalars": plan["totalValidScalars"], "inputApplicationsPerPass": 5 * plan["totalValidScalars"],
            "repeats": plan["repeats"], "nativeProductOrTableRead": False}


def public_evidence(admitted, execute):
    result = {}
    for name in PUBLIC_FILES:
        relative = "labs/case-converter-source/" + name
        raw = (BASE / name).read_bytes()
        result[relative] = {"bytes": len(raw), "sha256": digest(raw)}
        if execute and raw != git(ROOT, "show", admitted + ":" + relative):
            raise ValueError("Publish exact laboratory before execution: " + relative)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    helper = load_helper(protocol)
    compiled, sources = helper.load_source(args.source, protocol)  # Compile only, never exec.
    admitted = git(ROOT, "rev-parse", "HEAD").decode().strip()
    public = public_evidence(admitted, args.execute)
    evidence = dict(schema=1, protocolId=protocol["id"], protocolSha256=PROTOCOL_SHA256,
                    backendCommit=protocol["backendCommit"], sourceFiles=sources, helperFiles=protocol["helperFiles"],
                    laboratoryFiles=public, cases=len(protocol["cases"]), casingDigestPlans=85, executed=False,
                    profile=protocol["profile"])
    if not args.execute:
        if args.output is not None:
            parser.error("--output requires --execute")
        print(json.dumps(evidence, indent=2))
        return
    if (platform.python_implementation() != "CPython" or platform.python_version() != protocol["python"]
            or unicodedata.unidata_version != protocol["unicodeVersion"] or sys.flags.optimize):
        raise ValueError("Execution requires CPython 3.12.10, Unicode 15.0.0 and assertions")
    destination = helper.output_path(args.output, args.source)
    executable = Path(sys.executable).resolve()
    executable_record = dict(name=executable.name, bytes=executable.stat().st_size, sha256=digest(executable.read_bytes()))
    destination.mkdir(parents=True, exist_ok=False)
    records = asyncio.run(run_all(helper, compiled, protocol))
    casing = collect_casing(protocol)
    if git(ROOT, "rev-parse", "HEAD").decode().strip() != admitted:
        raise ValueError("Collector HEAD changed")
    if public_evidence(admitted, True) != public:
        raise ValueError("Published files changed")
    verify_helpers(protocol)
    _, after_sources = helper.load_source(args.source, protocol)
    if after_sources != sources:
        raise ValueError("Frozen source identity changed")
    if executable_record != dict(name=executable.name, bytes=executable.stat().st_size, sha256=digest(executable.read_bytes())):
        raise ValueError("Interpreter executable changed")
    evidence.update(executed=True, collectorCommit=admitted, python=platform.python_version(),
                    implementation=platform.python_implementation(), unicodeVersion=unicodedata.unidata_version,
                    platform=platform.system(), executable=executable_record, referenceOutputs=records,
                    helperCasing=casing, limitations=[protocol["scope"], protocol["comparison"], protocol["fixtureEncoding"],
                    "Off runs retain hashes only; middle raw records permit independent hash recomputation.",
                    "Interpreter executable hash does not attest every dynamic Python library.",
                    "85 casing digests do not exhaust every multi-character Unicode context."])
    raw = json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with (destination / "reference.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(cases=len(records), casingDigests=85, bytes=len(raw), sha256=digest(raw))))


if __name__ == "__main__":
    main()
