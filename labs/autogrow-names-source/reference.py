"""Prospective TemplateNames laboratory; default preflight executes no source."""
import argparse
import asyncio
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import string
import subprocess
import sys
import types

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
PUBLIC_FILES = ("protocol.json", "reference.py", "README.md")
PROTOCOL_SHA256 = "08b18c44494d41fef84d4055f0ef7baab1d092db0f6706af7a5043a14f8f9a8c"


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


def load_protocol():
    raw = (BASE / "protocol.json").read_bytes()
    if digest(raw) != PROTOCOL_SHA256:
        raise ValueError("Prospective protocol identity mismatch")
    value = json.loads(raw, object_pairs_hook=unique_object)
    if value["schema"] != 1 or value["id"] != "autogrow-template-names-v1":
        raise ValueError("Unexpected protocol")
    if len(value["cases"]) != 36 or len({c["id"] for c in value["cases"]}) != 36:
        raise ValueError("Expected 36 unique prospective cases")
    return value


def verify_helpers(protocol):
    """Verify raw helpers before importing even their inert driver definitions."""
    for record in protocol["helperFiles"]:
        raw = (ROOT / record["path"]).read_bytes()
        if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
            raise ValueError("Published helper identity mismatch: " + record["path"])
        if raw != git(ROOT, "show", protocol["helperCommit"] + ":" + record["path"]):
            raise ValueError("Published helper commit mismatch: " + record["path"])


def load_helper(protocol):
    verify_helpers(protocol)
    spec = importlib.util.spec_from_file_location(
        "_autogrow_names_driver_helper", ROOT / "labs/autogrow-source/reference.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)  # Driver only: no upstream declarations execute.
    return helper


@contextlib.contextmanager
def source_modules(helper, compiled):
    # This function is reached only after the explicit publication gate.
    with helper.source_modules(compiled) as (io, execution, _, blocker_type):
        name = "_autogrow_names_string_source"
        if name in sys.modules:
            raise ValueError("Laboratory module name already exists")
        module = types.ModuleType(name)
        module.__dict__.update(io=io, string=string)
        sys.modules[name] = module
        try:
            exec(compiled["comfy_extras/nodes_string.py"], module.__dict__)
            yield io, execution, module.StringFormat, blocker_type
        finally:
            del sys.modules[name]


def fixture_class(io, case, observations, encode):
    config = case["template"]
    options = copy.deepcopy(config.get("prototypeOptions", {}))
    kind = config.get("prototype", "AnyType")
    if kind == "MatchType":
        match = io.MatchType.Template("type", allowed_types=[io.AnyType])
        prototype = io.MatchType.Input("value", template=match, **options)
    elif kind == "Autogrow":
        nested = io.Autogrow.TemplateNames(io.AnyType.Input("value"), ["nested"])
        prototype = io.Autogrow.Input("value", template=nested)
    else:
        prototype = getattr(io, kind).Input("value", **options)
    observations["prototypeBefore"] = encode(prototype.as_dict())
    caller_names = copy.deepcopy(config["names"])
    observations["callerNamesBefore"] = copy.deepcopy(caller_names)
    template = io.Autogrow.TemplateNames(prototype, caller_names, min=config.get("min", 1))
    observations["prototypeAfter"] = encode(prototype.as_dict())
    observations["templateAtConstruction"] = encode(template.as_dict())
    if "mutateCallerNames" in config:
        caller_names[:] = config["mutateCallerNames"]
    observations["callerNamesAfter"] = copy.deepcopy(caller_names)
    observations["templateAfterCallerMutation"] = encode(template.as_dict())
    observations["cachedInputOrder"] = [item.id for item in template.get_all()]
    group = io.Autogrow.Input("values", template=template, **config.get("groupOptions", {}))
    inputs = []
    for entry in config.get("ordinaryInputs", ["@group"]):
        if entry == "@group":
            inputs.append(group)
        elif entry == "@other":
            other = config["otherGroup"]
            inputs.append(io.Autogrow.Input("other", template=io.Autogrow.TemplateNames(
                io.AnyType.Input("value"), other["names"], min=other.get("min", 1))))
        else:
            inputs.append(io.AnyType.Input(entry))
    schema = io.Schema(node_id="FixtureNames", display_name="Fixture Names", category="laboratory",
                       is_input_list=config.get("isInputList", False), inputs=inputs,
                       outputs=[io.AnyType.Output(display_name="arguments")])

    class FixtureNames(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return schema

        @classmethod
        def execute(cls, **kwargs):
            # Transparent argument carrier, not a replacement source algorithm.
            return io.NodeOutput(kwargs)

    return FixtureNames


async def run_case(helper, compiled, case, observe):
    result = {"id": case["id"], "portClassification": case["portClassification"],
              "bodyOrigin": "source.StringFormat" if case["kind"] == "string-format" else "fixture.argument-carrier"}
    with source_modules(helper, compiled) as (io, execution, formatter, blocker_type):
        encode = lambda value: helper.encode(value, blocker_type)
        stage = "construct"
        reports, cache_reads, nested_returns, calls = [], [], [], []
        prompt = copy.deepcopy(case["promptInputs"])
        cached = helper.decode_cache(copy.deepcopy(case["cache"]), blocker_type)
        before_prompt, before_cache = encode(prompt), encode(cached)
        execution.server = types.SimpleNamespace(client_id="fixture-client")
        execution.server.send_sync = lambda event, payload, client: reports.append(
            {"event": event, "payload": payload, "client": client})
        execution.prompt_id, execution.unique_id = "fixture-prompt", case["id"]
        execution.class_type = "StringFormat" if case["kind"] == "string-format" else "FixtureNames"
        execution.executed = ["fixture-upstream"]

        class FixtureCache:
            def get_cache(self, producer, consumer):
                cache_reads.append([producer, consumer])
                return types.SimpleNamespace(outputs=cached[producer]) if producer in cached else None

        try:
            node = formatter if case["kind"] == "string-format" else fixture_class(io, case, result, encode)
            stage = "schema"
            node.GET_SCHEMA()
            node.RELATIVE_PYTHON_MODULE = ("comfy_extras.nodes_string" if node is formatter else
                                          "laboratory.autogrow_names")
            result["objectInfo"] = encode(node.GET_NODE_INFO_V1())
            input_types = node.INPUT_TYPES()
            result["inputTypes"] = encode(input_types)
            finalized, hidden, v3 = io.get_finalized_class_inputs(input_types, prompt)
            result["finalized"] = encode(finalized)
            result["finalizedOrder"] = {key: list(value) for key, value in finalized.items()}
            result["v3Data"] = encode(v3)
            result["dynamicPathOrder"] = list(v3.get("dynamic_paths", {}))
            result["hidden"] = encode(hidden)
            stage = "acquisition"
            flat, missing, actual_v3 = execution.get_input_data(
                prompt, node, case["id"], execution_list=FixtureCache())
            result.update(flatInputs=encode(flat), flatOrder=list(flat), missingKeys=encode(missing),
                          acquiredV3Data=encode(actual_v3), cacheReads=cache_reads)
            stage = "mapper"
            target_code = node.execute.__func__.__code__
            nested_code = io.build_nested_inputs.__code__

            def ordered(value):
                return {"values": encode(value), "rootOrder": list(value) if isinstance(value, dict) else None,
                        "groupOrders": {k: list(v) for k, v in value.items() if isinstance(v, dict)}
                        if isinstance(value, dict) else None}

            def profile(frame, event, arg):
                try:
                    if frame.f_code is target_code and event == "call":
                        kwargs = ({"values": frame.f_locals["values"], "f_string": frame.f_locals["f_string"]}
                                  if node is formatter else frame.f_locals["kwargs"])
                        calls.append(ordered(kwargs))
                    elif frame.f_code is nested_code and event == "return":
                        nested_returns.append(ordered(arg))
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
        except (AssertionError, ValueError, TypeError, IndexError) as error:
            if {"stage": stage, "type": type(error).__name__} not in case["permittedSourceErrors"]:
                raise
            result.update(status="raised", stage=stage, errorType=type(error).__name__, errorMessage=str(error))
        result.update(reports=reports, inputsAfter=encode(prompt), cacheAfter=encode(cached))
        if result["inputsAfter"] != before_prompt or result["cacheAfter"] != before_cache:
            raise AssertionError("Source input/cache fixture changed")
        if observe:
            result["observation"] = {"method": "sys.setprofile real binder and body code objects",
                                     "bodyCalls": calls, "nestedInputReturns": nested_returns}
        return result


async def run_all(helper, compiled, protocol):
    records = []
    for case in protocol["cases"]:
        before = await run_case(helper, compiled, case, False)
        observed = await run_case(helper, compiled, case, True)
        after = await run_case(helper, compiled, case, False)
        neutral = {key: value for key, value in observed.items() if key != "observation"}
        hashes = [digest(helper.canonical(value)) for value in (before, neutral, after)]
        if len(set(hashes)) != 1:
            raise AssertionError("Observer or repeat changed source result: " + case["id"])
        observed["repeats"] = {"sequence": ["off", "on", "off"], "sha256": hashes, "bitIdentical": True}
        records.append(observed)
    return records


def public_evidence(admitted, execute):
    result = {}
    for name in PUBLIC_FILES:
        relative = "labs/autogrow-names-source/" + name
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
    helper = load_helper(protocol)
    compiled, source_evidence = helper.load_source(args.source, protocol)
    admitted = git(ROOT, "rev-parse", "HEAD").decode().strip()
    public = public_evidence(admitted, args.execute)
    evidence = {"schema": 1, "protocolId": protocol["id"], "protocolSha256": PROTOCOL_SHA256,
                "backendCommit": protocol["backendCommit"], "sourceFiles": source_evidence,
                "laboratoryFiles": public, "helperCommit": protocol["helperCommit"],
                "helperFiles": protocol["helperFiles"], "cases": len(protocol["cases"]), "executed": False}
    if not args.execute:
        if args.output is not None:
            parser.error("--output is only valid with --execute")
        print(json.dumps(evidence, indent=2))
        return
    if platform.python_version() != protocol["python"] or sys.flags.optimize:
        raise ValueError("Execution requires prospective Python version and enabled assertions")
    destination = helper.output_path(args.output, args.source)
    destination.mkdir(parents=True, exist_ok=False)
    results = asyncio.run(run_all(helper, compiled, protocol))
    if git(ROOT, "rev-parse", "HEAD").decode().strip() != admitted:
        raise ValueError("Collector HEAD changed during execution")
    if public_evidence(admitted, True) != public:
        raise ValueError("Published files changed during execution")
    verify_helpers(protocol)
    _, after_sources = helper.load_source(args.source, protocol)
    if after_sources != source_evidence:
        raise ValueError("Source identity changed during execution")
    evidence.update(executed=True, collectorCommit=admitted, python=platform.python_version(),
                    platform=platform.system(), referenceOutputs=results,
                    limitations=[protocol["scope"], protocol["comparison"], protocol["fixtureEncoding"],
                                 protocol["fixtureBody"],
                                 "No PromptExecutor, required-input graph validation, HTTP, C# or formatter port qualification."])
    raw = json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with (destination / "reference.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"cases": len(results), "bytes": len(raw), "sha256": digest(raw)}))


if __name__ == "__main__":
    main()
