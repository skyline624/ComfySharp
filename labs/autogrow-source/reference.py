"""Prospective AST-only laboratory. Preflight never executes source declarations."""
import abc
import argparse
import ast
import asyncio
import collections
import collections.abc
import contextlib
import contextvars
import copy
import dataclasses
import enum
import hashlib
import inspect
import json
import pathlib
import platform
import subprocess
import sys
import types
import typing

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent.parent
PROTOCOL_SHA256 = "772b018f80b35dd7e1d4f4655ab6d58ccbc62c34fcc915f5a93e9cb8ad1d164b"
PUBLIC_FILES = ("protocol.json", "reference.py", "README.md")


def digest(data):
    return hashlib.sha256(data).hexdigest()


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
    protocol = json.loads(raw, object_pairs_hook=unique_object)
    if protocol["schema"] != 1 or protocol["id"] != "autogrow-prefix-createlist-v1":
        raise ValueError("Unexpected protocol")
    cases = protocol["cases"]
    if len(cases) != 36 or len({case["id"] for case in cases}) != 36:
        raise ValueError("Expected 36 unique prospective cases")
    return protocol


def select(tree, name):
    if name == "@future-annotations":
        candidates = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    elif name == "@setup-dynamic":
        candidates = [node for node in tree.body if isinstance(node, ast.If) and "DYNAMIC_INPUT_LOOKUP" in ast.unparse(node.test)]
    else:
        domain = ast.walk(tree) if name == "execution_block_cb" else tree.body
        candidates = [node for node in domain if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
        if not candidates:
            candidates = [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
                isinstance(target, ast.Name) and target.id == name for target in
                (node.targets if isinstance(node, ast.Assign) else [node.target]))]
    if len(candidates) != 1:
        raise ValueError("Source declaration is missing or ambiguous: " + name)
    return candidates[0]


def load_source(source_root, protocol):
    """Hash, parse and compile exact AST nodes; no exec, constructor or upstream import."""
    compiled, evidence = {}, []
    for record in protocol["sources"]:
        raw = git(source_root, "show", protocol["backendCommit"] + ":" + record["path"])
        if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
            raise ValueError("Frozen source identity mismatch: " + record["path"])
        text = raw.decode("utf-8")
        tree = ast.parse(text, filename=record["path"])
        selected = []
        for declaration in record["declarations"]:
            node = select(tree, declaration["name"])
            if (node.lineno != declaration["line"] or
                    digest(ast.dump(node, include_attributes=False).encode()) != declaration["astSha256"] or
                    digest(ast.get_source_segment(text, node).encode()) != declaration["segmentSha256"]):
                raise ValueError("Frozen AST identity mismatch: " + declaration["name"])
            # No whole-module import or synthetic replacement declaration is admitted.
            if any(isinstance(child, (ast.Import, ast.ImportFrom)) and not
                   (isinstance(child, ast.ImportFrom) and child.module == "__future__")
                   for child in ast.walk(node)):
                raise ValueError("Unexpected import in selected source: " + declaration["name"])
            selected.append(node)
        compiled[record["path"]] = compile(ast.Module(body=selected, type_ignores=[]), record["path"], "exec", dont_inherit=True)
        evidence.append(copy.deepcopy(record))
    return compiled, evidence


def stdlib_namespace():
    # These are real stdlib objects, not source-class replacements. Python 3.12's
    # final/NotRequired are the annotation/decorator counterparts named in protocol.
    return dict(copy=copy, inspect=inspect, asyncio=asyncio, contextvars=contextvars,
                ABC=abc.ABC, abstractmethod=abc.abstractmethod,
                Counter=collections.Counter, Iterable=collections.abc.Iterable,
                asdict=dataclasses.asdict, dataclass=dataclasses.dataclass, field=dataclasses.field,
                Enum=enum.Enum, Any=typing.Any, Callable=typing.Callable, Literal=typing.Literal,
                TypedDict=typing.TypedDict, TypeVar=typing.TypeVar, TYPE_CHECKING=False,
                NotRequired=typing.NotRequired, final=typing.final, Optional=typing.Optional,
                NamedTuple=typing.NamedTuple, Type=typing.Type)


@contextlib.contextmanager
def source_modules(compiled):
    """Execute admitted declarations only, after publication gate in main."""
    installed = []

    def module(label, source, dependencies=None):
        name = "_autogrow_reference_" + label
        if name in sys.modules:
            raise ValueError("Laboratory module name already exists")
        value = types.ModuleType(name)
        value.__dict__.update(stdlib_namespace())
        value.__dict__.update(dependencies or {})
        sys.modules[name] = value
        installed.append(name)
        exec(compiled[source], value.__dict__)
        return value

    try:
        internal = module("internal", "comfy_api/internal/__init__.py")
        graph_utils = module("graph_utils", "comfy_execution/graph_utils.py")
        legacy = module("legacy", "comfy/comfy_types/node_typing.py")
        io = module("io", "comfy_api/latest/_io.py", {
            **{name: getattr(internal, name) for name in (
                "_ComfyNodeInternal", "_NodeOutputInternal", "classproperty", "copy_class",
                "first_real_override", "is_class", "prune_dict", "shallow_clone_class")},
            "ExecutionBlocker": graph_utils.ExecutionBlocker})
        graph = module("graph", "comfy_execution/graph.py", {
            name: getattr(legacy, name) for name in ("ComfyNodeABC", "InputTypeDict", "InputTypeOptions")})
        context = module("context", "comfy_execution/utils.py")
        execution = module("execution", "execution.py", {
            "_ComfyNodeInternal": internal._ComfyNodeInternal, "_NodeOutputInternal": internal._NodeOutputInternal,
            "is_class": internal.is_class, "make_locked_method_func": internal.make_locked_method_func,
            "ExecutionBlocker": graph_utils.ExecutionBlocker, "is_link": graph_utils.is_link,
            "get_input_info": graph.get_input_info, "io": io, "_io": io,
            "CurrentNodeContext": context.CurrentNodeContext})
        toolkit = module("toolkit", "comfy_extras/nodes_toolkit.py", {"io": io})
        yield io, execution, toolkit.CreateList, graph_utils.ExecutionBlocker
    finally:
        for name in reversed(installed):
            del sys.modules[name]


def encode(value, blocker_type):
    if isinstance(value, blocker_type):
        return {"$blocker": value.message}
    if isinstance(value, dict):
        return {key: encode(item, blocker_type) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item, blocker_type) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Unserializable source observation: " + type(value).__name__)


def decode_cache(value, blocker_type):
    if isinstance(value, dict):
        if set(value) == {"$blocker"}:
            message = value["$blocker"]
            if message is not None and not isinstance(message, str):
                raise ValueError("Invalid blocker fixture message")
            return blocker_type(message)
        return {key: decode_cache(item, blocker_type) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_cache(item, blocker_type) for item in value]
    return value


def template_class(io, create_list, case, observations):
    config = case["template"]
    allowed = [str if name == "python-str" else getattr(io, name) for name in config.get("allowedTypes", ["AnyType"])]
    match = io.MatchType.Template("type", allowed_types=allowed)
    options = copy.deepcopy(config.get("prototypeOptions", {}))
    kind = config.get("prototype", "MatchType")
    if kind == "MatchType":
        prototype = io.MatchType.Input("input", template=match, **options)
    elif kind == "Autogrow":
        nested = io.Autogrow.TemplatePrefix(io.MatchType.Input("input", template=match), "nested")
        prototype = io.Autogrow.Input("input", template=nested)
    else:
        prototype = getattr(io, kind).Input("input", **options)
    observations["prototypeBefore"] = encode(prototype.as_dict(), io.ExecutionBlocker)
    template = io.Autogrow.TemplatePrefix(prototype, config.get("prefix", "input"),
                                         min=config.get("min", 1), max=config.get("max", 10))
    observations["prototypeAfter"] = encode(prototype.as_dict(), io.ExecutionBlocker)
    schema = io.Schema(node_id="FixturePrefix", display_name="Fixture Prefix", category="laboratory",
                       is_input_list=True,
                       inputs=[io.Autogrow.Input("inputs", template=template, **config.get("groupOptions", {}))],
                       outputs=[io.MatchType.Output(template=match, is_output_list=True, display_name="list")])

    class FixturePrefix(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return schema

        @classmethod
        def execute(cls, inputs):
            # The generic fixture delegates concatenation to the unchanged source
            # CreateList method; it does not reimplement its calculation.
            return create_list.execute(inputs)

    return FixturePrefix


async def run_case(compiled, case, observe):
    result = {"id": case["id"], "portClassification": case["portClassification"]}
    with source_modules(compiled) as (io, execution, create_list, blocker_type):
        stage = "construct"
        reports, cache_reads, nested_returns, calls = [], [], [], []
        prompt = copy.deepcopy(case["promptInputs"])
        cached = decode_cache(copy.deepcopy(case["cache"]), blocker_type)
        before_prompt = encode(prompt, blocker_type)
        before_cache = encode(cached, blocker_type)
        execution.server = types.SimpleNamespace(client_id="fixture-client")
        execution.server.send_sync = lambda event, payload, client: reports.append({"event": event, "payload": payload, "client": client})
        execution.prompt_id, execution.unique_id = "fixture-prompt", case["id"]
        execution.class_type, execution.executed = "CreateList", ["fixture-upstream"]

        class FixtureCache:
            def get_cache(self, producer, consumer):
                cache_reads.append([producer, consumer])
                return types.SimpleNamespace(outputs=cached[producer]) if producer in cached else None

        try:
            node = create_list if case["kind"] == "create-list" else template_class(io, create_list, case, result)
            stage = "schema"
            node.GET_SCHEMA()  # Same ordering as the pinned loader before module assignment.
            node.RELATIVE_PYTHON_MODULE = "comfy_extras.nodes_toolkit" if node is create_list else "laboratory.autogrow_prefix"
            result["objectInfo"] = encode(node.GET_NODE_INFO_V1(), blocker_type)
            input_types = node.INPUT_TYPES()
            result["inputTypes"] = encode(input_types, blocker_type)
            finalized, hidden, v3 = io.get_finalized_class_inputs(input_types, prompt)
            result["finalized"] = encode(finalized, blocker_type)
            result["finalizedOrder"] = {key: list(value) for key, value in finalized.items()}
            result["v3Data"] = encode(v3, blocker_type)
            result["dynamicPathOrder"] = list(v3.get("dynamic_paths", {}))
            result["hidden"] = encode(hidden, blocker_type)
            stage = "acquisition"
            flat, missing, actual_v3 = execution.get_input_data(prompt, node, case["id"], execution_list=FixtureCache())
            result["flatInputs"] = encode(flat, blocker_type)
            result["flatOrder"] = list(flat)
            result["missingKeys"] = encode(missing, blocker_type)
            result["acquiredV3Data"] = encode(actual_v3, blocker_type)
            result["cacheReads"] = cache_reads
            stage = "mapper"
            create_code = create_list.execute.__func__.__code__
            nested_code = io.build_nested_inputs.__code__

            def profile(frame, event, arg):
                try:
                    if frame.f_code is create_code and event == "call":
                        inputs = frame.f_locals["inputs"]
                        calls.append({"inputs": encode(inputs, blocker_type), "order": list(inputs)})
                    elif frame.f_code is nested_code and event == "return":
                        nested_returns.append({"values": encode(arg, blocker_type),
                                               "rootOrder": list(arg) if isinstance(arg, dict) else None,
                                               "groupOrder": list(arg["inputs"]) if isinstance(arg, dict) and isinstance(arg.get("inputs"), dict) else None})
                except Exception as error:
                    raise RuntimeError("Source observation failed") from error

            if sys.getprofile() is not None:
                raise RuntimeError("An external profiler would confound source observations")
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
            result.update(status="returned", outputs=encode(outputs, blocker_type), ui=encode(ui, blocker_type), hasSubgraph=expanded)
        except (AssertionError, ValueError, TypeError) as error:
            # Only prospectively malformed inputs are admitted as source error records.
            if {"stage": stage, "type": type(error).__name__} not in case["permittedSourceErrors"]:
                raise
            result.update(status="raised", stage=stage, errorType=type(error).__name__, errorMessage=str(error))
        result["reports"] = reports
        result["inputsAfter"] = encode(prompt, blocker_type)
        result["cacheAfter"] = encode(cached, blocker_type)
        if result["inputsAfter"] != before_prompt or result["cacheAfter"] != before_cache:
            raise AssertionError("Source input/cache fixture changed")
        if observe:
            result["observation"] = {"method": "sys.setprofile call/return events; unchanged source code objects", "createListCalls": calls, "nestedInputReturns": nested_returns}
        return result


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


async def run_all(compiled, protocol):
    records = []
    for case in protocol["cases"]:
        before = await run_case(compiled, case, False)
        observed = await run_case(compiled, case, True)
        after = await run_case(compiled, case, False)
        neutral = {key: value for key, value in observed.items() if key != "observation"}
        hashes = [digest(canonical(value)) for value in (before, neutral, after)]
        if len(set(hashes)) != 1:
            raise AssertionError("Observer or repeat changed source result: " + case["id"])
        observed["repeats"] = {"sequence": ["off", "on", "off"], "sha256": hashes, "bitIdentical": True}
        records.append(observed)
    return records


def output_path(value, source):
    if value is None or not value.is_absolute() or value.exists():
        raise ValueError("Execution requires a new absolute output directory")
    path = value.resolve()
    if any(path.is_relative_to(protected.resolve()) for protected in (ROOT, source)):
        raise ValueError("Output must be outside both repositories")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    protocol = load_protocol()
    compiled, source_evidence = load_source(args.source, protocol)
    admitted = git(ROOT, "rev-parse", "HEAD").decode().strip()
    public = {}
    for name in PUBLIC_FILES:
        relative = "labs/autogrow-source/" + name
        raw = (BASE / name).read_bytes()
        public[relative] = {"bytes": len(raw), "sha256": digest(raw)}
        if args.execute and raw != git(ROOT, "show", admitted + ":" + relative):
            raise ValueError("Publish the exact laboratory before source execution: " + relative)
    evidence = {"schema": 1, "protocolId": protocol["id"], "protocolSha256": PROTOCOL_SHA256,
                "backendCommit": protocol["backendCommit"], "sourceFiles": source_evidence,
                "laboratoryFiles": public, "cases": len(protocol["cases"]), "executed": False}
    if not args.execute:
        if args.output is not None:
            parser.error("--output is only valid with --execute")
        print(json.dumps(evidence, indent=2))
        return
    if platform.python_version() != protocol["python"]:
        raise ValueError("Execution requires prospective Python " + protocol["python"])
    if sys.flags.optimize:
        raise ValueError("Source assertions must not be disabled")
    destination = output_path(args.output, args.source)
    destination.mkdir(parents=True, exist_ok=False)
    results = asyncio.run(run_all(compiled, protocol))
    if git(ROOT, "rev-parse", "HEAD").decode().strip() != admitted:
        raise ValueError("Collector HEAD changed during source execution")
    for relative, record in public.items():
        raw = (ROOT / relative).read_bytes()
        if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
            raise ValueError("Published laboratory file changed during source execution")
    # Reverify immutable source reads as well; no working-tree source import occurred.
    _, after_sources = load_source(args.source, protocol)
    if after_sources != source_evidence:
        raise ValueError("Source identity changed during collection")
    evidence.update(executed=True, collectorCommit=admitted, python=platform.python_version(),
                    platform=platform.system(), referenceOutputs=results,
                    limitations=[protocol["scope"], protocol["comparison"], protocol["fixtureEncoding"],
                                 "Fixture cache/server adapters replace infrastructure only; no PromptExecutor or required-input graph validation ran.",
                                 "Unreached type families and DynamicCombo/DynamicSlot behavior are not qualified.",
                                 "Profiling observations must be neutral against both fresh off executions."])
    raw = json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with (destination / "reference.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"cases": len(results), "bytes": len(raw), "sha256": digest(raw)}))


if __name__ == "__main__":
    main()
