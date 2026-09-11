"""Prospective, stdlib-only source laboratory. Never imported by the product."""
import argparse
import ast
import asyncio
import contextlib
import hashlib
import inspect
import json
import pathlib
import platform
import subprocess
import sys
import types

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent.parent
DECLARATIONS = (
    "_async_map_node_over_list", "resolve_map_node_over_list_results",
    "merge_result_data", "get_output_from_returns", "execution_block_cb",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def load_source(source_root, protocol):
    declarations, identities = [], []
    for name, expected in protocol["sourceFiles"].items():
        raw = git(source_root, "show", protocol["backendCommit"] + ":" + name)
        if digest(raw) != expected:
            raise ValueError("Source identity mismatch: " + name)
        source = raw.decode("utf-8")
        tree = ast.parse(source, filename=name)
        wanted = ["ExecutionBlocker"] if name.endswith("graph_utils.py") else DECLARATIONS
        for identifier in wanted:
            matches = [node for node in ast.walk(tree)
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name == identifier]
            if len(matches) != 1:
                raise ValueError("Declaration is not unique: " + identifier)
            node = matches[0]
            declarations.append(node)
            identities.append({
                "source": name, "sourceBytes": len(raw), "sourceSha256": digest(raw),
                "declaration": identifier, "line": node.lineno,
                "astSha256": digest(ast.dump(node, include_attributes=False).encode("utf-8")),
                "segmentSha256": digest(ast.get_source_segment(source, node).encode("utf-8")),
            })
    # Compiling does not execute a declaration or import any upstream module.
    return compile(ast.Module(body=declarations, type_ignores=[]), "<frozen-source-declarations>", "exec"), identities


def validate_protocol(protocol):
    if protocol["schema"] != 1 or protocol["name"] != "execution-blocker-v1-mapper":
        raise ValueError("Unexpected laboratory protocol")
    ids = [case["id"] for case in protocol["cases"]]
    if len(ids) != 24 or len(set(ids)) != len(ids):
        raise ValueError("Expected 24 unique prospective cases")
    bodies = {"echo", "list", "partial", "block", "lazy", "constant", "ui", "blockUi", "throw"}
    for case in protocol["cases"]:
        if case["body"] not in bodies or not isinstance(case["inputs"], dict):
            raise ValueError("Invalid fixture body/inputs")
        if not all(isinstance(value, list) for value in case["inputs"].values()):
            raise ValueError("Inputs must be execution lists")
        flags = case.get("outputIsList", [False])
        if not flags or not all(isinstance(flag, bool) for flag in flags):
            raise ValueError("Invalid output slots")
        if (case["body"] == "partial") != (len(flags) == 2 and case["body"] != "block"):
            raise ValueError("Partial fixture requires two output slots")


class InterruptFixture(Exception):
    pass


class FixtureNodeError(Exception):
    pass


async def run_case(code, case):
    # Named infrastructure adapters. No fixture inherits either V3 sentinel.
    class V3NodeSentinel:
        pass

    class V3OutputSentinel:
        pass

    calls, reports, contexts, pre_calls = [], [], [], []
    interrupt_checks = 0

    def before_node_execution():
        nonlocal interrupt_checks
        interrupt_checks += 1
        if interrupt_checks == case.get("interruptAt"):
            raise InterruptFixture("fixture interruption")

    @contextlib.contextmanager
    def current_context(prompt_id, unique_id, list_index):
        contexts.append({"promptId": prompt_id, "nodeId": unique_id, "index": list_index})
        yield

    server = types.SimpleNamespace(client_id="fixture-client")
    server.send_sync = lambda event, payload, client: reports.append(
        {"type": event, "payload": payload, "client": client})
    environment = {
        "__name__": "frozen_blocker_source",
        "asyncio": asyncio, "inspect": inspect, "is_class": inspect.isclass,
        "_ComfyNodeInternal": V3NodeSentinel, "_NodeOutputInternal": V3OutputSentinel,
        "CurrentNodeContext": current_context,
        "nodes": types.SimpleNamespace(before_node_execution=before_node_execution),
        "server": server, "prompt_id": "fixture-prompt", "unique_id": case["id"],
        "class_type": "LaboratoryV1Node", "executed": ["fixture-upstream"],
    }
    # Execute only the unchanged declarations whose AST/file identities were checked.
    exec(code, environment)
    blocker_type = environment["ExecutionBlocker"]

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"$blocker"}:
                if value["$blocker"] is not None and not isinstance(value["$blocker"], str):
                    raise ValueError("Invalid fixture blocker message")
                return blocker_type(value["$blocker"])
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value

    def encode(value):
        if isinstance(value, blocker_type):
            return {"$blocker": value.message}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if value is None or isinstance(value, (bool, int, str)):
            return value
        raise TypeError("Unexpected fixture result type: " + type(value).__name__)

    def fixture_body(**inputs):
        calls.append(encode(inputs))
        kind = case["body"]
        if kind == "echo":
            return (inputs["x"],)
        if kind == "list":
            return ([inputs["x"]],)
        if kind == "partial":
            return (blocker_type(case.get("message")), inputs["x"])
        if kind == "block":
            return blocker_type(case.get("message"))
        if kind == "lazy":
            return ["when_true" if inputs["select"] else "when_false"]
        if kind == "constant":
            return (1,)
        if kind == "ui":
            return {"result": (inputs["x"],), "ui": {"values": [inputs["x"]]}}
        if kind == "blockUi":
            return {"result": blocker_type(case.get("message")), "ui": {"values": ["producer"]}}
        if kind == "throw":
            raise FixtureNodeError("fixture ordinary error")
        raise ValueError("Unknown fixture body")

    async def async_body(**inputs):
        await asyncio.sleep(0)
        return fixture_body(**inputs)

    node = types.SimpleNamespace(
        INPUT_IS_LIST=case.get("inputIsList", False),
        RETURN_TYPES=("ANY",) * len(case.get("outputIsList", [False])),
        OUTPUT_IS_LIST=tuple(case.get("outputIsList", [False])),
        run=async_body if case.get("async", False) else fixture_body,
    )
    inputs = decode(case["inputs"])
    before = encode(inputs)
    result = {"id": case["id"], "calls": calls, "reports": reports,
              "contexts": contexts, "preExecuteIndices": pre_calls}
    try:
        mapped = await environment["_async_map_node_over_list"](
            "fixture-prompt", case["id"], node, inputs, "run", allow_interrupt=True,
            execution_block_cb=environment["execution_block_cb"] if case.get("callback", True) else None,
            pre_execute_cb=pre_calls.append)
        result["pendingTasksBeforeResolution"] = sum(isinstance(item, asyncio.Task) and not item.done() for item in mapped)
        returned = await environment["resolve_map_node_over_list_results"](mapped)
        result["mappedReturns"] = encode(returned)
        if case["body"] != "lazy":
            output, ui, expanded = environment["get_output_from_returns"](returned, node)
            result.update(outputs=encode(output), ui=encode(ui), hasSubgraph=expanded)
        result["status"] = "returned"
    except (IndexError, InterruptFixture, FixtureNodeError) as error:
        result.update(status="raised", errorType=type(error).__name__)
    result["interruptChecks"] = interrupt_checks
    result["inputsAfter"] = encode(inputs)
    if result["inputsAfter"] != before:
        raise AssertionError("Source laboratory input changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    admitted_commit = git(ROOT, "rev-parse", "HEAD").decode().strip()
    protocol_bytes = (BASE / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_protocol(protocol)
    code, identities = load_source(args.source, protocol)
    public_files = {}
    for name in ("protocol.json", "reference.py", "README.md"):
        relative = "labs/execution-blocker-source/" + name
        content = (BASE / name).read_bytes()
        public_files[relative] = {"bytes": len(content), "sha256": digest(content)}
        if args.execute and content != git(ROOT, "show", admitted_commit + ":" + relative):
            raise ValueError("Publish the exact laboratory before source execution: " + relative)
    evidence = {
        "schema": 1, "backendCommit": protocol["backendCommit"], "protocolSha256": digest(protocol_bytes),
        "sourceDeclarations": identities, "laboratoryFiles": public_files,
        "scope": protocol["scope"], "cases": len(protocol["cases"]), "executed": False,
    }
    if not args.execute:
        if args.output is not None:
            parser.error("--output is only valid with --execute")
        print(json.dumps(evidence, indent=2))
        return
    if args.output is None or not args.output.is_absolute() or args.output.exists():
        parser.error("--execute requires a new absolute output directory")
    output_directory = args.output.resolve()
    if any(output_directory.is_relative_to(protected.resolve()) for protected in (ROOT, args.source)):
        parser.error("The output directory must be outside both repository trees")
    output_directory.mkdir(parents=True, exist_ok=False)
    results = asyncio.run(run_all(code, protocol["cases"]))
    if git(ROOT, "rev-parse", "HEAD").decode().strip() != admitted_commit:
        raise ValueError("Collector revision changed during source execution")
    for relative, identity in public_files.items():
        content = (ROOT / relative).read_bytes()
        if len(content) != identity["bytes"] or digest(content) != identity["sha256"]:
            raise ValueError("Laboratory file changed during source execution: " + relative)
    evidence.update(executed=True, collectorCommit=admitted_commit,
                    python=platform.python_version(), platform=platform.system(),
                    referenceOutputs=results,
                    limitations=["No PromptExecutor, graph expansion, V3 node execution, persistent cache or model runtime executed.",
                                 "Fixture interruption/context/server and V3 sentinel adapters replace infrastructure only.",
                                 "Lazy mapped returns are recorded without claiming whole scheduler restaging.",
                                 "Fixture blocker tags are laboratory encoding, never product JSON literals."])
    output = json.dumps(evidence, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    with (output_directory / "reference.json").open("xb") as stream:
        stream.write(output)
    print(json.dumps({"cases": len(results), "bytes": len(output), "sha256": digest(output)}))


async def run_all(code, cases):
    return [await run_case(code, case) for case in cases]


if __name__ == "__main__":
    main()
