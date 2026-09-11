"""Isolated Linux runtime experiment; never imports a model or updates fixtures.

Run only with the separate pinned CPU laboratory interpreter. Source children use
the unchanged source runner; product children retain the committed expectations.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time


MODES = ("auto", "avx2", "default")
SUITES = {"sampling": "SdSamplingReferenceTests", "unet": "SdUnetReferenceTests",
          "vae": "ClassicalVaeReferenceTests", "guidance": "SdDenoiserReferenceTests"}
ENVIRONMENT = ("ATEN_CPU_CAPABILITY", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "MKL_CBWR",
               "ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def invariant_build(build):
    # __config__.show includes the selected dispatch class. Preserve raw manifests
    # but exclude only this measured variable when comparing build identity.
    return "\n".join(line for line in build.splitlines()
                     if not line.strip().removeprefix("- ").startswith("CPU capability usage:"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(value, prefix=""):
    if isinstance(value, dict):
        if {"shape", "dtype", "values", "sha256"}.issubset(value):
            yield prefix, value
        else:
            for key, child in value.items():
                yield from records(child, prefix + "/" + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from records(child, prefix + "/" + str(index))


def numbers(record):
    values = record["values"]
    require(math.prod(record["shape"]) == len(values), "Tensor shape/value count mismatch.")
    code = {"float32": "f", "int64": "q"}.get(record["dtype"])
    require(code is not None, "Unexpected diagnostic tensor dtype.")
    converted = [float(value) if isinstance(value, str) else value for value in values]
    packed = struct.pack("<" + code * len(converted), *converted)
    require(hashlib.sha256(packed).hexdigest() == record["sha256"], "Tensor payload hash mismatch.")
    # JSON spellings can differ while representing the same F32 payload (for
    # example .NET's shortest round-trip form versus Python's wider decimal).
    return struct.unpack("<" + code * len(converted), packed)


def compare(actual, expected):
    require(actual["shape"] == expected["shape"] and actual["dtype"] == expected["dtype"],
            "Compared tensor contracts differ.")
    left, right = numbers(actual), numbers(expected)
    bad, maximum, first = 0, 0.0, None
    for index, (a, e) in enumerate(zip(left, right)):
        if math.isfinite(a) and math.isfinite(e):
            delta = abs(a - e)
            maximum = max(maximum, delta)
            match = a == e if expected["dtype"] == "int64" else delta <= 3e-5 + 3e-5 * abs(e)
        else:
            match = (math.isnan(a) and math.isnan(e)) or a == e
        if not match:
            bad += 1
            if first is None:
                first = index
    return {"elements": len(left), "maximumFiniteAbsoluteError": maximum, "outsideBound": bad,
            "firstViolation": first, "byteIdentical": actual["sha256"] == expected["sha256"]}


def input_identity(component, document):
    if component == "unet":
        return {"models": document["models"], "cases": [
            {key: case[key] for key in ("id", "model", "latent", "timesteps", "context")}
            for case in document["cases"]]}
    if component == "vae":
        return {"config": document["config"], "sourceConfig": document["sourceConfig"],
                "parameters": document["parameters"], "cases": [
                    {key: case[key] for key in ("id", "kind", "input")} for case in document["cases"]]}
    if component == "guidance":
        return {"config": document["config"], "sourceConfig": document["sourceConfig"],
                "parameters": document["parameters"], "cases": [
                    {key: case[key] for key in ("id", "scale", "latent", "sigma", "positive", "negative",
                                               "concatEligible", "commonLength")}
                    for case in document["cases"]]}
    return {"schedule": {key: document["schedule"][key] for key in
                         ("timestepInputs", "sigmaInputs", "percentInputs")},
            "cases": [{key: case[key] for key in ("id", "predictionKind", "sigma", "latent", "prediction")}
                      for case in document["cases"]],
            "guidance": {key: document["guidance"][key] for key in ("conditional", "unconditional")},
            "latentInput": document["latentFormats"]["input"]}


def source_documents(directory, expected_scripts):
    manifest = load(directory / "manifest.json")
    require(manifest["scripts"] == expected_scripts, "Source scripts changed.")
    require(manifest["profile"] == "sd-components-native210-cpu-f32-v1" and manifest["target"] == "linux-x64"
            and manifest["backendCommit"] == "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a", "Source profile/target differs.")
    require(manifest["runtime"]["threads"] == manifest["runtime"]["interopThreads"] == 1,
            "Source thread configuration differs.")
    documents = {}
    for component in manifest["components"]:
        path = directory / component["file"]
        require(path.name == component["file"] and path.parent == directory, "Unexpected component path.")
        require(digest(path) == component["sha256"], "Source component hash mismatch.")
        documents[component["name"]] = load(path)
        for _, record in records(documents[component["name"]]):
            numbers(record)
    require(set(documents) == set(SUITES), "Source component set differs.")
    return manifest, documents


def product_comparison(product, documents, *, offset=None, include_guidance=True):
    result = {"unet": {}, "guidance": {}, "missing": []}
    require(len(documents["unet"]["cases"]) == 8, "Expected eight source U-Net cases.")
    for case in documents["unet"]["cases"]:
        path = product / "unet" / "traces" / (case["id"].replace("/", "--") + ".json")
        if not path.exists():
            result["missing"].append("unet/" + case["id"])
            continue
        trace = load(path)
        require(trace["synthetic"] and trace["id"] == case["id"], "Wrong U-Net trace identity.")
        allocation = trace["latentInput"]
        shape = case["latent"]["shape"]
        stride = [math.prod(shape[index + 1:]) for index in range(len(shape))]
        require(allocation["allocation"] == ("fixtureTensor" if offset is None else "nativeOffset")
                and allocation["offsetElements"] == offset, "Requested latent allocation control was not applied.")
        require(allocation["shape"] == shape and allocation["stride"] == stride, "Latent allocation changed shape/stride.")
        require(allocation["fixtureSha256"] == allocation["sha256"] == case["latent"]["sha256"], "Offset changed latent bytes.")
        require(isinstance(allocation["aligned64"], bool), "Latent alignment evidence is missing.")
        if offset is not None:
            require(allocation["aligned64"] == (offset == 0), "Latent offset/alignment control differs.")
        values = trace["tensors"]
        result["unet"][case["id"]] = {"output": compare(values["output"], case["output"]),
            "boundaries": {name: compare(values[name], value) for name, value in case["intermediates"].items()}}
        result["unet"][case["id"]]["latentInput"] = allocation
    if not include_guidance:
        return result
    require(sum(len(case["outputs"]) for case in documents["guidance"]["cases"]) == 6,
            "Expected six source guidance outputs.")
    for case in documents["guidance"]["cases"]:
        for policy, expected in case["outputs"].items():
            path = product / "guidance" / "traces" / (case["id"] + "--" + policy + ".json")
            if not path.exists():
                result["missing"].append("guidance/" + case["id"] + "/" + policy)
                continue
            trace = load(path)
            require(trace["synthetic"] and trace["id"] == case["id"] + "/" + policy, "Wrong guidance trace identity.")
            require(trace["parameters"] == documents["guidance"]["parameters"], "Guidance parameter identities differ.")
            require(trace["inputHashes"] == {key: case[key]["sha256"] for key in ("latent", "sigma", "positive", "negative")},
                    "Guidance input bytes differ.")
            result["guidance"][case["id"] + "/" + policy] = compare(trace["tensors"]["output"], expected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "This diagnostic requires Linux x64.")
    require(sys.version_info[:3] == (3, 12, 10), "Use the pinned laboratory interpreter.")
    repo, source = args.repository.resolve(strict=True), args.source_directory.resolve(strict=True)
    require(args.output.is_absolute(), "Output must be absolute.")
    output = args.output.resolve()
    require(not output.exists(), "Diagnostic output must be absent.")
    for protected in (repo, source):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output), "Output overlaps protected input.")
    fixture_directory = repo / "tests/ComfySharp.Inference.Tests/Fixtures"
    fixture_hashes = lambda: {str(p.relative_to(fixture_directory)): digest(p) for p in fixture_directory.rglob("*") if p.is_file()}
    before = fixture_hashes()
    accepted = load(fixture_directory / "sd-components.linux-x64.manifest.json")
    script_hashes = lambda: {p.name: digest(p) for p in (repo / "labs/sd-source").glob("*.py")}
    require(script_hashes() == accepted["scripts"], "Checked-in source scripts differ from their accepted manifest.")
    output.mkdir(parents=True)
    cpu = {}
    for line in Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0].splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("vendor_id", "cpu family", "model", "model name", "stepping", "flags"):
            cpu[key.strip()] = value.strip()
    native_directory = repo / "tests/ComfySharp.Inference.Tests/bin/native/linux-x64/cpu/Release/net10.0"
    binaries = [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(native_directory.rglob("*.so*")) if p.is_file()]
    require(binaries, "Built native product binaries are missing.")
    write(output / "host.json", {"cpu": cpu, "kernel": platform.release(),
          "runnerImage": {key: os.environ.get(key) for key in ("ImageOS", "ImageVersion")},
          "environment": {key: os.environ.get(key) for key in ENVIRONMENT},
          "sourceScripts": accepted["scripts"], "productNativeBinaries": binaries,
          "requirementsSha256": digest(repo / "labs/clip-source/requirements-linux-x64.txt"),
          "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()})
    runs = []
    errors = []

    def execute(label, command, directory, env):
        directory.mkdir(parents=True, exist_ok=True)
        print(label + ": start", flush=True)
        start = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", timeout=300)
            exit_code, text = result.returncode, result.stdout
        except subprocess.TimeoutExpired as error:
            exit_code = 124
            text = error.stdout or ""
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            text += "\nDiagnostic child exceeded 300 seconds.\n"
        for private in (str(repo), str(source), str(output), str(Path(sys.prefix)), str(Path.home())):
            text = text.replace(private, "<local>")
        (directory / "process.log").write_text(text, encoding="utf-8")
        runs.append({"label": label, "exitCode": exit_code, "seconds": time.perf_counter() - start,
                     "environment": {key: env.get(key) for key in (*ENVIRONMENT, "COMFYSHARP_SD_LATENT_OFFSET")}})
        write(output / "runs.json", runs)
        print(label + ": exit " + str(exit_code), flush=True)

    def environment(mode):
        env = os.environ.copy()
        for key in ("ATEN_CPU_CAPABILITY", "COMFYSHARP_SD_LATENT_OFFSET", "COMFYSHARP_SD_UNET_TRACE_DIR", "COMFYSHARP_SD_GUIDANCE_TRACE_DIR"):
            env.pop(key, None)
        if mode != "auto":
            env["ATEN_CPU_CAPABILITY"] = mode
        return env

    for mode in MODES:
        execute("source/" + mode, [sys.executable, "-I", "-B", str(repo / "labs/sd-source/reference.py"),
                "--source-directory", str(source), "--output", str(output / mode / "source"),
                "--components", *SUITES], output / mode / "source-process", environment(mode))

    def product_run(mode, component, directory, offset=None):
        env = environment(mode)
        if offset is not None:
            env["COMFYSHARP_SD_LATENT_OFFSET"] = str(offset)
        env["COMFYSHARP_SD_UNET_TRACE_DIR"] = str(directory / "traces")
        env["COMFYSHARP_SD_GUIDANCE_TRACE_DIR"] = str(directory / "traces")
        execute("product/" + mode + "/" + component + ("/offset-" + str(offset) if offset is not None else ""),
                ["dotnet", "test", "tests/ComfySharp.Inference.Tests", "--no-build", "--no-restore", "-c", "Release",
                 "-p:NativeRuntime=linux-x64", "--filter", "FullyQualifiedName~" + SUITES[component],
                 "--logger", "trx", "--results-directory", str(directory / "results")], directory, env)

    for mode in MODES:
        for component in SUITES:
            product_run(mode, component, output / mode / "product" / component)
    for offset in range(16):
        product_run("auto", "unet", output / "offsets" / f"offset-{offset:02d}" / "product" / "unet", offset)

    comparison = {"diagnosticOnly": True, "qualification": "none", "absoluteTolerance": 3e-5,
                  "relativeTolerance": 3e-5, "sourceVsSource": {}, "productVsSourceSameRequestedMode": {},
                  "productVsSourceAllModes": {},
                  "offsetProductVsAutoSource": {}}
    sources = {}
    for mode in MODES:
        try:
            sources[mode] = source_documents(output / mode / "source", accepted["scripts"])
            capability = sources[mode][0]["runtime"]["cpuCapability"]
            if mode == "avx2":
                require(capability == "AVX2", "AVX2 request was not observed.")
            elif mode == "default":
                require(capability in ("DEFAULT", "NO AVX"), "DEFAULT request was not observed.")
            report = product_comparison(output / mode / "product", sources[mode][1])
            comparison["productVsSourceSameRequestedMode"][mode] = report
            if report["missing"]:
                errors.append(mode + ": required product traces missing: " + ", ".join(report["missing"]))
        except Exception as error:
            errors.append(mode + ": " + str(error))
    for product_mode in MODES:
        for source_mode, source_data in sources.items():
            try:
                comparison["productVsSourceAllModes"][product_mode + "-product_vs_" + source_mode + "-source"] = product_comparison(
                    output / product_mode / "product", source_data[1])
            except Exception as error:
                errors.append(product_mode + "/" + source_mode + ": " + str(error))
    for first, second in itertools.combinations(sources, 2):
        try:
            a, b = sources[first], sources[second]
            for field in ("python", "torch", "torchGit", "threads", "interopThreads", "nativeLibraries"):
                require(a[0]["runtime"][field] == b[0]["runtime"][field], "Non-dispatch source runtime differs: " + field)
            require(invariant_build(a[0]["runtime"]["build"]) == invariant_build(b[0]["runtime"]["build"]),
                    "Non-dispatch build configuration differs.")
            pair = {}
            for component in SUITES:
                require(a[1][component]["sources"] == b[1][component]["sources"], "Source AST identities differ.")
                require(input_identity(component, a[1][component]) == input_identity(component, b[1][component]),
                        "Source parameter/config/input identities differ: " + component)
                left, right = dict(records(a[1][component])), dict(records(b[1][component]))
                require(left.keys() == right.keys(), "Source tensor keys differ.")
                pair[component] = {key: compare(right[key], left[key]) for key in left}
            comparison["sourceVsSource"][first + "-expected_vs_" + second] = pair
        except Exception as error:
            errors.append(first + "/" + second + ": " + str(error))
    if "auto" in sources:
        for offset in range(16):
            try:
                report = product_comparison(output / "offsets" / f"offset-{offset:02d}" / "product", sources["auto"][1],
                                            offset=offset, include_guidance=False)
                report.pop("guidance")
                comparison["offsetProductVsAutoSource"][str(offset)] = report
                if report["missing"]:
                    errors.append("offset-" + str(offset) + ": required U-Net traces missing: " + ", ".join(report["missing"]))
            except Exception as error:
                errors.append("offset-" + str(offset) + ": " + str(error))
    if fixture_hashes() != before:
        errors.append("Accepted fixtures changed during the diagnostic.")
    if script_hashes() != accepted["scripts"]:
        errors.append("Source scripts changed during the diagnostic.")
    write(output / "comparisons.json", comparison)
    write(output / "status.json", {"runs": runs, "integrityErrors": errors,
          "acceptedTestsFailed": any(run["exitCode"] != 0 for run in runs),
          "qualification": "none; diagnostic closeness never replaces committed expectations"})
    return 1 if errors or any(run["exitCode"] != 0 for run in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
