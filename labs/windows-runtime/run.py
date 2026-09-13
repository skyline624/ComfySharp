"""Same-host CPU controls. Preserve historical test failures; never replace fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = ("ATEN_CPU_CAPABILITY", "MKL_ENABLE_INSTRUCTIONS", "MKL_CBWR",
            "ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
PROFILES = {
    "auto": {},
    "aten-avx2": {"ATEN_CPU_CAPABILITY": "avx2"},
    "math-avx2": {"ATEN_CPU_CAPABILITY": "avx2", "MKL_ENABLE_INSTRUCTIONS": "AVX2",
                  "ONEDNN_MAX_CPU_ISA": "AVX2", "DNNL_MAX_CPU_ISA": "AVX2"},
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(command, output, env, permitted=(0,)):
    start = time.perf_counter()
    with output.open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=600)
    if result.returncode not in permitted:
        raise RuntimeError(f"Process failed with {result.returncode}; see {output.name}")
    return {"exitCode": result.returncode, "seconds": time.perf_counter() - start}


def verdict(directory):
    files = list(directory.glob("*.trx"))
    if len(files) != 1:
        raise RuntimeError("Expected exactly one TRX per test process")
    root = ET.parse(files[0]).getroot()
    ns = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
    counters = root.find("t:ResultSummary/t:Counters", ns)
    results = root.findall("t:Results/t:UnitTestResult", ns)
    if counters is None or int(counters.attrib["total"]) != len(results):
        raise RuntimeError("Incomplete test evidence")
    return {"total": len(results), "passed": int(counters.attrib["passed"]),
            "failed": int(counters.attrib["failed"]),
            "notExecuted": int(counters.attrib["notExecuted"]),
            "failures": [r.attrib["testName"] for r in results if r.attrib["outcome"] == "Failed"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise RuntimeError("This experiment targets Windows x64")
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # No native import in this parent: every mode gets fresh source/.NET processes.
    base = os.environ.copy()
    for name in list(base):
        if name.startswith("COMFYSHARP_") or name in CONTROLS:
            base.pop(name)
    summary = {"scope": "Same-host reduced SD training and historical ordinary CPU tests; no pretrained model qualification",
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "runId": os.environ.get("GITHUB_RUN_ID"),
               "collectorSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "profiles": {}}
    run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer | ConvertTo-Json"],
        output / "cpu.json", base)
    for profile, controls in PROFILES.items():
        print(profile + ": source and native controls", flush=True)
        target = output / profile
        target.mkdir()
        env = dict(base, **controls)
        row = {"requestedControls": controls}
        summary["profiles"][profile] = row
        row["source"] = run([sys.executable, "-I", "-B", "labs/training-trace/source.py", "--source", str(source),
                             "--output", str(target / "source")], target / "source.log", env)
        traced = dict(env, COMFYSHARP_TRAINING_TRACE_DIR=str(target / "traces"),
                      COMFYSHARP_TRAINING_TRACE_COMPLETE="1", COMFYSHARP_TRAINING_INTEROP_ONE="1",
                      COMFYSHARP_TRAINING_NATIVE_IDENTITY="1")
        command = ["dotnet", "test", "tests/ComfySharp.Inference.Tests", "--no-build", "--no-restore", "-c", "Release",
                   "-p:NativeRuntime=win-x64", "--logger", "trx"]
        results = target / "training-tests"
        row["training"] = run(command + ["--filter", "FullyQualifiedName~SdAdapterInitializationTests", "--results-directory", str(results)],
                              target / "training.log", traced, (0, 1))
        row["training"].update(verdict(results))
        if row["training"]["total"] != 3:
            raise RuntimeError("The frozen training experiment requires two source cases and one lifetime test")
        # Complete observations remain available even if original assertions failed.
        run([sys.executable, "-I", "-S", "-B", "labs/training-trace/compare.py", "--source", str(target / "source"),
             "--actual", str(target / "traces"), "--output", str(target / "comparison.json"), "--require-complete"],
            target / "comparison.log", base)
        cases = read(target / "comparison.json")["cases"]
        row["sameHostComparisons"] = [{"case": c["case"], "captures": len(c["comparisons"]),
            "exactCaptures": sum(v["exact"] for v in c["comparisons"]),
            "baseWeightDifferences": c["baseWeightDifferences"],
            "outsideOriginalTolerance": [v["name"] for v in c["comparisons"] if v["outsideOriginalTolerance"]]} for c in cases]
        row["sameHostFailed"] = any(c["baseWeightDifferences"] or c["outsideOriginalTolerance"] for c in row["sameHostComparisons"])
        row["native"] = []
        for case in range(2):
            actual = read(target / "traces" / f"case-{case}.json")
            expected = read(target / "source" / f"case-{case}.json")
            if actual["target"] != "win-x64" or expected["target"] != "win-x64":
                raise RuntimeError("Source and product must both target Windows x64")
            if not actual["completed"] or actual["threads"] != 1 or actual["interopThreads"] != 1:
                raise RuntimeError("Incomplete training/thread control")
            native = actual["native"]
            if not native or not native["cpuCapability"] or not native["buildConfiguration"]:
                raise RuntimeError("Effective native identity missing")
            if any("python" in lib["name"].lower() for lib in native["libraries"]):
                raise RuntimeError("Python library loaded in .NET")
            row["native"].append({"case": case, "actual": native,
                "sourceCpuCapability": expected["cpuCapability"],
                "sameCpuCapability": native["cpuCapability"] == expected["cpuCapability"],
                "sameBuildConfiguration": native["buildConfiguration"] == read(target / "source" / "manifest.json")["torchBuild"]})
        results = target / "ordinary-tests"
        row["ordinary"] = run(command + ["--filter", "FullyQualifiedName!~ClipStockReferenceTests", "--results-directory", str(results)],
                              target / "ordinary.log", env, (0, 1))
        row["ordinary"].update(verdict(results))
        if profile != "auto" and row["ordinary"]["total"] != summary["profiles"]["auto"]["ordinary"]["total"]:
            raise RuntimeError("CPU control profiles executed different test inventories")
        write(target / "result.json", row)
        print(profile + ": " + json.dumps({"training": row["training"]["failed"], "ordinary": row["ordinary"]["failed"],
                                            "sameHost": row["sameHostComparisons"]}), flush=True)
    write(output / "result.json", summary)
    # A diagnostic run never converts a recorded test failure into a successful gate.
    return 1 if any(r["sameHostFailed"] or any(r[k]["exitCode"] or r[k]["notExecuted"]
        for k in ("training", "ordinary")) for r in summary["profiles"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
