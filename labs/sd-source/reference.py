"""Opt-in source laboratory. Not part of the product or distributed .NET tests."""
import argparse
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time

# -I deliberately removes the script directory. Admit only this checked-in lab.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COMMIT, PROFILE, require, write_document


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", nargs="+", choices=("sampling", "unet", "vae", "guidance"), default=["sampling"])
    args = parser.parse_args()
    require(sys.version_info[:3] == (3, 12, 10), "Use the pinned Python 3.12.10 laboratory runtime.")
    require(importlib.metadata.version("torch") in ("2.10.0", "2.10.0+cpu"), "Use PyTorch 2.10.0 CPU.")
    source = args.source_directory.resolve(strict=True)
    require(source.is_dir(), "Source snapshot must be a directory.")
    require(args.output.is_absolute(), "Output must be an explicit absolute directory.")
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[2]
    for protected in (repo, source):
        require(not (output.is_relative_to(protected) or protected.is_relative_to(output)),
                "Laboratory output must be separate from the repository and source snapshot.")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Output must be absent or empty. Existing evidence is not overwritten.")

    import torch
    require(torch.version.cuda is None, "CUDA wheels are outside this CPU reference protocol.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = {("Windows", "AMD64"): "win-x64", ("Linux", "x86_64"): "linux-x64",
              ("Darwin", "arm64"): "osx-arm64"}.get((platform.system(), platform.machine()))
    require(target is not None, "Platform is outside the declared V1 CPU targets.")
    output.mkdir(parents=True, exist_ok=True)
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    natives = [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
               for p in sorted(torch_lib.iterdir()) if p.is_file() and
               (p.suffix.lower() in (".dll", ".dylib", ".so") or ".so." in p.name)]
    build = torch.__config__.show()
    # No user home, laboratory path or checkpoint path belongs in public evidence.
    for private in (str(Path.home()), str(Path(sys.prefix)), str(source), str(output)):
        build = build.replace(private, "<local>")
    scripts = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    manifest = {"backendCommit": COMMIT, "profile": PROFILE, "target": target,
                "comparison": {"absoluteTolerance": 3e-5, "relativeTolerance": 3e-5,
                               "discrete": "exact", "nonFinite": "class-and-infinity-sign"},
                "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                            "torchGit": torch.version.git_version, "threads": torch.get_num_threads(),
                            "interopThreads": torch.get_num_interop_threads(), "build": build,
                            "cpuCapability": torch.backends.cpu.get_cpu_capability(), "nativeLibraries": natives},
                "syntheticWeights": True, "pretrainedWeightsUsed": False,
                "parameterRecipe": "sha256-name-lcg-high16-power2-v1", "scripts": scripts, "components": []}
    for component in dict.fromkeys(args.components):
        evidence = []
        start = time.perf_counter()
        with torch.no_grad():
            document = importlib.import_module(component).generate(source, evidence)
        document = {"backendCommit": COMMIT, "profile": PROFILE, "target": target,
                    "sources": evidence, **document}
        path = output / (component + ".json")
        write_document(path, document)
        manifest["components"].append({"name": component, "file": path.name, "sha256": digest(path),
                                       "bytes": path.stat().st_size, "elapsedSeconds": time.perf_counter() - start})
        print(component + ": completed from verified frozen source", flush=True)
    write_document(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
