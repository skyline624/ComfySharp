"""Explicit synthetic stock U-Net source collection; no product or golden inputs."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LAB = Path(__file__).resolve().parent
PROFILE = "sd-unet-stock-small-native210-cpu-f32-v1"
COLLECTOR_VERSION = "stock-source-v2-canonical-lock-content"
CASES_SHA256 = "ed7e18588e3ed6dda90714f0ad41664c33d601bdb68bbac31c8eea5ea9ee3e59"
RECIPE = "sha256-name-lcg-high16-power2-v1"
CHUNK = 262144
MASK = 0xffffffff
HELPERS = {
    "common.py": "970ee60f97d8c535a7076eba91c2cd38baecf88cf023f68ac8965fbf8a6853c3",
    "unet.py": "57813ce615d443601511e3657d56039bf32796e2b7904a7d0ff2d64be55c520f",
}
LOCKS = {
    "win-x64": "53becb18e5c1ea63de4ee8f6eacdd482bcd992827be25439a0a84a89cbc099d5",
    "linux-x64": "84df56cd98339e8dfec9b6f765312758706476ed1328d5f95f6a06433b9bb719",
    "osx-arm64": "329ab59df4b1cd1b5dd16eaa2a84d3f0e8e4c807f6bf984246c70290e406dec4",
}
SOURCES = {
    "comfy/ldm/modules/diffusionmodules/util.py": "fb58652a35521fc23bdcb75d91adace8e4cc79e2d5b13af1617a38d0c0f7142e",
    "comfy/ldm/modules/attention.py": "9cafaafaf93ff53e8cbefb5e4a204014019985df2da8c1996bf40f1235fc2960",
    "comfy/ldm/modules/diffusionmodules/openaimodel.py": "9d27fb036cab8a262ef3d866a643f7fdc40994022616f1b8be14b7d919f57f96",
}
_NATIVE_LIBRARIES = None


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def lock_attestation(raw, expected_canonical):
    """Only CRLF pairs become LF. No whitespace, encoding or content normalization."""
    canonical = raw.replace(b"\r\n", b"\n")
    canonical_sha = hashlib.sha256(canonical).hexdigest()
    require(canonical_sha == expected_canonical, "Canonical dependency lock content changed.")
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n") - crlf
    bare_cr = raw.count(b"\r") - crlf
    form = "bare_cr_present" if bare_cr else "mixed_lf_crlf" if crlf and lf else "crlf" if crlf else "lf" if lf else "none"
    return {"rawSha256": hashlib.sha256(raw).hexdigest(), "rawBytes": len(raw),
        "lineEndings": {"form": form, "crlfCount": crlf, "lfCount": lf, "bareCrCount": bare_cr},
        "canonicalSha256": canonical_sha, "canonicalBytes": len(canonical),
        "normalizationRule": "replace_CRLF_with_LF_only_no_other_byte_changes"}


class Provenance(dict):
    def __init__(self):
        super().__init__()
        self.raw_sizes = {}
        self.lock_before = None
        self.lock_after = None


def capture_provenance(source, target):
    paths = {"reference.py": (Path(__file__), None), "cases-v1.json": (LAB / "cases-v1.json", CASES_SHA256)}
    paths.update({"helpers/" + name: (ROOT / "labs/sd-source" / name, sha) for name, sha in HELPERS.items()})
    paths["requirements/" + target] = (ROOT / "labs/clip-source" / ("requirements-"+target+".txt"), LOCKS[target])
    for relative, expected in SOURCES.items():
        path = (source / relative).resolve(strict=True)
        require(path.is_relative_to(source), "Source path escapes its snapshot.")
        paths["source/" + relative] = (path, expected)
    captured = Provenance()
    for label, (path, expected) in paths.items():
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if label.startswith("requirements/"):
            captured.lock_before = lock_attestation(raw, expected)
        else:
            require(expected is None or actual == expected, "Provenance hash mismatch: " + label)
        captured[label] = (path, actual)
        captured.raw_sizes[label] = len(raw)
    return captured


def verify_provenance(captured):
    lock_after = None
    for label, (path, expected) in captured.items():
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, "Raw provenance changed during collection: " + label)
        if isinstance(captured, Provenance):
            require(len(raw) == captured.raw_sizes[label], "Raw provenance length changed during collection: " + label)
            if label.startswith("requirements/"):
                lock_after = lock_attestation(raw, captured.lock_before["canonicalSha256"])
                require(lock_after == captured.lock_before, "Raw lock attestation changed during collection.")
    if isinstance(captured, Provenance):
        require(lock_after is not None, "Dependency lock final attestation is missing.")
        captured.lock_after = lock_after
    return {label: sha for label, (_, sha) in captured.items()}


def publish_manifest(output, document, captured):
    # No final manifest exists until every executed input file has been rehashed.
    try:
        document["provenanceVerifiedAfterExecution"] = verify_provenance(captured)
        if isinstance(captured, Provenance):
            document["dependencyLockAttestation"] = {"before": captured.lock_before, "after": captured.lock_after}
        write_json(output / "manifest.json", document)
    except Exception as error:
        try:
            write_json(output / "incomplete.json", {"status": "collection_incomplete", "errorType": type(error).__name__,
                "reason": "provenance_verification_or_final_publication_failed", "modelCompatibility": "not_assessed"})
        except Exception:
            pass
        raise
    require(document["outputHashesRepeat"], "Source forward hashes differ between repetitions.")


def write_json(path, value):
    # Exclusive creation preserves prior evidence. Failed collection may leave a tmp,
    # but never a success manifest or overwrite an earlier completed document.
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not path.exists(), "Evidence output already exists.")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # Hard-link creation is atomic and fails if the final file exists, on all targets.
    os.link(temporary, path)
    temporary.unlink()


def verified_helpers():
    directory = ROOT / "labs/sd-source"
    for name, expected in HELPERS.items():
        require(digest(directory / name) == expected, "Frozen source helper changed: " + name)
    require("common" not in sys.modules, "Unexpected preloaded source helper.")
    modules = {}
    for name in ("common", "unet"):
        spec = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["common"], modules["unet"]


def descriptor(name, shape, parameter):
    require(isinstance(name, str) and name.strip(), "A recipe name is required.")
    require(shape and all(isinstance(x, int) and x > 0 for x in shape), "Invalid recipe shape.")
    elements = math.prod(shape)
    require(elements <= 2**63 - 1, "Recipe element count overflows Int64.")
    seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
    exponent, offset = -15, 0
    if parameter and len(shape) == 1:
        offset = int(name.endswith(".weight"))
        exponent = -18 if offset else -20
    elif parameter:
        exponent -= ((math.prod(shape[1:]) - 1).bit_length() + 1) // 2
    return {"seed": seed, "offset": offset, "exponent": exponent, "elements": elements}


class Filler:
    """Exactly two 1 MiB NumPy scratch buffers; writes native destination in place."""
    def __init__(self, np):
        self.np = np
        self.indices = np.arange(CHUNK, dtype=np.uint32)
        self.work = np.empty(CHUNK, dtype=np.uint32)
        require(self.indices.nbytes + self.work.nbytes == 2 * 1024 * 1024, "Scratch size changed.")

    def chunk(self, destination, start, recipe):
        np = self.np
        count = destination.size
        require(0 <= count <= CHUNK and 0 <= start <= recipe["elements"] - count, "Chunk exceeds recipe bounds.")
        require(destination.dtype == np.dtype("float32") and destination.flags.c_contiguous,
                "Destination chunk must be contiguous F32.")
        work = self.work[:count]
        np.multiply(self.indices[:count], np.uint32(1664525), out=work)
        word = ((start & MASK) * 1664525 + recipe["seed"]) & MASK
        np.add(work, np.uint32(word), out=work)
        np.right_shift(work, np.uint32(16), out=work)
        signed = work.view(np.int32)
        np.subtract(signed, np.int32(32768), out=signed)
        np.copyto(destination, signed, casting="unsafe")
        np.multiply(destination, np.float32(2.0 ** recipe["exponent"]), out=destination)
        if recipe["offset"]:
            np.add(destination, np.float32(recipe["offset"]), out=destination)

    def fill(self, name, tensor, parameter=True, chunk_elements=CHUNK):
        require(1 <= chunk_elements <= CHUNK, "Chunk size out of range.")
        shape = list(tensor.shape)
        recipe = descriptor(name, shape, parameter)
        array = native_view(tensor)
        flat = array.reshape(-1)
        require(self.np.shares_memory(array, flat), "Unexpected destination reshape copy.")
        value = hashlib.sha256()
        for start in range(0, recipe["elements"], chunk_elements):
            destination = flat[start:start + chunk_elements]
            self.chunk(destination, start, recipe)
            value.update(memoryview(destination).cast("B"))
        return {"name": name, "shape": shape, "stride": list(tensor.stride()),
                "dtype": "float32", "byteOrder": "little", "bytes": recipe["elements"] * 4,
                "sha256": value.hexdigest(), "aligned64": True}


def native_view(tensor):
    require(str(tensor.dtype) == "torch.float32" and tensor.device.type == "cpu" and tensor.is_contiguous(),
            "Expected contiguous CPU F32 tensor.")
    require(tensor.data_ptr() % 64 == 0, "Native tensor must be aligned to 64 bytes.")
    array = tensor.detach().numpy()
    require(array.__array_interface__["data"][0] == tensor.data_ptr(), "NumPy view unexpectedly copied tensor.")
    return array


def tensor_hash(tensor):
    return hashlib.sha256(memoryview(native_view(tensor)).cast("B")).hexdigest()


def write_tensor(tensor, filename, output):
    array = native_view(tensor)
    data = memoryview(array).cast("B")
    sha = hashlib.sha256()
    with (output / filename).open("xb") as stream:
        for start in range(0, len(data), CHUNK * 4):
            chunk = data[start:start + CHUNK * 4]
            stream.write(chunk)
            sha.update(chunk)
    return {"file": filename, "shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": "float32", "byteOrder": "little", "bytes": len(data),
            "aligned64": True, "sha256": sha.hexdigest()}


def memory():
    try:
        if os.name == "nt":
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb),
                    "Process memory observation failed.")
            return {"status": "available", "residentBytes": counters.WorkingSetSize,
                    "peakResidentBytes": counters.PeakWorkingSetSize, "privateBytes": counters.PrivateUsage,
                    "collector": "GetProcessMemoryInfo"}
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        current = None
        if sys.platform.startswith("linux"):
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"): current = int(line.split()[1]) * 1024
        return {"status": "available" if current is not None else "peak_only", "residentBytes": current,
                "peakResidentBytes": peak, "collector": "getrusage_ru_maxrss_and_proc_self_status"}
    except Exception as error:
        return {"status": "unavailable", "errorType": type(error).__name__}


def budget_check(budget, remaining=0):
    observed = memory()
    # Peak is conservative when current RSS is unavailable (e.g. macOS).
    current = observed.get("residentBytes") or observed.get("peakResidentBytes")
    require(current is not None and current > 0, "Process memory could not be observed for admission.")
    if current + remaining > budget:
        raise MemoryError("Declared diagnostic budget is insufficient.")
    return observed


def loaded_paths():
    if sys.platform.startswith("linux"):
        # A library usually owns several maps. Deduplicate exact pathname, never basename.
        seen = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/") and fields[5] not in seen:
                seen.add(fields[5])
                yield Path(fields[5])
    elif os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        handle = kernel.GetCurrentProcess()
        modules = (wintypes.HMODULE * 4096)()
        needed = wintypes.DWORD()
        require(psapi.EnumProcessModules(handle, modules, ctypes.sizeof(modules), ctypes.byref(needed)), "Cannot enumerate modules.")
        require(needed.value <= ctypes.sizeof(modules), "Loaded module list exceeded diagnostic bound.")
        for module in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            text = ctypes.create_unicode_buffer(32768)
            count = psapi.GetModuleFileNameExW(handle, module, text, len(text))
            require(0 < count < len(text), "Cannot observe a module filename.")
            yield Path(text.value)
    elif sys.platform == "darwin":
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        for index in range(dyld._dyld_image_count()):
            name = dyld._dyld_get_image_name(index)
            require(name is not None, "Cannot observe a loaded image.")
            yield Path(os.fsdecode(name))
    else:
        raise NotImplementedError("Unsupported native observation platform.")


def runtime_identity(torch, private):
    global _NATIVE_LIBRARIES
    if _NATIVE_LIBRARIES is None:
        libraries, errors = [], []
        try:
            binding = Path(torch._C.__file__).name
            for path in loaded_paths():
                if path.name != binding and not path.name.lower().startswith(("torch", "libtorch", "c10", "libc10", "libomp", "libiomp", "libgomp")):
                    continue
                record = {"instance": len(libraries) + 1, "name": path.name}
                try:
                    record.update(status="available", bytes=path.stat().st_size, sha256=digest(path))
                except Exception as error:
                    record.update(status="unavailable", errorType=type(error).__name__)
                libraries.append(record)
        except Exception as error:
            errors.append(type(error).__name__)
        _NATIVE_LIBRARIES = {"capturePoint": "after_first_forward_once_per_process",
            "status": "unavailable" if not libraries else "partial" if errors or any(x["status"] != "available" for x in libraries) else "available",
            "libraries": libraries, "errors": errors}
    build = torch.__config__.show()
    for path in (Path.home(), Path(sys.prefix), ROOT, *private):
        build = build.replace(str(path), "<local>")
    requested = os.environ.get("ATEN_CPU_CAPABILITY")
    return {"python": platform.python_version(), "os": platform.system(), "osVersion": platform.version(),
            "architecture": platform.machine(), "torch": torch.__version__, "torchGit": torch.version.git_version,
            "build": build, "threads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads(),
            "gradEnabled": torch.is_grad_enabled(), "requestedCapability": requested or "unset",
            "actualGlobalCapability": torch.backends.cpu.get_cpu_capability(),
            "capabilityScope": "global_ATen_capability_not_per_operator_or_BLAS_dispatch",
            "nativeLibraries": _NATIVE_LIBRARIES}


def source_configuration(helper, case):
    selected = case["configuration"]
    linear = selected["useLinearProjection"]
    config = helper.source_configuration(linear)
    config.update(model_channels=selected["baseChannels"], context_dim=selected["contextSize"],
                  num_heads=-1 if linear else selected["headParameter"],
                  num_head_channels=selected["headParameter"] if linear else -1)
    return config


def execute_case(torch, np, helper, source, case, output, budget, reduced=False, legacy_recipe=None):
    evidence = []
    model_type, operations = helper.source_model_type(source, evidence)
    config = source_configuration(helper, case)
    start, cpu_start = time.perf_counter(), time.process_time()
    with torch.no_grad():
        # Metadata construction derives the definitive schema from upstream code.
        model = model_type(**config, dtype=torch.float32, device="meta", operations=operations)
        named = list(model.named_parameters())
        require(len(named) == 686 and len({name for name, _ in named}) == 686, "Source U-Net parameter topology changed.")
        require(not list(model.named_buffers()), "Unexpected unfilled source module buffer.")
        resident = sum(value.numel() * 4 for _, value in named)
        largest = max(value.numel() * 4 for _, value in named)
        if not reduced:
            require(resident == case["expectedWeightBytes"] and largest == case["largestParameterBytes"], "Source stock descriptor changed.")
        input_bytes = (math.prod(case["latentShape"]) + math.prod(case["contextShape"]) + len(case["timesteps"])) * 4
        before = budget_check(budget, resident + input_bytes + 2*1024**2 + (256 if reduced else 1280)*1024**2)
        del named  # Metadata references only; never a second CPU bank.
        model.to_empty(device="cpu")
        model.requires_grad_(False)
        model.eval()
        filler = Filler(np)
        parameters = []
        for name, value in sorted(model.named_parameters()):
            budget_check(budget)
            parameters.append(filler.fill(name, value))
            if reduced:
                # Bounded source-vs-source input check, never used for stock fill.
                expected = legacy_recipe(name, list(value.shape), parameter=True)
                require(torch.equal(value, expected), "Reduced module parameter differs from legacy source input recipe.")
                del expected
        generated = budget_check(budget)
        generation = {"wallSeconds": time.perf_counter()-start, "cpuSeconds": time.process_time()-cpu_start}
        latent = torch.empty(case["latentShape"], dtype=torch.float32, device="cpu")
        context = torch.empty(case["contextShape"], dtype=torch.float32, device="cpu")
        timesteps = torch.empty([len(case["timesteps"])], dtype=torch.float32, device="cpu")
        filler.fill(case["id"] + "/latent", latent, False)
        filler.fill(case["id"] + "/context", context, False)
        native_view(timesteps)[:] = case["timesteps"]
        inputs = [write_tensor(latent, "latent.f32", output), write_tensor(context, "context.f32", output),
                  write_tensor(timesteps, "timesteps.f32", output)]
        del filler
        executions, identity, output_record = [], None, None
        for repetition in range(3):
            budget_check(budget)
            start, cpu_start = time.perf_counter(), time.process_time()
            result = model._forward(latent, timesteps=timesteps, context=context, y=None, control=None, transformer_options={})
            elapsed = {"wallSeconds": time.perf_counter()-start, "cpuSeconds": time.process_time()-cpu_start}
            require(list(result.shape) == case["outputShape"], "Source output shape changed.")
            sha = tensor_hash(result)
            if repetition == 0:
                output_record = write_tensor(result, "output.f32", output)
                identity = runtime_identity(torch, [source, output])
            del result
            executions.append({"iteration": repetition+1, "sha256": sha, **elapsed, "memoryAfterOutputDispose": budget_check(budget)})
        repeated = len({item["sha256"] for item in executions}) == 1
        document = {"profile": PROFILE, "collectorVersion": COLLECTOR_VERSION,
            "diagnosticOnly": True, "synthetic": True, "pretrainedWeightsUsed": False,
            "modelCompatibility": "not_assessed", "numericalQualification": "not_performed",
            "configurationKind": "reduced_self_test" if reduced else "stock",
            "status": "ok" if repeated else "non_repeatable", "caseId": case["id"],
            "configuration": case["configuration"], "sourceConfig": config, "sourceConfigSha256": canonical_hash(config),
            "sources": evidence, "parameterRecipe": RECIPE, "parameters": parameters, "inputs": inputs,
            "output": output_record, "executions": executions, "outputHashesRepeat": repeated,
            "runtime": identity, "generation": generation,
            "memory": {"beforeGeneration": before, "afterGeneration": generated, "residentParameterBytes": resident,
                "largestParameterBytes": largest, "fillScratchBytes": 2*1024**2, "requestedBudgetBytes": budget,
                "peakGuarantee": False},
            "adapter": {"allocation": "one_meta_module_to_empty_cpu_bank", "fillOrder": "ordinal_parameter_name",
                "fill": "bounded_NumPy_ufuncs_into_native_destination", "chunkElements": CHUNK,
                "legacyRecipeParameterChecks": len(parameters) if reduced else None,
                "parameterHashes": "destination_bytes_no_parameter_payloads_exported",
                "entryPoint": "unchanged_frozen_UNetModel._forward", "attention": "frozen_attention_basic"}}
        return document


def bounded_self_test(torch, np, common, helper, source, output, captured):
    filler, checks = Filler(np), []
    with torch.no_grad():
        for name, shape, parameter in [("test.norm.weight", [13], True), ("test.bias", [13], True),
                ("test.conv.weight", [3, 2, 3, 3], True), ("test.matrix.weight", [11, 7], True),
                ("test.input", [262147], False)]:
            expected = common.synthetic_tensor(name, shape, parameter=parameter)
            tensor = torch.empty(shape, dtype=torch.float32, device="cpu")
            sizes = (1, 3, 7, CHUNK) if tensor.numel() < 100 else (CHUNK-1, CHUNK)
            for chunk in sizes:
                record = filler.fill(name, tensor, parameter, chunk)
                require(torch.equal(expected, tensor), "Bounded recipe differs from unchanged source recipe.")
                require(record["sha256"] == tensor_hash(tensor), "Destination hash mismatch.")
                checks.append({"name": name, "chunkElements": chunk, "sha256": record["sha256"]})
            del expected, tensor
        for start in (CHUNK-3, 2**32-3):
            recipe = descriptor("test.wrap", [2**32+16], False)
            values = np.empty(7, dtype=np.float32)
            filler.chunk(values, start, recipe)
            expected = b"".join(struct.pack("<f", ((((i*1664525+recipe["seed"]) & MASK) >> 16)-32768) * 2.0**-15)
                                for i in range(start,start+7))
            require(memoryview(values).cast("B").tobytes() == expected, "Independent scalar modular recipe differs.")
            checks.append({"name": "test.wrap", "start": start, "sha256": hashlib.sha256(expected).hexdigest()})
        for invalid in (("", [1]), ("bad", []), ("bad", [0]), ("bad", [2**63])):
            try:
                descriptor(*invalid, False)
            except ValueError:
                checks.append({"validationRejected": True})
            else:
                raise AssertionError("Malformed descriptor accepted.")
    del filler
    for model in ("sd15", "sd2"):
        directory = output / model
        directory.mkdir()
        case = {"id": "sd-stock-source-selftest-v1/"+model+"-reduced/square",
                "configuration": {"baseChannels": 32, "contextSize": 16,
                    "headMode": "fixedCount" if model == "sd15" else "fixedSize",
                    "headParameter": 4 if model == "sd15" else 8, "useLinearProjection": model == "sd2"},
                "latentShape": [1,4,8,8], "contextShape": [1,3,16], "timesteps": [0.125], "outputShape": [1,4,8,8]}
        document = execute_case(torch, np, helper, source, case, directory, 2048*1024**2, reduced=True,
                                legacy_recipe=common.synthetic_tensor)
        publish_manifest(directory, document, captured)
    final_provenance = verify_provenance(captured)
    write_json(output / "self-test.json", {"status": "ok", "collectorVersion": COLLECTOR_VERSION,
        "configurationKind": "reduced_self_test",
        "recipeChecks": checks, "reducedGraphs": ["sd15", "sd2"], "stockExecuted": False,
        "dependencyLockAttestation": {"before": captured.lock_before, "after": captured.lock_after},
        "provenanceVerifiedAfterExecution": final_provenance})


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("Invalid laboratory arguments; use --help.")


def main():
    parser = Parser(description=__doc__)
    parser.add_argument("--model", choices=("sd15", "sd2"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--inspect", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--memory-budget-mib", type=int)
    parser.add_argument("--chunk-elements", type=int, default=CHUNK)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    require(args.chunk_elements == CHUNK and args.repeat == 3, "The frozen profile requires 1 MiB chunks and three repetitions.")
    require(sys.byteorder == "little", "Only little-endian hosts are supported.")
    require(digest(LAB / "cases-v1.json") == CASES_SHA256, "Frozen case definition changed.")
    protocol = json.loads((LAB / "cases-v1.json").read_text(encoding="utf-8"))
    require(protocol["profile"] == PROFILE and protocol["parameterRecipe"] == RECIPE, "Unexpected case protocol.")
    require(protocol["comparison"] == {"absoluteTolerance": 3e-5, "relativeTolerance": 3e-5,
        "nonFinite": "class-and-infinity-sign", "shapeDtypeByteOrderAndInputs": "exact"}, "Comparison protocol changed.")
    for name, sha in HELPERS.items():
        require(digest(ROOT / "labs/sd-source" / name) == sha, "Frozen helper hash mismatch.")
    require(not args.self_test or (args.model is None and not args.synthetic and args.memory_budget_mib is None),
            "Self-test has fixed reduced configurations and budget.")
    require(args.self_test or args.model is not None, "Select one stock model.")
    require(args.memory_budget_mib is None or args.memory_budget_mib > 0, "Budget must be positive.")
    if not (args.execute or args.self_test):
        require(args.output is None and args.source_directory is None, "Plan does not use input or output directories.")
        case = protocol["cases"][args.model]
        estimate = case["expectedWeightBytes"] + (math.prod(case["latentShape"])+math.prod(case["contextShape"])+1)*4 + 2*1024**2
        estimate += sum(protocol["execution"][key] for key in ("runtimeAllowanceBytes", "graphAllowanceBytes", "headroomBytes"))
        if args.memory_budget_mib is not None and estimate > args.memory_budget_mib*1024**2:
            raise MemoryError("Declared budget below planning estimate.")
        print(json.dumps({"status": "ok", "operation": "plan", "profile": PROFILE,
            "collectorVersion": COLLECTOR_VERSION, "case": case,
            "estimatedProcessBytes": estimate, "estimateIsPeakGuarantee": False,
            "nativeInitialized": False, "weightsGenerated": False, "modelCompatibility": "not_assessed"}))
        return
    require(args.source_directory is not None and args.output is not None, "Execution requires source and output directories.")
    require(args.self_test or (args.synthetic and args.memory_budget_mib is not None), "Stock execution requires explicit synthetic mode and budget.")
    if not args.self_test:
        case = protocol["cases"][args.model]
        minimum = case["expectedWeightBytes"] + (math.prod(case["latentShape"])+math.prod(case["contextShape"])+1)*4 + 2*1024**2
        minimum += sum(protocol["execution"][key] for key in ("runtimeAllowanceBytes", "graphAllowanceBytes", "headroomBytes"))
        if args.memory_budget_mib*1024**2 < minimum:
            raise MemoryError("Declared budget below planning estimate.")
    source = args.source_directory.resolve(strict=True)
    require(source.is_dir() and args.output.is_absolute(), "Source must be a directory and output must be absolute.")
    output = args.output.resolve()
    require(not output.exists(), "Output must be absent; existing evidence is never overwritten.")
    for protected in (ROOT, source):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output), "Output overlaps protected input.")
    require(sys.version_info[:3] == (3,12,10), "Use Python 3.12.10.")
    target = {("Windows","AMD64"): "win-x64", ("Linux","x86_64"): "linux-x64", ("Darwin","arm64"): "osx-arm64"}.get((platform.system(),platform.machine()))
    require(target in LOCKS, "Unsupported laboratory platform.")
    lock_attestation((ROOT / "labs/clip-source" / ("requirements-"+target+".txt")).read_bytes(), LOCKS[target])
    require(importlib.metadata.version("torch") in ("2.10.0", "2.10.0+cpu"), "Use pinned CPU PyTorch.")
    require(importlib.metadata.version("numpy") == "2.2.6" and importlib.metadata.version("einops") == "0.8.1", "Use pinned dependencies.")
    capability = os.environ.get("ATEN_CPU_CAPABILITY")
    require(capability in (None,"default","avx2","avx512"), "Unknown requested ATen capability.")
    captured = capture_provenance(source, target)
    import torch
    import numpy as np
    require(torch.version.cuda is None, "CUDA is outside the CPU source profile.")
    require(torch.version.git_version == "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0", "Unexpected PyTorch source revision.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    common, helper = verified_helpers()
    output.mkdir(parents=True)
    write_json(output / "collection.json", {"profile": PROFILE, "collectorVersion": COLLECTOR_VERSION,
        "status": "collection_started_not_qualified",
        "operation": "reduced_self_test" if args.self_test else "stock", "target": target,
        "backendCommit": protocol["backendCommit"], "protocolSha256": digest(LAB / "cases-v1.json"),
        "scriptSha256": captured["reference.py"][1], "verifiedHelpers": HELPERS,
        "dependencyLockSha256": captured.lock_before["rawSha256"],
        "dependencyLockAttestation": captured.lock_before,
        "provenanceBeforeExecution": {label: sha for label, (_, sha) in captured.items()}})
    if args.self_test:
        bounded_self_test(torch,np,common,helper,source,output,captured)
    else:
        document = execute_case(torch,np,helper,source,protocol["cases"][args.model],output,args.memory_budget_mib*1024**2)
        publish_manifest(output, document, captured)
    print(json.dumps({"status": "ok", "operation": "reduced_self_test" if args.self_test else "stock",
        "profile": PROFILE, "modelCompatibility": "not_assessed", "output": "self-test.json" if args.self_test else "manifest.json"}))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(json.dumps({"status": "cancelled", "modelCompatibility": "not_assessed"}), file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(json.dumps({"status": "error", "errorType": type(error).__name__, "modelCompatibility": "not_assessed"}), file=sys.stderr)
        sys.exit(3 if isinstance(error,MemoryError) else 1)
