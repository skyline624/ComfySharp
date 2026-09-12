"""Statically decode pinned CPython C data. Never execute C/Python source or builtin casing."""
import argparse
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
PROVENANCE_SHA256 = "8193e086fb80c5597d7fc6c92ffc8749aab78b469b77bcd0681b3da82cb0eff6"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON property")
        result[key] = value
    return result


def scalar(value):
    return 0 <= value <= 0x10FFFF and not 0xD800 <= value <= 0xDFFF


def array(source, name, records=False):
    matches = list(re.finditer(r"\b" + re.escape(name) + r"\[\]\s*=\s*\{(.*?)\n\};", source, re.S))
    if len(matches) != 1:
        raise ValueError("Expected one C data array: " + name)
    body = re.sub(r"/\*.*?\*/", "", matches[0][1], flags=re.S)
    if records:
        groups = re.findall(r"\{([^{}]*)\}", body)
        if re.sub(r"\{[^{}]*\}|[\s,]", "", body):
            raise ValueError("Unsupported C record syntax")
        result = []
        for group in groups:
            if not re.fullmatch(r"\s*-?\d+(?:\s*,\s*-?\d+){5}\s*", group):
                raise ValueError("Expected six decimal C record fields")
            result.append([int(x) for x in group.split(",")])
        return result
    if re.sub(r"-?\d+|[\s,]", "", body):
        raise ValueError("Unsupported C integer array syntax")
    return [int(x) for x in re.findall(r"-?\d+", body)]


def ranges(values):
    result = []
    for value in values:
        if result and result[-1][1] + 1 == value:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def extract(directory):
    extractor_hash = digest(Path(__file__).read_bytes())
    provenance_raw = (BASE / "provenance.json").read_bytes()
    if digest(provenance_raw) != PROVENANCE_SHA256:
        raise ValueError("Extraction provenance changed")
    provenance = json.loads(provenance_raw, object_pairs_hook=unique_object)
    original = {}
    for path, pin in provenance["sources"].items():
        raw = (directory / path).read_bytes()
        if len(raw) != pin["bytes"] or digest(raw) != pin["sha256"]:
            raise ValueError("Pinned source bytes differ: " + path)
        original[path] = raw.decode("utf-8")
    types = original["Objects/unicodetype_db.h"]
    ctype = original["Objects/unicodectype.c"]
    generator = original["Tools/unicode/makeunicodedata.py"]
    if re.findall(r'^UNIDATA_VERSION = "([^"]+)"', generator, re.M) != ["15.0.0"]:
        raise ValueError("Unexpected Unicode database version")
    shifts = re.findall(r"^#define SHIFT (\d+)$", types, re.M)
    if shifts != ["7"]:
        raise ValueError("Unexpected CPython index shift")
    shift = int(shifts[0])
    for name, value in [("CASED_MASK", 0x2000), ("CASE_IGNORABLE_MASK", 0x1000), ("EXTENDED_CASE_MASK", 0x4000)]:
        masks = re.findall(r"^#define " + name + r" (0x[0-9a-fA-F]+)$", ctype, re.M)
        if len(masks) != 1 or int(masks[0], 16) != value:
            raise ValueError("Unexpected property bit: " + name)
    records = array(types, "_PyUnicode_TypeRecords", True)
    extended = array(types, "_PyUnicode_ExtendedCase")
    index1 = array(types, "index1")
    index2 = array(types, "index2")
    if len(index1) != (0x110000 >> shift) or len(index2) % (1 << shift):
        raise ValueError("Unexpected index dimensions")
    if any(i < 0 or (i + 1) << shift > len(index2) for i in index1):
        raise ValueError("First-level index outside table")
    if any(i < 0 or i >= len(records) for i in index2) or not all(scalar(c) for c in extended):
        raise ValueError("Invalid second-level index or expansion scalar")
    mappings, cased, ignorable = [], [], []
    scalar_count = 0
    for cp in range(0x110000):
        if not scalar(cp):
            continue
        scalar_count += 1
        record = records[index2[(index1[cp >> shift] << shift) + (cp & ((1 << shift) - 1))]]
        lower, flags = record[1], record[5]
        if flags & 0x4000:
            start, length = lower & 0xFFFF, lower >> 24
            if not 1 <= length <= 3 or start + length > len(extended):
                raise ValueError("Invalid full lowercase expansion")
            mapped = extended[start:start + length]
        else:
            mapped = [cp + lower]
        if not all(scalar(c) for c in mapped):
            raise ValueError("Lowercase mapping is not a Unicode scalar")
        if mapped != [cp]:
            mappings.append([cp, mapped])
        if flags & 0x2000:
            cased.append(cp)
        if flags & 0x1000:
            ignorable.append(cp)
    if scalar_count != 1112064:
        raise ValueError("Incomplete scalar traversal")
    result = {
        "schema": 1, "pythonVersion": "3.12.10", "unicodeVersion": "15.0.0",
        "cpythonCommit": provenance["commit"], "provenanceSha256": PROVENANCE_SHA256,
        "extractorSha256": extractor_hash,
        "lower": mappings, "casedRanges": ranges(cased), "caseIgnorableRanges": ranges(ignorable),
        "counts": {"scalars": scalar_count, "typeRecords": len(records), "extendedScalars": len(extended),
                   "index1": len(index1), "index2": len(index2), "lowerMappings": len(mappings),
                   "expandingMappings": sum(len(m[1]) > 1 for m in mappings),
                   "casedScalars": len(cased), "caseIgnorableScalars": len(ignorable)}
    }
    # Recheck the actual files after extraction before returning any generated bytes.
    if digest((BASE / "provenance.json").read_bytes()) != PROVENANCE_SHA256:
        raise ValueError("Provenance changed during extraction")
    if digest(Path(__file__).read_bytes()) != extractor_hash:
        raise ValueError("Extractor changed during extraction")
    for path, pin in provenance["sources"].items():
        if digest((directory / path).read_bytes()) != pin["sha256"]:
            raise ValueError("Source changed during extraction")
    return (json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n").encode(), result["counts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path, help="Create an absent JSON resource exclusively")
    group.add_argument("--check", type=Path, help="Compare generated bytes to an existing resource without writing")
    args = parser.parse_args()
    raw, counts = extract(args.source_directory)
    if args.output:
        target = args.output.resolve()
        if target.is_relative_to(args.source_directory.resolve()):
            raise ValueError("Do not write into the pinned source directory")
        with target.open("xb") as stream:
            stream.write(raw)
    if args.check and args.check.read_bytes() != raw:
        raise ValueError("Generated bytes differ from the checked resource")
    print(json.dumps({"bytes": len(raw), "sha256": digest(raw), "counts": counts,
                      "builtinCasingExecuted": False, "upstreamSourceExecuted": False}))


if __name__ == "__main__":
    main()
