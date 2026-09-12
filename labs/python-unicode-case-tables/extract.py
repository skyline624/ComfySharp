"""Statically extract pinned CPython full upper/title C tables, without executing casing."""
import argparse
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
PROVENANCE_SHA256 = "f453902eb2ba3a21f042667b17d132dcaa052555005e9c77c48fbc4bb104f4fa"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate provenance property")
        value[key] = item
    return value


def scalar(value):
    return 0 <= value <= 0x10FFFF and not 0xD800 <= value <= 0xDFFF


def array(source, name, records=False):
    matches = list(re.finditer(r"\b" + re.escape(name) + r"\[\]\s*=\s*\{(.*?)\n\};", source, re.S))
    if len(matches) != 1:
        raise ValueError("Expected exactly one C array: " + name)
    body = re.sub(r"/\*.*?\*/", "", matches[0][1], flags=re.S)
    if records:
        groups = re.findall(r"\{([^{}]*)\}", body)
        if re.sub(r"\{[^{}]*\}|[\s,]", "", body):
            raise ValueError("Unsupported C record syntax")
        result = []
        for group in groups:
            if not re.fullmatch(r"\s*-?\d+(?:\s*,\s*-?\d+){5}\s*", group):
                raise ValueError("Expected six decimal C record fields")
            result.append([int(item) for item in group.split(",")])
        return result
    if re.sub(r"-?\d+|[\s,]", "", body):
        raise ValueError("Unsupported C integer array syntax")
    return [int(item) for item in re.findall(r"-?\d+", body)]


def extract(directory):
    extractor_hash = digest(Path(__file__).read_bytes())
    provenance_raw = (BASE / "provenance.json").read_bytes()
    if digest(provenance_raw) != PROVENANCE_SHA256:
        raise ValueError("Pinned provenance changed")
    provenance = json.loads(provenance_raw, object_pairs_hook=unique_object)
    original = {}
    for name, pin in provenance["sources"].items():
        raw = (directory / name).read_bytes()
        if len(raw) != pin["bytes"] or digest(raw) != pin["sha256"]:
            raise ValueError("Pinned source differs: " + name)
        original[name] = raw.decode("utf-8")
    types = original["Objects/unicodetype_db.h"]
    ctype = original["Objects/unicodectype.c"]
    if re.findall(r'^UNIDATA_VERSION = "([^"]+)"', original["Tools/unicode/makeunicodedata.py"], re.M) != ["15.0.0"]:
        raise ValueError("Unexpected Unicode version")
    if re.findall(r"^#define SHIFT (\d+)$", types, re.M) != ["7"]:
        raise ValueError("Unexpected C index shift")
    if re.findall(r"^#define EXTENDED_CASE_MASK (0x[0-9a-fA-F]+)$", ctype, re.M) != ["0x4000"]:
        raise ValueError("Unexpected full mapping mask")
    records = array(types, "_PyUnicode_TypeRecords", True)
    extended = array(types, "_PyUnicode_ExtendedCase")
    index1 = array(types, "index1")
    index2 = array(types, "index2")
    if (len(records), len(extended), len(index1), len(index2)) != (504, 1236, 8704, 36224):
        raise ValueError("Unexpected C table dimensions")
    if any(i < 0 or (i + 1) << 7 > len(index2) for i in index1):
        raise ValueError("First-level C index out of bounds")
    if any(i < 0 or i >= len(records) for i in index2) or not all(scalar(c) for c in extended):
        raise ValueError("Invalid type index or extended scalar")
    mappings = {"upper": [], "title": []}
    count = 0
    for cp in range(0x110000):
        if not scalar(cp):
            continue
        count += 1
        record = records[index2[(index1[cp >> 7] << 7) + (cp & 127)]]
        # C struct order is upper, lower, title, decimal, digit, flags.
        for name, field in (("upper", 0), ("title", 2)):
            encoded = record[field]
            if record[5] & 0x4000:
                start, length = encoded & 0xFFFF, encoded >> 24
                if not 1 <= length <= 3 or start + length > len(extended):
                    raise ValueError("Invalid full mapping expansion")
                mapped = extended[start:start + length]
            else:
                mapped = [cp + encoded]
            if not all(scalar(c) for c in mapped):
                raise ValueError("Full mapping contains a non-scalar")
            if mapped != [cp]:
                mappings[name].append([cp, mapped])
    if count != 1112064:
        raise ValueError("Incomplete scalar traversal")
    counts = {"scalars": count, "typeRecords": len(records), "extendedScalars": len(extended),
              "index1": len(index1), "index2": len(index2)}
    for name, values in mappings.items():
        counts[name + "Mappings"] = len(values)
        counts[name + "Expansions"] = sum(len(value[1]) > 1 for value in values)
    result = {"schema": 1, "pythonVersion": "3.12.10", "unicodeVersion": "15.0.0",
              "cpythonCommit": provenance["commit"], "provenanceSha256": PROVENANCE_SHA256,
              "extractorSha256": extractor_hash, **mappings, "counts": counts}
    # Exact byte identities are checked again before any generated resource is returned.
    if digest(Path(__file__).read_bytes()) != extractor_hash or digest((BASE / "provenance.json").read_bytes()) != PROVENANCE_SHA256:
        raise ValueError("Extraction code/provenance changed during extraction")
    for name, pin in provenance["sources"].items():
        raw = (directory / name).read_bytes()
        if len(raw) != pin["bytes"] or digest(raw) != pin["sha256"]:
            raise ValueError("Source changed during extraction")
    return (json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n").encode(), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--output", type=Path, help="Create an absent resource exclusively")
    target.add_argument("--check", type=Path, help="Compare regenerated bytes without writing")
    args = parser.parse_args()
    raw, counts = extract(args.source_directory)
    if args.output:
        path = args.output.resolve()
        if path.is_relative_to(args.source_directory.resolve()):
            raise ValueError("Refusing to write into pinned original sources")
        with path.open("xb") as stream:
            stream.write(raw)
    if args.check and args.check.read_bytes() != raw:
        raise ValueError("Generated bytes differ from checked resource")
    print(json.dumps({"bytes": len(raw), "sha256": digest(raw), "counts": counts,
                      "builtinCasingExecuted": False, "upstreamSourceExecuted": False}))


if __name__ == "__main__":
    main()
