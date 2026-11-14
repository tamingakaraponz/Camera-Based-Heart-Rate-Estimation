"""Validate the development-only inputs for the new dataset benchmark.

This first stage never calculates heart rate or opens held-out videos.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def development_inputs(split_path):
    split = json.loads(split_path.read_text(encoding="utf-8"))
    inventory_path = (ROOT / split["inventory"]).resolve()
    if not inventory_path.is_relative_to(ROOT):
        raise ValueError("Inventory must be inside this project")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    dataset = Path(inventory["dataset_root"]).resolve(strict=True)
    dev, held = set(split["development"]), set(split["held_out"])
    if not dev or not held or dev & held:
        raise ValueError("Development and held-out lists must be nonempty and disjoint")
    records = [r for r in inventory["records"] if r["subject"] in dev]
    if len(records) != len(dev) or {r["subject"] for r in records} != dev:
        raise ValueError("Development subjects missing or duplicated")
    result = []
    for record in records:
        if not record.get("pair_present") or not record.get("metadata_checks_passed"):
            raise ValueError(f"Invalid inventory record: {record['subject']}")
        paths = []
        for kind in ("video", "reference"):
            path = Path(record[kind]).resolve(strict=True)
            if not path.is_relative_to(dataset) or path.parent.name != record["subject"]:
                raise ValueError(f"Unexpected source path: {path}")
            stat = path.stat()
            if stat.st_size != record[kind + "_bytes"] or stat.st_mtime_ns != record[kind + "_mtime_ns"]:
                raise ValueError(f"Source changed since inventory: {path}")
            paths.append(str(path))
        result.append({"subject": record["subject"], "video": paths[0], "reference": paths[1]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "new_dataset_split.json")
    args = parser.parse_args()
    try:
        inputs = development_inputs(args.split.resolve(strict=True))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "inputs_validated_only", "python": sys.executable,
                      "development": inputs, "heart_rate_computed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
