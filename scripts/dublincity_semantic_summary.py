#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import cloudComPy as cc


DEFAULT_INPUT_GLOB = "/data2/point-cloud-datasets/DublinCity/**/*.bin"
DEFAULT_MAX_FILES = 20
SEMANTIC_KEYWORDS = ("class", "label", "semantic")
RICH_LABEL_TOKENS = ("facade", "roof", "window", "door")

SUMMARY_JSON_PATH = "workspace/dublincity_semantic_summary.json"
SUMMARY_CSV_PATH = "workspace/dublincity_semantic_summary.csv"
SUMMARY_NOTES_PATH = "workspace/dublincity_semantic_notes.txt"
FILELIST_PATH = "workspace/dublincity_bin_filelist.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize scalar fields in DublinCity .bin files")
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    return parser.parse_args()


def load_entities(path):
    if hasattr(cc, "importFile"):
        loaded = cc.importFile(path)
        if loaded is not None:
            return "importFile", loaded
    if hasattr(cc, "loadFile"):
        loaded = cc.loadFile(path)
        if loaded is not None:
            return "loadFile", loaded
    raise RuntimeError(f"Unable to load file with importFile/loadFile: {path}")


def is_point_cloud_like(entity):
    if entity is None:
        return False
    type_name = type(entity).__name__.lower()
    if "pointcloud" in type_name:
        return True
    has_sf = hasattr(entity, "getNumberOfScalarFields")
    has_size = hasattr(entity, "size") or hasattr(entity, "getNumberOfPoints")
    return has_sf and has_size


def get_children(entity):
    if not hasattr(entity, "getChildrenNumber") or not hasattr(entity, "getChild"):
        return []
    child_count = entity.getChildrenNumber()
    if not isinstance(child_count, int) or child_count <= 0:
        return []
    return [entity.getChild(i) for i in range(child_count)]


def get_point_count(cloud):
    if hasattr(cloud, "size"):
        point_count = cloud.size()
        if isinstance(point_count, int):
            return point_count
    if hasattr(cloud, "getNumberOfPoints"):
        point_count = cloud.getNumberOfPoints()
        if isinstance(point_count, int):
            return point_count
    return -1


def get_scalar_fields(cloud):
    if not hasattr(cloud, "getNumberOfScalarFields"):
        return []
    sf_count = cloud.getNumberOfScalarFields()
    if not isinstance(sf_count, int) or sf_count <= 0:
        return []

    fields = []
    for i in range(sf_count):
        name = cloud.getScalarFieldName(i) if hasattr(cloud, "getScalarFieldName") else f"sf_{i}"
        if not isinstance(name, str):
            name = f"sf_{i}"
        sf_obj = cloud.getScalarField(i) if hasattr(cloud, "getScalarField") else None
        fields.append((name, sf_obj))
    return fields


def scalar_to_numpy(sf_obj):
    if sf_obj is None:
        return np.array([], dtype=np.float64)

    if hasattr(sf_obj, "toNpArrayCopy"):
        arr = sf_obj.toNpArrayCopy()
        if arr is not None:
            return np.asarray(arr, dtype=np.float64)

    size = sf_obj.currentSize() if hasattr(sf_obj, "currentSize") else None
    if not isinstance(size, int):
        size = sf_obj.size() if hasattr(sf_obj, "size") else None
    if not isinstance(size, int) or size <= 0:
        return np.array([], dtype=np.float64)

    out = np.empty(size, dtype=np.float64)
    for i in range(size):
        out[i] = float(sf_obj.getValue(i))
    return out


def finite_values(arr):
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def is_integerish(arr, tolerance=1e-6):
    if arr.size == 0:
        return False
    rounded = np.rint(arr)
    return bool(np.max(np.abs(arr - rounded)) <= tolerance)


def update_field_stats(field_stats, field_name, values):
    stats = field_stats[field_name]
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    stats["min"] = vmin if stats["min"] is None else min(stats["min"], vmin)
    stats["max"] = vmax if stats["max"] is None else max(stats["max"], vmax)
    stats["num_values"] += int(values.size)

    if is_integerish(values):
        ints = np.rint(values).astype(np.int64)
        unique_vals, counts = np.unique(ints, return_counts=True)
        for val, count in zip(unique_vals, counts):
            stats["histogram"][int(val)] += int(count)


def top_counts(counter, top_k=8):
    return [{"value": int(v), "count": int(c)} for v, c in counter.most_common(top_k)]


def analyze_loaded_entities(loaded, global_field_stats):
    visited = set()
    stack = [loaded]

    result = {
        "entities": 0,
        "point_clouds": 0,
        "total_points": 0,
        "scalar_fields_seen": set(),
        "candidate_fields": set(),
        "candidate_counters": defaultdict(Counter),
    }

    while stack:
        node = stack.pop()

        if isinstance(node, (list, tuple)):
            stack.extend(node)
            continue

        if node is None or isinstance(node, (str, bytes, int, float, bool)):
            continue

        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        result["entities"] += 1

        if is_point_cloud_like(node):
            result["point_clouds"] += 1
            point_count = get_point_count(node)
            if point_count > 0:
                result["total_points"] += point_count

            for field_name, sf_obj in get_scalar_fields(node):
                result["scalar_fields_seen"].add(field_name)
                values = finite_values(scalar_to_numpy(sf_obj))
                if values.size == 0:
                    continue

                update_field_stats(global_field_stats, field_name, values)

                lower_name = field_name.lower()
                if any(k in lower_name for k in SEMANTIC_KEYWORDS):
                    result["candidate_fields"].add(field_name)
                    if is_integerish(values):
                        ints = np.rint(values).astype(np.int64)
                        unique_vals, counts = np.unique(ints, return_counts=True)
                        for val, count in zip(unique_vals, counts):
                            result["candidate_counters"][field_name][int(val)] += int(count)

        stack.extend(get_children(node))

    return result


def choose_best_candidate(candidate_counters):
    if not candidate_counters:
        return "", 0, ""

    best_name = ""
    best_unique = -1
    best_total = -1

    for name, counter in candidate_counters.items():
        unique_count = len(counter)
        total_count = int(sum(counter.values()))
        if unique_count > best_unique or (unique_count == best_unique and total_count > best_total):
            best_name = name
            best_unique = unique_count
            best_total = total_count

    top_text = ";".join([f"{v}:{c}" for v, c in candidate_counters[best_name].most_common(8)])
    return best_name, best_unique, top_text


def write_outputs(discovered_files, scanned_records, global_field_stats, candidate_union, richer_labels):
    os.makedirs("workspace", exist_ok=True)

    with open(FILELIST_PATH, "w", encoding="utf-8") as f:
        for path in discovered_files[:30]:
            f.write(path + "\n")

    field_json = {}
    for field_name, stats in sorted(global_field_stats.items()):
        field_json[field_name] = {
            "min": stats["min"],
            "max": stats["max"],
            "num_values": stats["num_values"],
            "histogram": {str(v): int(c) for v, c in sorted(stats["histogram"].items())},
        }

    summary_json = {
        "files_discovered": len(discovered_files),
        "files_scanned": len(scanned_records),
        "candidate_semantic_fields": sorted(candidate_union),
        "richer_labels_detected": richer_labels,
        "fields": field_json,
        "files": scanned_records,
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    csv_columns = [
        "file_path",
        "load_method",
        "entities",
        "point_clouds",
        "total_points",
        "scalar_fields_seen",
        "candidate_fields",
        "best_candidate_field",
        "best_candidate_unique_classes",
        "best_candidate_top_counts",
    ]
    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in scanned_records:
            writer.writerow(
                {
                    "file_path": row["file_path"],
                    "load_method": row["load_method"],
                    "entities": row["entities"],
                    "point_clouds": row["point_clouds"],
                    "total_points": row["total_points"],
                    "scalar_fields_seen": "|".join(row["scalar_fields_seen"]),
                    "candidate_fields": "|".join(row["candidate_fields"]),
                    "best_candidate_field": row["best_candidate_field"],
                    "best_candidate_unique_classes": row["best_candidate_unique_classes"],
                    "best_candidate_top_counts": row["best_candidate_top_counts"],
                }
            )

    max_candidate_classes = 0
    strongest_field = ""
    for field_name in sorted(candidate_union):
        stats = global_field_stats[field_name]
        class_count = len(stats["histogram"])
        if class_count > max_candidate_classes:
            max_candidate_classes = class_count
            strongest_field = field_name

    if not candidate_union:
        conclusion = "No likely semantic fields were detected (no scalar field names with class/label/semantic)."
    elif richer_labels:
        conclusion = (
            "Likely rich semantic labels are present: candidate fields exist and either class cardinality exceeds 6 "
            "or field names contain building-part tokens such as facade/roof/window/door."
        )
    else:
        conclusion = (
            "Candidate semantic fields exist but labels appear coarse (<=6 classes and no facade/roof/window/door tokens in names)."
        )

    notes_lines = [
        "DublinCity semantic summary",
        f"files discovered: {len(discovered_files)}",
        f"files scanned: {len(scanned_records)}",
        f"candidate semantic fields: {', '.join(sorted(candidate_union)) if candidate_union else '<none>'}",
        f"richer labels detected: {richer_labels}",
        f"strongest field by class count: {strongest_field or '<none>'} ({max_candidate_classes} classes)",
        "",
        "Conclusion:",
        conclusion,
        "",
        f"JSON: {SUMMARY_JSON_PATH}",
        f"CSV: {SUMMARY_CSV_PATH}",
        f"File list: {FILELIST_PATH}",
    ]
    with open(SUMMARY_NOTES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(notes_lines) + "\n")


def main():
    args = parse_args()

    all_files = sorted(glob.glob(args.input_glob, recursive=True))
    scan_files = all_files[: max(0, args.max_files)]

    global_field_stats = defaultdict(
        lambda: {
            "min": None,
            "max": None,
            "num_values": 0,
            "histogram": Counter(),
        }
    )

    scanned_records = []
    candidate_union = set()

    for path in scan_files:
        load_method, loaded = load_entities(path)
        analyzed = analyze_loaded_entities(loaded, global_field_stats)

        best_field, best_unique, best_top = choose_best_candidate(analyzed["candidate_counters"])

        candidate_union.update(analyzed["candidate_fields"])

        scanned_records.append(
            {
                "file_path": path,
                "load_method": load_method,
                "entities": analyzed["entities"],
                "point_clouds": analyzed["point_clouds"],
                "total_points": analyzed["total_points"],
                "scalar_fields_seen": sorted(analyzed["scalar_fields_seen"]),
                "candidate_fields": sorted(analyzed["candidate_fields"]),
                "best_candidate_field": best_field,
                "best_candidate_unique_classes": int(best_unique if best_unique >= 0 else 0),
                "best_candidate_top_counts": best_top,
            }
        )

    max_classes = 0
    for field_name in candidate_union:
        max_classes = max(max_classes, len(global_field_stats[field_name]["histogram"]))

    token_hit = any(any(token in name.lower() for token in RICH_LABEL_TOKENS) for name in candidate_union)
    richer_labels = bool(max_classes > 6 or token_hit)

    write_outputs(all_files, scanned_records, global_field_stats, candidate_union, richer_labels)

    print(f"files_found={len(all_files)} files_scanned={len(scanned_records)}")
    union_text = ", ".join(sorted(candidate_union)) if candidate_union else "<none>"
    print(f"candidate_semantic_fields={union_text}")
    print(f"richer_labels={richer_labels} (max_candidate_classes={max_classes}, token_hit={token_hit})")
    print(f"filelist={FILELIST_PATH}")
    print(f"json={SUMMARY_JSON_PATH}")
    print(f"csv={SUMMARY_CSV_PATH}")
    print(f"notes={SUMMARY_NOTES_PATH}")


if __name__ == "__main__":
    main()
