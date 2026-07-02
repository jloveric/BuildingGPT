#!/usr/bin/env python3
import os
import math
from collections import Counter

import numpy as np
import cloudComPy as cc

INPUT_PATH = "/data2/point-cloud-datasets/DublinCity/T_315500_234500_NE.bin"
OUTPUT_PATH = "workspace/dublincity_entity_report_NE.txt"
SAMPLE_LIMIT = 1_000_000
TOP_K = 12
KEYWORDS = ("label", "class", "semantic")


def safe_call(obj, method_name, *args):
    if hasattr(obj, method_name):
        method = getattr(obj, method_name)
        try:
            return method(*args)
        except Exception:
            return None
    return None


def load_entities(path):
    attempts = []
    for method_name in ("importFile", "loadFile"):
        if hasattr(cc, method_name):
            method = getattr(cc, method_name)
            try:
                loaded = method(path)
                return method_name, loaded, attempts
            except Exception as exc:
                attempts.append((method_name, str(exc)))
    raise RuntimeError(f"Failed to load {path}. Attempts: {attempts}")


def entity_name(entity, fallback="<unnamed>"):
    n = safe_call(entity, "getName")
    if isinstance(n, str) and n:
        return n
    return fallback


def class_id(entity):
    return safe_call(entity, "getClassID")


def is_point_cloud_like(entity):
    if entity is None:
        return False

    tname = type(entity).__name__.lower()
    if "pointcloud" in tname:
        return True

    if hasattr(entity, "getNumberOfScalarFields") and hasattr(entity, "size"):
        return True

    point_cloud_type = None
    if hasattr(cc, "CC_TYPES") and hasattr(cc.CC_TYPES, "POINT_CLOUD"):
        point_cloud_type = cc.CC_TYPES.POINT_CLOUD
    if point_cloud_type is not None and hasattr(entity, "isKindOf"):
        try:
            if entity.isKindOf(point_cloud_type):
                return True
        except Exception:
            pass

    cid = class_id(entity)
    if isinstance(cid, int) and point_cloud_type is not None and cid == int(point_cloud_type):
        return True

    return False


def get_children(entity):
    out = []
    n = safe_call(entity, "getChildrenNumber")
    if isinstance(n, int) and n > 0 and hasattr(entity, "getChild"):
        for i in range(n):
            child = safe_call(entity, "getChild", i)
            out.append((f"child[{i}]", child))
    return out


def get_point_count(cloud):
    size = safe_call(cloud, "size")
    if isinstance(size, int):
        return size
    size = safe_call(cloud, "getNumberOfPoints")
    if isinstance(size, int):
        return size
    return None


def get_scalar_fields(cloud):
    result = []
    n = safe_call(cloud, "getNumberOfScalarFields")
    if not isinstance(n, int) or n < 0:
        return result
    for i in range(n):
        name = safe_call(cloud, "getScalarFieldName", i)
        sf = safe_call(cloud, "getScalarField", i)
        if not isinstance(name, str):
            name = f"sf_{i}"
        result.append((i, name, sf))
    return result


def scalar_to_numpy(sf):
    if sf is None:
        return np.array([], dtype=np.float64)

    if hasattr(sf, "toNpArrayCopy"):
        try:
            arr = sf.toNpArrayCopy()
            if arr is not None:
                return np.asarray(arr)
        except Exception:
            pass

    n = safe_call(sf, "currentSize")
    if not isinstance(n, int):
        n = safe_call(sf, "size")
    if not isinstance(n, int) or n <= 0:
        return np.array([], dtype=np.float64)

    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = safe_call(sf, "getValue", i)
        vals[i] = np.nan if v is None else float(v)
    return vals


def sampled_values(arr, limit=SAMPLE_LIMIT):
    n = arr.shape[0]
    if n <= limit:
        return arr
    idx = np.linspace(0, n - 1, num=limit, dtype=np.int64)
    return arr[idx]


def unique_counts(arr):
    if arr.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    if np.issubdtype(arr.dtype, np.floating):
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    vals, cnts = np.unique(arr, return_counts=True)
    order = np.argsort(cnts)[::-1]
    return vals[order], cnts[order]


def fmt_value(v):
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        if math.isfinite(float(v)) and float(v).is_integer():
            return str(int(v))
        return f"{float(v):.6g}"
    return str(v)


def walk_any(obj, label, parent_path, lines, stats, visited):
    if isinstance(obj, tuple):
        path = f"{parent_path}/{label}"
        lines.append(f"[tuple] {path} (len={len(obj)})")
        stats["entities"] += 1
        for i, item in enumerate(obj):
            walk_any(item, f"tuple[{i}]", path, lines, stats, visited)
        return

    if isinstance(obj, list):
        path = f"{parent_path}/{label}"
        lines.append(f"[list] {path} (len={len(obj)})")
        stats["entities"] += 1
        for i, item in enumerate(obj):
            walk_any(item, f"list[{i}]", path, lines, stats, visited)
        return

    if isinstance(obj, str):
        path = f"{parent_path}/{label}"
        lines.append(f"[str] {path}: {obj}")
        stats["entities"] += 1
        return

    if obj is None:
        path = f"{parent_path}/{label}"
        lines.append(f"[None] {path}")
        stats["entities"] += 1
        return

    obj_id = id(obj)
    if obj_id in visited:
        path = f"{parent_path}/{label}"
        lines.append(f"[ref] {path}: already visited")
        return
    visited.add(obj_id)

    nm = entity_name(obj, fallback=type(obj).__name__)
    cid = class_id(obj)
    tname = type(obj).__name__
    path = f"{parent_path}/{label}/{nm}"
    lines.append(f"[entity] {path} type={tname} class_id={cid}")
    stats["entities"] += 1

    if is_point_cloud_like(obj):
        stats["point_clouds"] += 1
        pc_points = get_point_count(obj)
        lines.append(f"  point_count: {pc_points}")

        sfs = get_scalar_fields(obj)
        sf_names = [name for _, name, _ in sfs]
        lines.append(f"  scalar_fields({len(sf_names)}): {sf_names}")

        for _, sf_name, sf in sfs:
            sf_name_l = sf_name.lower()
            is_keyword = any(k in sf_name_l for k in KEYWORDS)

            arr = scalar_to_numpy(sf)
            arr_sample = sampled_values(arr)
            vals, cnts = unique_counts(arr_sample)
            n_unique = int(vals.size)

            candidate = is_keyword or (n_unique <= 64)
            if candidate:
                stats["candidate_fields"].add(sf_name)
                lines.append(
                    f"    candidate_sf: {sf_name} | unique_in_sample={n_unique} | sample_size={arr_sample.size}"
                )
                top_pairs = list(zip(vals[:TOP_K], cnts[:TOP_K]))
                for v, c in top_pairs:
                    lines.append(f"      {fmt_value(v)}: {int(c)}")

    for child_label, child in get_children(obj):
        walk_any(child, child_label, path, lines, stats, visited)


def main():
    method_name, loaded, attempts = load_entities(INPUT_PATH)

    lines = []
    lines.append(f"Input: {INPUT_PATH}")
    lines.append(f"Load method used: {method_name}")
    if attempts:
        lines.append(f"Other load attempts: {attempts}")
    lines.append("")
    lines.append("=== Entity Tree Report ===")

    stats = {
        "entities": 0,
        "point_clouds": 0,
        "candidate_fields": set(),
    }
    visited = set()

    walk_any(loaded, "root", "", lines, stats, visited)

    lines.append("")
    lines.append("=== Summary ===")
    lines.append(f"entities_total: {stats['entities']}")
    lines.append(f"point_clouds_total: {stats['point_clouds']}")
    lines.append(
        "candidate_semantic_fields: "
        + (", ".join(sorted(stats["candidate_fields"])) if stats["candidate_fields"] else "<none>")
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written: {OUTPUT_PATH}")
    print(f"Entities: {stats['entities']}")
    print(f"Point clouds: {stats['point_clouds']}")
    print(
        "Candidate semantic fields: "
        + (", ".join(sorted(stats["candidate_fields"])) if stats["candidate_fields"] else "<none>")
    )


if __name__ == "__main__":
    main()
