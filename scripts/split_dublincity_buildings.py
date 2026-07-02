#!/usr/bin/env python3
"""Split DublinCity CloudCompare .bin into per-building .ply files.

This script extracts building instances from DublinCity's entity hierarchy using
point-cloud names (e.g. ``building_01``, ``building_01_roof_01``,
``building_01_window_03``), not coarse scalar-field class IDs.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from plyfile import PlyElement, PlyData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CloudCompare .bin path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where cluster .ply files are written.",
    )
    parser.add_argument(
        "--max-buildings",
        "--max-clusters",
        dest="max_buildings",
        type=int,
        default=5,
        help="Maximum number of largest buildings to export.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=500,
        help="Minimum number of points in an exported cluster.",
    )
    parser.add_argument(
        "--prefer-parts",
        action="store_true",
        help=(
            "If both a full cloud 'building_<id>' and part clouds exist, "
            "merge parts instead of using the full cloud directly."
        ),
    )
    return parser.parse_args()


def cloudcompy_available() -> bool:
    return importlib.util.find_spec("cloudComPy") is not None


def _pick_first_attr(obj: object, names: Sequence[str]) -> object | None:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _call_first_method(obj: object, names: Sequence[str], *args) -> object | None:
    for name in names:
        if hasattr(obj, name):
            method = getattr(obj, name)
            if callable(method):
                return method(*args)
    return None


def _to_xyz_array(points_obj: object) -> np.ndarray:
    if isinstance(points_obj, np.ndarray):
        arr = points_obj
    else:
        arr = np.asarray(points_obj)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3].astype(np.float32, copy=False)
    if arr.dtype.names and {"x", "y", "z"}.issubset(set(arr.dtype.names)):
        return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32, copy=False)
    raise RuntimeError("Could not convert loaded point object to Nx3 XYZ array.")


def _entity_name(obj: object, fallback: str) -> str:
    name = _call_first_method(obj, ["getName", "name"])
    if isinstance(name, str) and name:
        return name
    return fallback


def _class_id(obj: object) -> int | None:
    cid = _call_first_method(obj, ["getClassID"])
    if isinstance(cid, int):
        return cid
    return None


def _is_point_cloud_like(obj: object, cc_module: object) -> bool:
    if obj is None:
        return False
    tname = type(obj).__name__.lower()
    if "pointcloud" in tname:
        return True
    if hasattr(obj, "getNumberOfScalarFields") and (hasattr(obj, "size") or hasattr(obj, "getNumberOfPoints")):
        return True
    cc_types = getattr(cc_module, "CC_TYPES", None)
    point_cloud_type = getattr(cc_types, "POINT_CLOUD", None)
    if point_cloud_type is not None and hasattr(obj, "isKindOf"):
        try:
            if obj.isKindOf(point_cloud_type):
                return True
        except Exception:
            pass
    cid = _class_id(obj)
    if point_cloud_type is not None and isinstance(cid, int) and cid == int(point_cloud_type):
        return True
    return False


def _iter_children(obj: object) -> Iterable[object]:
    num = _call_first_method(obj, ["getChildrenNumber"])
    if isinstance(num, int) and num > 0 and hasattr(obj, "getChild"):
        for idx in range(num):
            child = _call_first_method(obj, ["getChild"], idx)
            if child is not None:
                yield child


def _extract_xyz_from_cloud(cloud_obj: object) -> np.ndarray:
    xyz_candidate = _call_first_method(cloud_obj, ["toNpArrayCopy", "toNpArray", "points", "getPoints"])
    if xyz_candidate is None:
        inner = _call_first_method(cloud_obj, ["getAssociatedCloud", "getPointCloud"])
        if inner is not None:
            xyz_candidate = _call_first_method(inner, ["toNpArrayCopy", "toNpArray", "points", "getPoints"])
    if xyz_candidate is None:
        raise RuntimeError("Cloud object does not expose points through known methods.")
    return _to_xyz_array(xyz_candidate)


def load_point_cloud_entities(input_path: Path) -> List[Tuple[str, np.ndarray]]:
    cc = importlib.import_module("cloudComPy")

    import_fn = _pick_first_attr(cc, ["importFile", "ImportFile", "loadFile"])
    if not callable(import_fn):
        raise RuntimeError(
            "cloudComPy import succeeded but no known loader exists "
            "(expected importFile or loadFile)."
        )

    loaded = import_fn(str(input_path))
    if loaded is None:
        raise RuntimeError(f"Failed to load input file: {input_path}")

    out: List[Tuple[str, np.ndarray]] = []
    stack: List[Tuple[str, object]] = [("root", loaded)]
    visited: set[int] = set()

    while stack:
        node_name, node = stack.pop()
        if node is None:
            continue
        if isinstance(node, (list, tuple)):
            for idx, item in enumerate(node):
                stack.append((f"{node_name}[{idx}]", item))
            continue
        if isinstance(node, str):
            continue

        obj_id = id(node)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        name = _entity_name(node, fallback=node_name)
        if _is_point_cloud_like(node, cc):
            try:
                xyz = _extract_xyz_from_cloud(node)
            except Exception:
                xyz = np.empty((0, 3), dtype=np.float32)
            if xyz.shape[0] > 0:
                out.append((name, xyz))

        for child in _iter_children(node):
            stack.append((name, child))

    return out


BUILDING_RE = re.compile(r"^building_(\d+)(?:_(.+))?$", re.IGNORECASE)


def parse_building_cloud_name(name: str) -> Tuple[int, str]:
    match = BUILDING_RE.match(name.strip())
    if not match:
        raise ValueError(f"Not a building cloud name: {name}")
    building_id = int(match.group(1))
    suffix = (match.group(2) or "").lower()
    if not suffix:
        return building_id, "building"
    if "roof" in suffix:
        return building_id, "roof"
    if "facade" in suffix:
        return building_id, "facade"
    if "window" in suffix:
        return building_id, "window"
    if "door" in suffix:
        return building_id, "door"
    return building_id, "other_part"


def aggregate_buildings(
    named_clouds: Sequence[Tuple[str, np.ndarray]],
    prefer_parts: bool,
) -> List[Tuple[int, np.ndarray, Dict[str, int]]]:
    grouped: Dict[int, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    for name, xyz in named_clouds:
        try:
            building_id, role = parse_building_cloud_name(name)
        except ValueError:
            continue
        grouped[building_id][role].append(xyz)

    merged: List[Tuple[int, np.ndarray, Dict[str, int]]] = []
    for building_id, role_map in grouped.items():
        base_clouds = role_map.get("building", [])
        part_roles = [role for role in role_map.keys() if role != "building"]
        part_clouds: List[np.ndarray] = []
        for role in part_roles:
            part_clouds.extend(role_map[role])

        if base_clouds and (not prefer_parts or not part_clouds):
            xyz = base_clouds[0]
        else:
            if not part_clouds and base_clouds:
                xyz = base_clouds[0]
            elif part_clouds:
                xyz = np.concatenate(part_clouds, axis=0)
            else:
                continue

        role_counts = {role: len(arrs) for role, arrs in role_map.items()}
        merged.append((building_id, xyz, role_counts))

    merged.sort(key=lambda item: item[1].shape[0], reverse=True)
    return merged


def write_ply_xyz(output_path: Path, xyz: np.ndarray) -> None:
    verts = np.empty(xyz.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts["x"] = xyz[:, 0]
    verts["y"] = xyz[:, 1]
    verts["z"] = xyz[:, 2]
    ply = PlyData([PlyElement.describe(verts, "vertex")], text=True)
    ply.write(str(output_path))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input.suffix.lower() != ".bin":
        raise RuntimeError("This script only accepts CloudCompare .bin input.")
    if not cloudcompy_available():
        raise RuntimeError(
            "cloudComPy is not available in this environment. "
            "Install cloudComPy in the same uv/python interpreter used to run this script."
        )
    named_clouds = load_point_cloud_entities(args.input)
    building_clouds = aggregate_buildings(named_clouds, prefer_parts=args.prefer_parts)

    exported = 0
    for rank, (building_id, cluster_xyz, role_counts) in enumerate(building_clouds, start=1):
        if cluster_xyz.shape[0] < args.min_points:
            continue
        out_path = args.output_dir / f"building_{building_id:03d}_{cluster_xyz.shape[0]}pts.ply"
        write_ply_xyz(out_path, cluster_xyz)
        exported += 1
        print(
            f"[exported] {out_path} ({cluster_xyz.shape[0]} points) "
            f"roles={role_counts}"
        )
        if exported >= args.max_buildings:
            break

    total_points = int(sum(xyz.shape[0] for _, xyz in named_clouds))
    print(f"[info] loaded point-cloud entities: {len(named_clouds)}")
    print(f"[info] total points across entities: {total_points}")
    print(f"[info] parsed building groups: {len(building_clouds)}")
    print(f"[info] exported buildings: {exported}")
    if exported == 0:
        top_sizes = [xyz.shape[0] for _, xyz, _ in building_clouds[:10]]
        print(f"[warn] no buildings exported; top building sizes: {top_sizes}")
        print(
            "[hint] Lower --min-points. "
            "If parsing missed labels, inspect entity names with dublincity_entity_report.py."
        )


if __name__ == "__main__":
    main()
