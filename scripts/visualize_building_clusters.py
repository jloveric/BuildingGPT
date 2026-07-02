#!/usr/bin/env python3
"""Render clustered .ply files into per-file PNG previews."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from plyfile import PlyData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing cluster_*.ply files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-file PNG outputs (one image per .ply).",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=6,
        help="Render the first K cluster files sorted by filename.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.3,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=45.0,
        help="Camera elevation angle in degrees (45 = oblique).",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-45.0,
        help="Matplotlib camera azimuth.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1200,
        help="Output image width/height for Pillow fallback renderer.",
    )
    return parser.parse_args()


def load_ply_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    verts = ply["vertex"].data
    return np.column_stack([verts["x"], verts["y"], verts["z"]]).astype(np.float32, copy=False)


def normalize_points(xyz: np.ndarray) -> np.ndarray:
    center = xyz.mean(axis=0, keepdims=True)
    shifted = xyz - center
    scale = np.max(np.linalg.norm(shifted, axis=1))
    if scale == 0:
        return shifted
    return shifted / scale


def collect_cluster_paths(input_dir: Path, max_clusters: int) -> List[Path]:
    candidates = sorted(input_dir.glob("cluster_*.ply"))
    if not candidates:
        candidates = sorted(input_dir.glob("*.ply"))
    return candidates[:max_clusters]


def _project_to_tile(xyz: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    xy = xyz[:, :2]
    z = xyz[:, 2]

    mins = np.min(xy, axis=0)
    maxs = np.max(xy, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    uv = (xy - mins) / span
    px = np.clip((uv[:, 0] * (tile_w - 1)).astype(np.int32), 0, tile_w - 1)
    py = np.clip(((1.0 - uv[:, 1]) * (tile_h - 1)).astype(np.int32), 0, tile_h - 1)

    zmin, zmax = float(np.min(z)), float(np.max(z))
    zspan = max(zmax - zmin, 1e-6)
    zn = (z - zmin) / zspan

    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    color_r = (40 + 215 * zn).astype(np.uint8)
    color_g = (130 + 110 * (1.0 - zn)).astype(np.uint8)
    color_b = (255 * (1.0 - zn)).astype(np.uint8)
    tile[py, px, 0] = color_r
    tile[py, px, 1] = color_g
    tile[py, px, 2] = color_b
    return tile


def render_pillow_oblique_single(
    xyz: np.ndarray,
    output_png: Path,
    azim_deg: float,
    elev_deg: float,
    image_size: int,
) -> None:
    center = xyz.mean(axis=0, keepdims=True)
    pts = xyz - center
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale

    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)

    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_el, sin_el = np.cos(el), np.sin(el)

    rz = np.array(
        [[cos_az, -sin_az, 0.0], [sin_az, cos_az, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_el, -sin_el], [0.0, sin_el, cos_el]],
        dtype=np.float64,
    )
    cam = pts @ rz.T @ rx.T

    u = cam[:, 0]
    v = cam[:, 1]
    depth = cam[:, 2]

    u_min, u_max = float(np.min(u)), float(np.max(u))
    v_min, v_max = float(np.min(v)), float(np.max(v))
    span_u = max(u_max - u_min, 1e-6)
    span_v = max(v_max - v_min, 1e-6)
    margin = 24
    w = image_size
    h = image_size
    px = ((u - u_min) / span_u * (w - 1 - 2 * margin) + margin).astype(np.int32)
    py = ((1.0 - (v - v_min) / span_v) * (h - 1 - 2 * margin) + margin).astype(np.int32)

    z = xyz[:, 2]
    z_min, z_max = float(np.min(z)), float(np.max(z))
    z_span = max(z_max - z_min, 1e-6)
    zn = (z - z_min) / z_span
    color_r = (40 + 215 * zn).astype(np.uint8)
    color_g = (120 + 110 * (1.0 - zn)).astype(np.uint8)
    color_b = (255 * (1.0 - zn)).astype(np.uint8)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    order = np.argsort(depth)  # draw far points first
    for i in order:
        x = px[i]
        y = py[i]
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            img[y - 1 : y + 2, x - 1 : x + 2, 0] = color_r[i]
            img[y - 1 : y + 2, x - 1 : x + 2, 1] = color_g[i]
            img[y - 1 : y + 2, x - 1 : x + 2, 2] = color_b[i]

    output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(output_png)
    print(f"[saved] {output_png} (Pillow oblique renderer)")


def main() -> None:
    args = parse_args()
    cluster_paths = collect_cluster_paths(args.input_dir, args.max_clusters)
    if not cluster_paths:
        raise RuntimeError(f"No PLY files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if importlib.util.find_spec("matplotlib") is None:
        for path in cluster_paths:
            xyz = load_ply_xyz(path)
            out = args.output_dir / f"{path.stem}.png"
            render_pillow_oblique_single(
                xyz=xyz,
                output_png=out,
                azim_deg=args.azim,
                elev_deg=args.elev,
                image_size=args.image_size,
            )
        print("[warn] matplotlib not found; generated oblique PNG fallback.")
        print("[hint] For interactive 3D viewing, open the .ply files in CloudCompare or MeshLab:")
        for path in cluster_paths:
            print(f"       {path}")
        return

    import matplotlib.pyplot as plt

    for path in cluster_paths:
        fig = plt.figure(figsize=(7.0, 6.5), dpi=160)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        xyz = load_ply_xyz(path)
        vis_xyz = normalize_points(xyz)
        ax.scatter(
            vis_xyz[:, 0],
            vis_xyz[:, 1],
            vis_xyz[:, 2],
            s=args.point_size,
            c=vis_xyz[:, 2],
            cmap="viridis",
            linewidths=0,
        )
        ax.set_title(f"{path.name}\n{xyz.shape[0]} pts", fontsize=8)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_axis_off()
        fig.tight_layout()
        out = args.output_dir / f"{path.stem}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
