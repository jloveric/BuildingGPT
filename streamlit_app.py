#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import trimesh
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/data2/point-cloud-datasets/MunichWF/pc_part")
DEFAULT_WORKSPACE = REPO_ROOT / "workspace"
DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained" / "ArAE.safetensors"


def normalize_points(points: np.ndarray, bound: float = 0.85) -> np.ndarray:
    points = points.astype(np.float32, copy=True)
    centroid = points.mean(axis=0)
    points -= centroid
    max_distance = np.linalg.norm(points, axis=1).max()
    if max_distance <= 1e-8:
        return points
    points /= max_distance
    points *= bound
    return points


def read_obj_vertices_edges(path: Path) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    vertices: List[List[float]] = []
    edges: set[Tuple[int, int]] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *_ = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
                continue

            if line.startswith("l "):
                parts = line.strip().split()[1:]
                ids: List[int] = []
                for token in parts:
                    ids.append(int(token.split("/")[0]) - 1)
                for i in range(len(ids) - 1):
                    a, b = ids[i], ids[i + 1]
                    if a != b:
                        edges.add((a, b) if a < b else (b, a))
                continue

            if line.startswith("f "):
                parts = line.strip().split()[1:]
                ids = [int(token.split("/")[0]) - 1 for token in parts]
                for i in range(len(ids)):
                    a, b = ids[i], ids[(i + 1) % len(ids)]
                    if a != b:
                        edges.add((a, b) if a < b else (b, a))

    return np.asarray(vertices, dtype=np.float32), sorted(edges)


def downsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def load_ply_points(path: Path, downsample_n: int, seed: int) -> np.ndarray:
    cloud = trimesh.load(path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float32)
    points = downsample_points(points, downsample_n, seed)
    return normalize_points(points)


@st.cache_data(show_spinner=False)
def list_point_clouds(data_dir: str) -> List[str]:
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("*.ply"))


def build_deck(
    points_xyz: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    output_vertices: np.ndarray,
    point_size: float,
    output_point_size: float,
    edge_width: float,
) -> pdk.Deck:
    point_rows = pd.DataFrame(
        {
            "position": points_xyz.tolist(),
            "color": [[60, 160, 240, 180]] * points_xyz.shape[0],
        }
    )

    edge_paths: List[List[List[float]]] = []
    for a, b in edges:
        if a < output_vertices.shape[0] and b < output_vertices.shape[0]:
            edge_paths.append([output_vertices[a].tolist(), output_vertices[b].tolist()])
    edge_rows = pd.DataFrame({"path": edge_paths, "color": [[240, 90, 50, 230]] * len(edge_paths)})
    output_rows = pd.DataFrame(
        {
            "position": output_vertices.tolist(),
            "color": [[255, 160, 20, 230]] * output_vertices.shape[0],
        }
    )

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=point_rows,
        get_position="position",
        radius_units="pixels",
        get_radius=point_size,
        get_color="color",
        pickable=True,
    )
    edge_rows_for_line = pd.DataFrame(
        {
            "source": [p[0] for p in edge_paths],
            "target": [p[1] for p in edge_paths],
            "color": [[240, 90, 50, 230]] * len(edge_paths),
        }
    )
    edge_layer = pdk.Layer(
        "LineLayer",
        data=edge_rows_for_line,
        get_source_position="source",
        get_target_position="target",
        get_color="color",
        width_units="pixels",
        get_width=edge_width,
        pickable=False,
    )
    output_vertex_layer = pdk.Layer(
        "ScatterplotLayer",
        data=output_rows,
        get_position="position",
        radius_units="pixels",
        get_radius=output_point_size,
        get_color="color",
        pickable=True,
    )

    if points_xyz.shape[0] > 0:
        target = points_xyz.mean(axis=0).tolist()
    else:
        target = [0.0, 0.0, 0.0]

    view_state = pdk.ViewState(target=target, zoom=4, rotation_x=35, rotation_orbit=35)
    return pdk.Deck(
        layers=[point_layer, edge_layer, output_vertex_layer],
        initial_view_state=view_state,
        views=[pdk.View(type="OrbitView", controller=True)],
        map_provider=None,
    )


def render_oblique_preview(
    points_xyz: np.ndarray,
    output_vertices: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    image_size: int = 900,
    azim_deg: float = 45.0,
    elev_deg: float = 35.0,
) -> Image.Image:
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)

    rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(el), -np.sin(el)],
            [0.0, np.sin(el), np.cos(el)],
        ],
        dtype=np.float32,
    )
    rot = rx @ rz

    pts_all = np.concatenate([points_xyz, output_vertices], axis=0)
    rot_all = pts_all @ rot.T
    xy = rot_all[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)

    pad = 20.0
    scale = min((image_size - 2 * pad) / span[0], (image_size - 2 * pad) / span[1])
    xy_img = (xy - min_xy) * scale + pad
    xy_img[:, 1] = image_size - xy_img[:, 1]

    in_n = points_xyz.shape[0]
    in_xy = xy_img[:in_n]
    out_xy = xy_img[in_n:]

    img = Image.new("RGB", (image_size, image_size), color=(12, 14, 18))
    draw = ImageDraw.Draw(img)

    for x, y in in_xy:
        draw.point((float(x), float(y)), fill=(80, 170, 255))

    for a, b in edges:
        if a < out_xy.shape[0] and b < out_xy.shape[0]:
            xa, ya = out_xy[a]
            xb, yb = out_xy[b]
            draw.line((float(xa), float(ya), float(xb), float(yb)), fill=(255, 160, 50), width=2)

    for x, y in out_xy:
        r = 2
        draw.ellipse((float(x - r), float(y - r), float(x + r), float(y + r)), fill=(255, 205, 80))

    return img


def run_inference(
    point_cloud_path: Path,
    workspace: Path,
    checkpoint: Path,
    seed: int,
    test_num_face: int,
    filter_interior_points: bool,
) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="streamlit_input_",
        dir=REPO_ROOT,
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(str(point_cloud_path) + "\n")
        test_path = Path(f.name)

    cmd = [
        "uv",
        "run",
        "python",
        "infer.py",
        "ArAE",
        "--workspace",
        str(workspace),
        "--resume",
        str(checkpoint),
        "--test_path",
        str(test_path),
        "--generate_mode",
        "sample",
        "--test_num_face",
        str(test_num_face),
        "--test_repeat",
        "1",
        "--seed",
        str(seed),
    ]
    if filter_interior_points:
        cmd.append("--filter-interior-points")

    try:
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        test_path.unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(page_title="BuildingGPT Pipeline", layout="wide")
    st.title("BuildingGPT Point Cloud Inference")

    with st.sidebar:
        st.header("Pipeline Settings")
        data_dir = Path(
            st.text_input("Point cloud folder", value=str(DEFAULT_DATA_DIR))
        )
        workspace = Path(
            st.text_input("Workspace", value=str(DEFAULT_WORKSPACE))
        )
        checkpoint = Path(
            st.text_input("Checkpoint", value=str(DEFAULT_CHECKPOINT))
        )
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)
        test_num_face = st.number_input("test_num_face", min_value=100, max_value=8000, value=1000, step=100)
        input_preview_points = st.number_input(
            "Input downsample points",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
        )
        filter_interior = st.checkbox("Use --filter-interior-points", value=False)
        point_size = st.slider("Point size", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
        output_point_size = st.slider("Output vertex size", min_value=1.0, max_value=8.0, value=4.0, step=0.5)
        edge_width = st.slider("Edge width", min_value=1.0, max_value=8.0, value=2.0, step=0.5)

    clouds = list_point_clouds(str(data_dir))
    if not clouds:
        st.error(f"No .ply files found in `{data_dir}`.")
        st.stop()

    selected = st.selectbox(
        "Select MunichWF point cloud",
        options=clouds,
        format_func=lambda p: Path(p).name,
    )
    selected_path = Path(selected)
    name = selected_path.stem

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.write(f"Selected: `{selected_path}`")
    with col_b:
        run_clicked = st.button("Run pipeline", type="primary")

    result_obj = workspace / f"{name}_seed{seed}_0.obj"
    input_obj = workspace / f"{name}_input_seed{seed}.obj"

    if run_clicked:
        with st.spinner("Running infer.py... this can take a while on large point clouds"):
            result = run_inference(
                point_cloud_path=selected_path,
                workspace=workspace,
                checkpoint=checkpoint,
                seed=int(seed),
                test_num_face=int(test_num_face),
                filter_interior_points=filter_interior,
            )
        st.subheader("Pipeline logs")
        st.code((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""), language="bash")
        if result.returncode != 0:
            st.error(f"infer.py failed with exit code {result.returncode}")
            st.stop()

    if not result_obj.exists():
        st.info("Run the pipeline to generate an output OBJ.")
        st.stop()

    if input_obj.exists():
        points_xyz, _ = read_obj_vertices_edges(input_obj)
    else:
        st.warning(
            "Normalized input OBJ not found in workspace; using a normalized preview from the raw PLY."
        )
        points_xyz = load_ply_points(selected_path, int(input_preview_points), int(seed))

    if points_xyz.shape[0] > int(input_preview_points):
        points_xyz = downsample_points(points_xyz, int(input_preview_points), int(seed))

    output_vertices, output_edges = read_obj_vertices_edges(result_obj)

    st.subheader("Overlay: Input Point Cloud + Model Output")
    st.caption(
        "Blue = normalized input point cloud, Orange = generated wireframe edges/vertices. "
        "The app uses normalized coordinates to align both geometries."
    )
    st.write(
        f"Input points shown: `{points_xyz.shape[0]}` | "
        f"Output vertices: `{output_vertices.shape[0]}` | "
        f"Output edges: `{len(output_edges)}`"
    )
    deck = build_deck(
        points_xyz=points_xyz,
        edges=output_edges,
        output_vertices=output_vertices,
        point_size=float(point_size),
        output_point_size=float(output_point_size),
        edge_width=float(edge_width),
    )
    st.pydeck_chart(deck, use_container_width=True)

    st.subheader("Fallback Preview")
    st.caption("Static oblique render of the same overlay (useful if WebGL rendering fails).")
    preview = render_oblique_preview(points_xyz, output_vertices, output_edges)
    st.image(preview, use_container_width=True)

    st.markdown(
        f"""
        **Output files**
        - Input (normalized): `{input_obj}`
        - Prediction: `{result_obj}`
        """
    )


if __name__ == "__main__":
    main()
