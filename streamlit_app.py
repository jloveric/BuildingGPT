#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import trimesh
import streamlit.components.v1 as components
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


def build_threejs_html(
    points_xyz: np.ndarray,
    output_vertices: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    height: int = 800,
) -> str:
    points_json = json.dumps(points_xyz.tolist())
    vertices_json = json.dumps(output_vertices.tolist())
    edges_json = json.dumps([[int(a), int(b)] for a, b in edges])

    template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #0c0e12;
      font-family: sans-serif;
    }}
    #wrap {{
      position: relative;
      width: 100%;
      height: __HEIGHT__px;
    }}
    #hud {{
      position: absolute;
      left: 12px;
      top: 12px;
      z-index: 10;
      color: #dbe8ff;
      background: rgba(7, 10, 18, 0.75);
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.35;
      pointer-events: none;
    }}
    canvas {{
      display: block;
    }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="hud">Loading Three.js viewer...</div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
  <script>
    const points = __POINTS__;
    const outputVertices = __VERTICES__;
    const edges = __EDGES__;

    const wrap = document.getElementById('wrap');
    const hud = document.getElementById('hud');

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0e12);

    const camera = new THREE.PerspectiveCamera(50, wrap.clientWidth / wrap.clientHeight, 0.001, 1000);
    camera.position.set(1.8, 1.8, 1.8);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(wrap.clientWidth, wrap.clientHeight);
    wrap.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    function makePointsGeometry(array, colorHex) {{
      const geometry = new THREE.BufferGeometry();
      const flat = new Float32Array(array.length * 3);
      for (let i = 0; i < array.length; i++) {{
        flat[i * 3 + 0] = array[i][0];
        flat[i * 3 + 1] = array[i][1];
        flat[i * 3 + 2] = array[i][2];
      }}
      geometry.setAttribute('position', new THREE.BufferAttribute(flat, 3));
      const material = new THREE.PointsMaterial({ size: 0.012, color: colorHex, sizeAttenuation: true });
      return new THREE.Points(geometry, material);
    }}

    const inputPoints = makePointsGeometry(points, 0x4fa3ff);
    scene.add(inputPoints);

    const outPointCloud = makePointsGeometry(outputVertices, 0xffb14a);
    outPointCloud.material.size = 0.016;
    scene.add(outPointCloud);

    const linePositions = [];
    for (const [a, b] of edges) {{
      if (a < outputVertices.length && b < outputVertices.length) {{
        const va = outputVertices[a];
        const vb = outputVertices[b];
        linePositions.push(va[0], va[1], va[2], vb[0], vb[1], vb[2]);
      }}
    }}
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(linePositions), 3));
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xff8c3a });
    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);

    const box = new THREE.Box3().setFromObject(scene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
    const dist = maxDim * 2.2;
    camera.position.copy(center.clone().add(new THREE.Vector3(dist, dist, dist)));
    controls.target.copy(center);
    controls.update();

    hud.innerHTML = `Input points: ${points.length}<br>Output vertices: ${outputVertices.length}<br>Edges: ${edges.length}<br>Drag to orbit, scroll to zoom`;

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener('resize', () => {{
      camera.aspect = wrap.clientWidth / wrap.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(wrap.clientWidth, wrap.clientHeight);
    }});
  </script>
</body>
</html>
"""
    return (
        template.replace("{{", "{")
        .replace("}}", "}")
        .replace("__HEIGHT__", str(height))
        .replace("__POINTS__", points_json)
        .replace("__VERTICES__", vertices_json)
        .replace("__EDGES__", edges_json)
    )


def build_canvas_viewer_html(
    points_xyz: np.ndarray,
    output_vertices: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    height: int = 800,
) -> str:
    points_json = json.dumps(points_xyz.tolist())
    vertices_json = json.dumps(output_vertices.tolist())
    edges_json = json.dumps([[int(a), int(b)] for a, b in edges])

    template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #0c0e12;
      font-family: sans-serif;
    }
    #wrap {
      position: relative;
      width: 100%;
      height: __HEIGHT__px;
      background: #0c0e12;
    }
    #hud {
      position: absolute;
      left: 12px;
      top: 12px;
      z-index: 10;
      color: #dbe8ff;
      background: rgba(7, 10, 18, 0.78);
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.35;
      pointer-events: none;
    }
    canvas {
      display: block;
      cursor: grab;
    }
    canvas:active {
      cursor: grabbing;
    }
  </style>
</head>
<body>
  <div id="wrap">
    <canvas id="viewer"></canvas>
    <div id="hud"></div>
  </div>
  <script>
    const points = __POINTS__;
    const outputVertices = __VERTICES__;
    const edges = __EDGES__;

    const wrap = document.getElementById("wrap");
    const canvas = document.getElementById("viewer");
    const ctx = canvas.getContext("2d");
    const hud = document.getElementById("hud");

    let az = Math.PI / 4;
    let el = Math.PI / 5;
    let zoom = 1.0;
    let isDragging = false;
    let lastX = 0;
    let lastY = 0;

    const all = points.concat(outputVertices);
    const center = [0, 0, 0];
    for (const p of all) {
      center[0] += p[0];
      center[1] += p[1];
      center[2] += p[2];
    }
    center[0] /= Math.max(all.length, 1);
    center[1] /= Math.max(all.length, 1);
    center[2] /= Math.max(all.length, 1);

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(wrap.clientWidth * dpr);
      canvas.height = Math.floor(wrap.clientHeight * dpr);
      canvas.style.width = wrap.clientWidth + "px";
      canvas.style.height = wrap.clientHeight + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function project(p) {
      let x = p[0] - center[0];
      let y = p[1] - center[1];
      let z = p[2] - center[2];

      const caz = Math.cos(az), saz = Math.sin(az);
      const x1 = x * caz - y * saz;
      const y1 = x * saz + y * caz;
      const z1 = z;

      const cel = Math.cos(el), sel = Math.sin(el);
      const y2 = y1 * cel - z1 * sel;
      const z2 = y1 * sel + z1 * cel;

      const scale = Math.min(wrap.clientWidth, wrap.clientHeight) * 0.42 * zoom;
      return [
        wrap.clientWidth / 2 + x1 * scale,
        wrap.clientHeight / 2 - y2 * scale,
        z2,
      ];
    }

    function draw() {
      ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);
      ctx.fillStyle = "#0c0e12";
      ctx.fillRect(0, 0, wrap.clientWidth, wrap.clientHeight);

      const projectedPoints = points.map(project);
      const projectedOut = outputVertices.map(project);

      ctx.fillStyle = "rgba(80, 170, 255, 0.72)";
      for (const p of projectedPoints) {
        ctx.fillRect(p[0], p[1], 1.4, 1.4);
      }

      ctx.strokeStyle = "rgba(255, 145, 55, 0.95)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const e of edges) {
        const a = e[0], b = e[1];
        if (a < projectedOut.length && b < projectedOut.length) {
          const pa = projectedOut[a];
          const pb = projectedOut[b];
          ctx.moveTo(pa[0], pa[1]);
          ctx.lineTo(pb[0], pb[1]);
        }
      }
      ctx.stroke();

      ctx.fillStyle = "rgba(255, 210, 90, 0.95)";
      for (const p of projectedOut) {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 2.8, 0, Math.PI * 2);
        ctx.fill();
      }

      hud.innerHTML =
        `Offline dynamic viewer<br>` +
        `Input points: ${points.length}<br>` +
        `Output vertices: ${outputVertices.length}<br>` +
        `Edges: ${edges.length}<br>` +
        `Drag to rotate, wheel to zoom`;
    }

    canvas.addEventListener("mousedown", (ev) => {
      isDragging = true;
      lastX = ev.clientX;
      lastY = ev.clientY;
    });
    window.addEventListener("mouseup", () => {
      isDragging = false;
    });
    window.addEventListener("mousemove", (ev) => {
      if (!isDragging) return;
      const dx = ev.clientX - lastX;
      const dy = ev.clientY - lastY;
      lastX = ev.clientX;
      lastY = ev.clientY;
      az += dx * 0.01;
      el = Math.max(-1.45, Math.min(1.45, el + dy * 0.01));
      draw();
    });
    canvas.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      zoom *= ev.deltaY < 0 ? 1.08 : 0.92;
      zoom = Math.max(0.15, Math.min(12.0, zoom));
      draw();
    }, { passive: false });

    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""
    return (
        template.replace("__HEIGHT__", str(height))
        .replace("__POINTS__", points_json)
        .replace("__VERTICES__", vertices_json)
        .replace("__EDGES__", edges_json)
    )


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
    tabs = st.tabs(["Dynamic viewer", "Current viewer", "Three.js viewer", "Fallback preview"])
    with tabs[0]:
        st.caption("Offline browser viewer: no CDN or WebGL library dependency. Drag to rotate, scroll to zoom.")
        components.html(
            build_canvas_viewer_html(points_xyz, output_vertices, output_edges),
            height=860,
            scrolling=False,
        )
    with tabs[1]:
        deck = build_deck(
            points_xyz=points_xyz,
            edges=output_edges,
            output_vertices=output_vertices,
            point_size=float(point_size),
            output_point_size=float(output_point_size),
            edge_width=float(edge_width),
        )
        st.pydeck_chart(deck, use_container_width=True)
    with tabs[2]:
        st.caption("Interactive Three.js orbit viewer embedded directly in Streamlit.")
        components.html(
            build_threejs_html(points_xyz, output_vertices, output_edges),
            height=860,
            scrolling=False,
        )
    with tabs[3]:
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
