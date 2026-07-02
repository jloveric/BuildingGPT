
'''
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import os
import tyro
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

import kiui
import trimesh
from kiui.op import recenter
from kiui.mesh_utils import clean_mesh

from core.options import AllConfigs, Options
from core.models import LMM
from core.utils import load_mesh, get_tokenizer
from core.utils import monkey_patch_transformers
from core.provider import tokenize_mesh
from core.pointcloud_filter import filter_interior_points

def load_wireframe(wireframe_file):
    vertices = []
    edges = set()
    num = 0
    with open(wireframe_file) as f:
        for lines in f.readlines():
            line = lines.strip().split(' ')
            if line[0] == 'v':
                vertices.append(line[1:])
            else:
                if line[0] == '#':
                    continue
                obj_data = np.array(line[1:], dtype=np.int32).reshape(2) - 1
                edges.add(tuple(sorted(obj_data)))
    vertices = np.array(vertices, dtype=np.float64)
    edges = np.array(list(edges))
    return vertices, edges

def normalize_points(points, bound=0.85):
    points = points.copy()
    centroid = points.mean(axis=0)
    points -= centroid
    max_distance = np.linalg.norm(points, axis=1).max()
    points /= max_distance
    points *= bound
    return points


def resolve_point_cloud_path(path: str) -> str:
    if os.path.isfile(path):
        return path

    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(repo_root, path),
        os.path.join(repo_root, f'{path}.ply'),
        os.path.join(repo_root, 'data', 'pc', path),
        os.path.join(repo_root, 'data', 'pc', f'{path}.ply'),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Point cloud not found for {path!r}. "
        "Pass a .ply file path or a basename under data/pc/."
    )


def sample_points_fast(points, num_points, device, pre_pool_size=200000):
    # For huge point clouds, first randomly reduce to a manageable pool.
    if len(points) > pre_pool_size:
        pool_idx = np.random.choice(len(points), pre_pool_size, replace=False)
        pool = points[pool_idx]
    else:
        pool = points

    if len(pool) <= num_points:
        sample_idx = np.random.choice(len(pool), num_points, replace=True)
        return pool[sample_idx]

    if device.type == 'cuda':
        import torch_cluster

        pc = torch.from_numpy(pool).float().to(device)
        batch_idx = torch.zeros(pc.shape[0], dtype=torch.long, device=device)
        ratio = num_points / pc.shape[0]
        fps_idx = torch_cluster.fps(pc, batch_idx, ratio=ratio)
        if fps_idx.shape[0] < num_points:
            extra = torch.randint(0, pc.shape[0], (num_points - fps_idx.shape[0],), device=device)
            fps_idx = torch.cat([fps_idx, extra], dim=0)
        fps_idx = fps_idx[:num_points]
        return pc[fps_idx].detach().cpu().numpy()

    # CPU fallback: random sample to avoid very slow O(N*K) Python FPS.
    sample_idx = np.random.choice(len(pool), num_points, replace=False)
    return pool[sample_idx]


monkey_patch_transformers()

opt = tyro.cli(AllConfigs)

kiui.seed_everything(opt.seed)

# model
model = LMM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sane?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().eval().to(device)

# load rembg
if opt.cond_mode == 'image':
    import rembg
    bg_remover = rembg.new_session()

# tokenizer
tokenizer, _ = get_tokenizer(opt)

def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(opt.workspace, exist_ok=True)

    if opt.cond_mode == 'image':
        input_image = kiui.read_image(path, mode='uint8')

        # bg removal
        carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
        mask = carved_image[..., -1] > 0
        image = recenter(carved_image, mask, border_ratio=0.2)
        image = image.astype(np.float32) / 255.0
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        kiui.write_image(os.path.join(opt.workspace, name + '.jpg'), image)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float().to(device)
        cond = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False) # match training data and DINO.

    elif opt.cond_mode == 'point':
        xyz_path = resolve_point_cloud_path(path)
        print(f'[INFO] Loading point cloud: {xyz_path}')
        pc = trimesh.load(xyz_path, process=False)  # process=False 避免对点云做额外处理
        points = pc.vertices  # (N, 3)
        print(f'[INFO] Point cloud size: {len(points)}')
        if opt.filter_interior_points:
            t_filter0 = time.time()
            points = filter_interior_points(
                points,
                num_views=opt.filter_num_views,
                azimuth_bins=opt.filter_azimuth_bins,
                elevation_bins=opt.filter_elevation_bins,
                view_radius_scale=opt.filter_view_radius_scale,
                max_points_for_filter=opt.filter_max_points,
            )
            t_filter1 = time.time()
            print(
                f'[INFO] Interior filter kept {len(points)} points '
                f'in {(t_filter1 - t_filter0):.2f}s'
            )
        v = sample_points_fast(points, opt.point_num, device=device)

        # Match eval preprocessing in GithubDataset: center, unit ball, scale to 0.85.
        v = normalize_points(v, bound=0.85)
        cond = torch.from_numpy(v).unsqueeze(0).float().to(device) # [N, 3]

        debug_pc_path = os.path.join(opt.workspace, f'{name}_input_seed{opt.seed}.obj')
        with open(debug_pc_path, 'w') as f:
            for vertex in v:
                f.write(f'v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')
        print(f'[INFO] Saved normalized input point cloud to {debug_pc_path}')
    elif opt.cond_mode == 'none':
        cond = torch.zeros((1, 0), dtype=torch.float32, device=device) # [1, 0], dummy cond to get batch size

    for i in range(opt.test_repeat):

        t0 = time.time()
        filename = f'{name}_seed{opt.seed}_{i}'
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                model.generate(cond, num_faces=opt.test_num_face[0], max_new_tokens=opt.test_max_seq_length, tokenizer=tokenizer, clean=True, file_path = f'{opt.workspace}/{filename}.obj')
        
        # single batch

        # timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        print(f'[INFO] Processing {path} --> {filename}.obj, time = {t1 - t0:.4f}s')
    

assert opt.test_path is not None

file_paths = []
with open(opt.test_path, 'r') as f:
    for line in f:
        # 去掉换行符和多余的空格，并确保只添加非空行
        line = line.strip()
        
        file_paths.append(line)

# 对每个数字进行处理
for path in file_paths:
    process(opt, path)
