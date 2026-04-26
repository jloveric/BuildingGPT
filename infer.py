
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

def normalize_mesh(vertices1,  bound=1):
    centroid = np.mean(np.vstack((vertices1))[:, 0:3], axis=0)
    vertices1 -= centroid
    max_distance = np.max(np.linalg.norm(np.vstack((vertices1)), axis=1))     
    vertices1 /= ((max_distance)/(bound))
    return vertices1

def fps(points, num_points):
    """ Farthest Point Sampling (FPS) """
    # points: [N, 3]
    # num_points: int, target number of points to sample
    N = points.shape[0]
    if N <= num_points:
        return points  # If the points are less than or equal to the target, return all points

    centroids = np.zeros((num_points, 3))  # To store the centroids (sampled points)
    distance = np.ones(N) * 1e10  # Initialize distance array

    # Start by selecting a random point
    idx = np.random.randint(0, N)
    centroids[0] = points[idx]
    distance[:] = np.linalg.norm(points - centroids[0], axis=1)

    for i in range(1, num_points):
        # Select the next point that is the farthest from the already chosen points
        farthest_idx = np.argmax(distance)
        centroids[i] = points[farthest_idx]

        # Update the distance array
        distance = np.minimum(distance, np.linalg.norm(points - centroids[i], axis=1))

    return centroids


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
        xyz_path_all = '/path/buildinggpt/data/pc'
        xyz_path = os.path.join(xyz_path_all, path+'.ply')
        pc = trimesh.load(xyz_path, process=False)  # process=False 避免对点云做额外处理
        points = pc.vertices  # (N, 3)
        num1 = 4096
        
        # 采样 self.opt.point_num 个点
        if len(points) >= num1:
            indices = np.random.choice(len(points), num1, replace=False)
        else:
            indices = np.random.choice(len(points), num1, replace=True)

        # num2 = 4096
        # if num1 < num2:
        #     extra_indices = np.random.choice(num1, num2 - num1, replace=True)
        #     indices = np.concatenate([indices, indices[extra_indices]])
        
        v = points[indices]
        #max_distance = np.max(np.linalg.norm(np.vstack(v), axis=1))  
        #v = v+ np.random.randn(*v.shape) * 0.05 * max_distance
        v = normalize_mesh(v)
                    
    
        cond = torch.from_numpy(v).unsqueeze(0).float().to(device) # [N, 3]

        
        # trimesh.PointCloud(v).export(f'{opt.workspace}/{name}_pc.obj')

        
        
    elif opt.cond_mode == 'none':
        cond = torch.zeros((1, 0), dtype=torch.float32, device=device) # [1, 0], dummy cond to get batch size

    for i in range(opt.test_repeat):

        t0 = time.time()
        filename = f'{name}_{i}'
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                model.generate(cond, num_faces=opt.test_num_face[0], max_new_tokens=opt.test_max_seq_length, tokenizer=tokenizer, clean=True, file_path = f'{opt.workspace}/{filename}.obj')
        
        # single batch

        # timing
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
