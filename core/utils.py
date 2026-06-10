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

import torch
import numpy as np
import trimesh
import megfile
from core.options import Options
import logging

def init_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # write to file
    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # print to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

def load_mesh(path):
    # path: string for local/s3 path
    if path.startswith('s3'):
        ext = path.split('.')[-1]
        with megfile.smart_open(path, 'rb') as f:
            _data = trimesh.load(file_obj=trimesh.util.wrap_as_stream(f.read()), file_type=ext)
    else:
        _data = trimesh.load(path)

    # always convert scene to mesh, and apply all transforms...
    if isinstance(_data, trimesh.Scene):
        # print(f"[INFO] load trimesh: concatenating {len(_data.geometry)} meshes.")
        _concat = []
        # loop the scene graph and apply transform to each mesh
        scene_graph = _data.graph.to_flattened() # dict {name: {transform: 4x4 mat, geometry: str}}
        for k, v in scene_graph.items():
            name = v['geometry']
            if name in _data.geometry and isinstance(_data.geometry[name], trimesh.Trimesh):
                transform = v['transform']
                _concat.append(_data.geometry[name].apply_transform(transform))
        _mesh = trimesh.util.concatenate(_concat)
    else:
        _mesh = _data
    
    vertices = _mesh.vertices
    faces = _mesh.faces

    return vertices, faces


def normalize_mesh(vertices1, v1, bound=0.95):

    centroid = np.mean(np.vstack((vertices1,v1))[:, 0:3], axis=0)
    vertices1 -= centroid
    v1 -= centroid
 
    
    max_distance = np.max(np.linalg.norm(np.vstack((vertices1, v1)), axis=1))     
    a = np.random.uniform(0, 1)
    vertices1 /= (max_distance)
    v1 /= (max_distance)
    
    
    if a < 0.4:
        vertices1 *= ((bound))
        
        v1 *= (bound)
        
    else:
        if a < 0.6:
            vertices1[:,0] *= ((bound))
            
            v1[:,0] *= (bound)
            
        else:
            if a < 0.8:
                vertices1[:,1] *= ((bound))
                
                v1[:,1] *= (bound)
               
            else:
                if a < 1:
                    vertices1[:,2] *= ((bound))
                    
                    v1[:,2] *= (bound)
                    
    return vertices1, v1
def normalize_mesh2(vertices1, v1, bound=1):

    centroid = np.mean(np.vstack((vertices1,v1))[:, 0:3], axis=0)
    vertices1 -= centroid
    v1 -= centroid
    
    
    max_distance = np.max(np.linalg.norm(np.vstack((vertices1, v1)), axis=1))     
    vertices1 /= (max_distance)
    v1 /= (max_distance)
    
    

    vertices1 *= ((bound))
    v1 *= (bound)
    
                    
    return vertices1, v1

def normalize_mesh3(vertices1_w, vertices1_l, v1,  bound=1):

    centroid = np.mean(np.vstack((vertices1_w, v1))[:, 0:3], axis=0)
    vertices1_w -= centroid
    vertices1_l -= centroid
    v1 -= centroid
    
    max_distance = np.max(np.linalg.norm(np.vstack((vertices1_w, v1)), axis=1))     
    vertices1_w /= (max_distance)
    vertices1_l /= (max_distance)
    v1 /= (max_distance)
    

    vertices1_w *= ((bound))
    vertices1_l *= ((bound))
    v1 *= (bound)
                    
    return vertices1_w, vertices1_l, v1

def normalize_mesh4(vertices1_w, vertices1_l, v1,vgt,  bound=1):

    centroid = np.mean(np.vstack((vgt, v1))[:, 0:3], axis=0)
    vertices1_w -= centroid
    vgt -= centroid
    vertices1_l -= centroid
    v1 -= centroid
    
    max_distance = np.max(np.linalg.norm(np.vstack((vgt, v1)), axis=1))     
    vertices1_w /= (max_distance)
    vertices1_l /= (max_distance)
    v1 /= (max_distance)
    vgt /= (max_distance)
    

    vertices1_w *= ((bound))
    vertices1_l *= ((bound))
    vgt *= ((bound))
    v1 *= (bound)
                    
    return vertices1_w, vertices1_l, v1, vgt
  
def get_tokenizer(opt: Options):
    if opt.use_meto:
        from meto import Engine
        tokenizer = Engine(discrete_bins=opt.discrete_bins, backend=opt.meto_backend)
        vocab_size = tokenizer.num_tokens + 3
    else:
        tokenizer = None
        vocab_size = opt.discrete_bins + 3
    return tokenizer, vocab_size


def quantize_num_faces(n):
    # 0: <=0, un cond
    # 1: 0-1000, low-poly
    # 2: 1000-2000, mid-poly
    # 3: 2000-4000, high-poly
    # 4: 4000-8000, ultra-poly
    if isinstance(n, int):
        if n <= 0:
            return 0
        elif n <= 1000:
            return 1
        elif n <= 2000:
            return 2
        elif n <= 4000:
            return 3
        elif n <= 8000:
            return 4
        else:
            return 5
    else: # torch tensor
        results = torch.zeros_like(n)
        # results[n <= 0] = 0
        results[(n > 0) & (n <= 1000)] = 1
        results[(n > 1000) & (n <= 2000)] = 2
        results[(n > 2000) & (n <= 4000)] = 3
        results[(n > 4000) & (n <= 8000)] = 4
        results[n > 8000] = 5
        return results
    
def monkey_patch_transformers():
    import torch
    import math
    from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor, ExponentialDecayLengthPenalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        # MODIFICATION: use input_ids.shape[0] instead of -1 to avoid confusion
        for batch_id, beam_sent in enumerate(input_ids.view(input_ids.shape[0], self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        scores_processed = scores + mask
        return scores_processed
    
    PrefixConstrainedLogitsProcessor.__call__ = __call__
    print(f'[INFO] monkey patched PrefixConstrainedLogitsProcessor.__call__')
