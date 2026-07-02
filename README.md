# [CVPR 2026] BuildingGPT: Auto-Regressive Building Wireframe Reconstruction Model with Reinforcement Learning
Official implementation of 'BuildingGPT: Auto-Regressive Building Wireframe Reconstruction Model with Reinforcement Learning'. 

## Abstract
In this paper, we propose BuildingGPT, a novel autoregressive
model for building wireframe reconstruction
from point clouds with reinforcement learning. Unlike
prior works based on detection or diffusion models, BuildingGPT
reformulates the building wireframe reconstruction
task into a sequence prediction problem. Based
on the hierarchical building wireframe tokenization, the
wireframe sequences are organized in a structurally- and
semantically-aware order for the next-token prediction. The
point cloud encoder first transforms the input point cloud
into a fixed-length latent code prepended before the wireframe
sequence. Then, BuildingGPT auto-regressively predicts
tokens conditioned on the latent code. After detokenization,
the building wireframe is obtained. To enhance
the model performance, we adopt a two-stage training
paradigm including the pre-training and post-training. After
the auto-regressive pre-training, Direct Preference Optimization
(DPO) is employed as a post-training strategy to
align reconstruction results with human preferences. Extensive
experiments on the large-scale MunichWF dataset
show that BuildingGPT outperforms existing state-of-theart
methods.

## Method
<img src="./imgs/overall.png" width=100% height=100%>

**Overall architecture of BuildingGPT.** Our BuildingGPT is trained in two stages. In the first stage, the model is pre-trained in an
auto-regressive manner. Given the latent code encoded by the point cloud encoder, the wireframe sequence is generated through next-token
prediction. In the second stage, we construct a preference pair dataset using the proposed Preference Score Function (PSF) and post-train
the model with Direct Preference Optimization (DPO) to further enhance reconstruction quality.

## Environment
```
git clone https://github.com/3dv-casia/BuildingGPT
cd BuildingGPT
uv sync
uv sync --group flash-attn
```

Requires [uv](https://docs.astral.sh/uv/). The first command creates a `.venv` and installs core dependencies; the second adds flash-attn (built without isolation, as upstream requires).

PyTorch is pinned to **CUDA 12.4** wheels (`torch 2.6.0+cu124`) via the PyTorch index in `pyproject.toml`, so it works with driver 12.4 without upgrading the system CUDA stack.

## Data
The processed dataset of MunichWF is stored in [this link](https://huggingface.co/datasets/zyl111/MunichWF/tree/main)

## Training 
```
# debug training
uv run accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train

# single-node training (use slurm for multi-nodes training)
uv run accelerate launch --config_file acc_configs/gpu8.yaml main.py ArAE --workspace workspace_train
```

## Inference 
```
# test_path is a text file with one input per line: a .ply path or a basename under data/pc/
echo "data/pc/your_pointcloud.ply" > my_input.txt

uv run python infer.py ArAE --workspace workspace --resume pretrained/ArAE.safetensors --test_path my_input.txt --generate_mode sample --test_num_face 1000 --test_repeat 1 --seed 42
```

## Checkpoints
The checkpoints can be downloaded in [this link](https://drive.google.com/file/d/1MLAF3pCVjt8Z27aDnfkMO_IokfN3B4P3/view?usp=drive_link).


## Citation
If you find BuildingGPT useful in your research, please cite our paper:
```
@inproceedings{liu2026BuildingGPT,
  title={BuildingGPT: Auto-Regressive Building Wireframe Reconstruction Model with Reinforcement Learning},
  author={Liu, Yuzhou and Zhu, Lingjie and Ye, Hanqiao and Liu, Yujun, and Huang, Shangfeng and Gao, Xiang and Wang, Ruisheng and Shen, Shuhan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={36400-36410},
  year={2026}
}
```

## Acknowledgment
We thank the following excellent projects especially EdgeRunner:
* [EdgeRunner](https://github.com/NVlabs/EdgeRunner/tree/main)
* [transformers](https://github.com/huggingface/transformers)
