PV-RAFT
===
This repository contains the PyTorch implementation for paper "3D Point-Voxel Correlation Fields for Scene Flow Estimation" (TPAMI 2023)\[[IEEE](https://ieeexplore.ieee.org/document/10178057)\]

<img src="PV_RAFT.png" width='600'>

## Installation

### Prerequisites
- Python 3.8
- PyTorch 1.8
- torch-scatter
- CUDA 10.2
- 8 * RTX 2080 Ti
- MinkowskiEngine
- tqdm, tensorboard, scipy, imageio, png

```Shell
conda create -n pvraft python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install tqdm tensorboard scipy imageio
pip install pypng
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
```

## Usage

### Data Preparation
We follow [HPLFlowNet](https://web.cs.ucdavis.edu/~yjlee/projects/cvpr2019-HPLFlowNet.pdf) to prepare FlyingThings3D and KITTI datasets. Please refer to [repo](https://github.com/laoreja/HPLFlowNet). Make sure the project structure look like this:
```Shell
RAFT_SceneFlow/
    data/
        FlyingThings3D_subset_processed_35m/
        kitti_processed/
    data_preprocess/
    datasets/
    experiments/
    model/
    modules/
    tools/
```
After downloading datasets, we need to preprocess them.
#### FlyingThings3D Dataset
```Shell
python process_flyingthings3d_subset.py --raw_data_path=path_src/FlyingThings3D_subset --save_path=path_dst/FlyingThings3D_subset_processed_35m --only_save_near_pts
```
You should replace `raw_data_path` and `save_path` with your own setting.

#### KITTI Dataset
```Shell
python process_kitti.py --raw_data_path=path_src/kitti --save_path=path_dst/kitti_processed --calib_path=calib_folder_path
```
You should replace `raw_data_path`, `save_path` and `calib_path` with your own setting.

### Train
```Shell
python -m torch.distributed.launch --nproc_per_node=8 train.py --exp_path=dpv_raft --batch_size=8 --gpus=0,1,2,3,4,5,6,7 --num_epochs=20 --max_points=8192 --iters=8  --root=./
```
where `exp_path` is the experiment folder name and `root` is the project root path. 

### Test
```Shell
python -m torch.distributed.launch --nproc_per_node=1 python test.py --dataset=KITTI --exp_path=dpv_raft --gpus=0 --max_points=8192 --iters=32 --root=./ --weights=./experiments/dpv_raft/checkpoints/best_checkpoint.params
```
where `dataset` should be chosen from `FT3D/KITTI`, and `weights` is the absolute path of checkpoint file. During testing, we use 32 iterations for KITTI and 8 iterations for FT3D.

### Reproduce results
You can download the checkpoint of DPV-RAFT model from [here](https://drive.google.com/file/d/1fpgMjtMP5n41t88C0W8Yi6np9E44VDpR/view?usp=sharing).

## Acknowledgement
Our code is based on [FLOT](https://github.com/valeoai/FLOT). We also refer to [RAFT](https://github.com/princeton-vl/RAFT) and [HPLFlowNet](https://github.com/laoreja/HPLFlowNet).

## Citation 
If you find our work useful in your research, please consider citing:
```
@article{wang2023dpvraft,
  title={3D Point-Voxel Correlation Fields for Scene Flow Estimation},
  author={Wang, Ziyi and Wei, Yi and Rao, Yongming and Zhou, Jie and Lu, Jiwen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```

