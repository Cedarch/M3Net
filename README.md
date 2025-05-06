## AAAI 2025 [M3Net: Multimodal Multi-task Learning for 3D Detection, Segmentation, and Occupancy Prediction in Autonomous Driving](https://arxiv.org/abs/2503.18100)
## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.

Build deformable 3D attention ops as follows:
```bash
cd pcdet/ops/deform_attn_3d 
python setup.py build_ext --inplace
```

If you want to use the Mamba version, please install the VMamba library as follows:
```bash
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
git checkout e9219a4fdc16bd0ac8252ae03834dfc3c189eb2a  # This version is used for training M3Net
pip install -r requirements.txt 
cd kernels/selective_scan && pip install .
```

## Data Preparation

### Dataset Preparation

* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and organize the downloaded files as follows: 

```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

- To install the Map expansion for bev map segmentation task, please download the files from [Map expansion](https://www.nuscenes.org/download) (Map expansion pack (v1.3)) and copy the files into your nuScenes maps folder, e.g. `/data/nuscenes/v1.0-trainval/maps` as follows:
```
OpenPCDet
├── maps
│   ├── ......
│   ├── boston-seaport.json
│   ├── singapore-onenorth.json
│   ├── singapore-queenstown.json
│   ├── singapore-hollandvillage.json
```

* Download the nuScenes-Occupancy dataset from [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md) through one of these links:

| Subset | Google Drive | Baidu Cloud | Size |
|:-------|:-------------|:------------|:-----|
| trainval-v0.1 | [link](link_url) | [link](link_url) (code:25ue) | approx. 5G |


```bash
mv nuScenes-Occupancy-v0.1.7z ./data/nuscenes
cd ./data/nuscenes
7za x nuScenes-Occupancy-v0.1.7z
```
folder as follows:
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval
|   |   |── nuScenes-Occupancy-v0.1  
├── pcdet
├── tools
```

* Generate the data infos by running the following command (it may take several hours): 

```python 
# Create dataset info file, lidar and image gt database following UniTR
git clone https://github.com/Haiyang-W/UniTR.git
cd UniTR
ln -s path/to/OpenPCDet/data/nuscenes/* ./data/nuscenes/
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval \
    --with_cam \
    --with_cam_gt \
    --share_memory # if use share mem for lidar and image gt sampling (about 24G+143G or 12G+72G)
# share mem will greatly improve your training speed, but need 150G or 75G extra cache mem. 
# NOTE: all the experiments used share memory. Share mem will not affect performance
```

Download pre-processed multi-task train pkl from [train pkl](https://drive.google.com/file/d/1N_YtR_SHc7ICm23zVZqqMoX22luoBk9L/view?usp=sharing) and val pkl from [val pkl](https://drive.google.com/file/d/19UY9C-4vedGgsxgCuh99K1QdECjfoMn9/view?usp=sharing) and put them into ./data/nuscenes/v1.0-trainval folder.

* The format of the generated data is as follows:
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
│   │   │   │── img_gt_database_10sweeps_withvelo
│   │   │   │── gt_database_10sweeps_withvelo
│   │   │   │── nuscenes_10sweeps_withvelo_lidar.npy # if open share mem
│   │   │   │── nuscenes_10sweeps_withvelo_img.npy # if open share mem
│   │   │   │── nuscenes_infos_10sweeps_train.pkl
|   |   |   |── nuscenes_infos_10sweeps_train_occ.pkl  
│   │   │   │── nuscenes_infos_10sweeps_val.pkl
│   │   │   │── nuscenes_infos_10sweeps_val_occ.pkl
│   │   │   │── nuscenes_dbinfos_10sweeps_withvelo.pkl
├── pcdet
├── tools
```


## Training
We train M3Net on 8*A100.
1.  Train the LSS backbone for BEV feature extraction by detection task.
```shell

python -m torch.distributed.launch --nproc_per_node=8 train.py --launcher pytorch \
--cfg_file cfgs/nuscenes_models/m3net_det.yaml \
--sync_bn  True  --batch_size 8 --workers 2 \
--set USE_TCS False OPTIMIZATION.NUM_EPOCHS 6
```
The ckpt will be saved in ../output/nuscenes_models/m3net_det/default/ckpt.
We only keep the weights of backbone and drop the weights of detection-specific head as follows:
```shell

python pcdet/utils/remove_head_weights.py --input_ckpt /path/to/input/checkpoint.pth --output_ckpt /path/to/output/checkpoint.pth
```
You can also use our pretrained weights: [download link](https://drive.google.com/file/d/1rgzpNfdw-tqBvMX2Bf-UQi46lIwwIgno/view?usp=sharing)

2.  Train detection task and BEV map segmentation task.
```shell
# Transformer
python -m torch.distributed.launch --nproc_per_node=8 train.py --launcher pytorch \
--cfg_file cfgs/nuscenes_models/m3net_det_seg.yaml \
--sync_bn True  --batch_size 8 --workers 2 --extra_tag transformer \
--pretrained_model path/to/preatrained_backbone
--set USE_VM_ENC False OPTIMIZATION.NUM_EPOCHS 12
#Mamba
python -m torch.distributed.launch --nproc_per_node=8 train.py --launcher pytorch \
--cfg_file cfgs/nuscenes_models/m3net_det_seg.yaml --extra_tag mamba \
--sync_bn True  --batch_size 8 --workers 2 \
--pretrained_model path/to/preatrained_backbone
--set USE_VM_ENC True TCS_WITH_CA False USE_MAMBA3D_ATTN True OPTIMIZATION.NUM_EPOCHS 12

```
The transformer-based ckpt will be saved in ../output/nuscenes_models/m3net_det_seg/transformer/ckpt.
The mamba-based ckpt will be saved in ../output/nuscenes_models/m3net_det_seg/mamba/ckpt.

3.  Train three tasks (det, seg, occ) simutaineously.
```shell
# Transformer
python -m torch.distributed.launch --nproc_per_node=8 train.py --launcher pytorch \
--cfg_file cfgs/nuscenes_models/m3net_det_map_occ.yaml \
--sync_bn True  --batch_size 8 --workers 2 \
--pretrained_model ../output/nuscenes_models/m3net_det_seg/transformer/ckpt/checkpoint_epoch_12.pth
--set USE_VM_ENC False OPTIMIZATION.NUM_EPOCHS 15 
# Mamba
python -m torch.distributed.launch --nproc_per_node=8 train.py --launcher pytorch \
--cfg_file cfgs/nuscenes_models/m3net_det_map_occ.yaml \
--sync_bn True  --batch_size 8 --workers 2 \
--pretrained_model ../output/nuscenes_models/m3net_det_seg/mamba/ckpt/checkpoint_epoch_12.pth
--set USE_VM_ENC True TCS_WITH_CA False USE_MAMBA3D_ATTN True MODEL.DENSE_HEAD.OCC_LOSS_WEIGHT 1.2 OPTIMIZATION.NUM_EPOCHS 15 

```


## Evaluation
* Test with a pretrained model:
```shell
# 8 gpus
python -m torch.distributed.launch --nproc_per_node=8 test.py --launcher pytorch \
--cfg_file cfgs/nuscenes_models/m3net_det_map_occ.yaml \
--batch_size 8 --workers 2 \
--ckpt /path/to/your/ckpt
```


## Performance
| Model | Det (NDS) | Seg (mIoU) | Occ (mIoU) | Ckpt |
|:------|:---------|:----------|:-----------|:-----|
| M3Net (transformer) | 72.4 | 70.4 | 23.3 | [download link](https://drive.google.com/file/d/1l9tU2YSWoNHWWc-MhJRJ1ssV3LdpK58j/view?usp=sharing)  |
| M3Net (mamba) | 71.8 | 70.8 | 24.1 | [download link](https://drive.google.com/file/d/1HGJ8DVz3LEA9Xiz5MXFd05BAyj9bppWX/view?usp=sharing) |

