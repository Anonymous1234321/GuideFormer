# GuideFormer
This repo is the PyTorch implementation of our paper "GuideFormer: Transformers for Image Guided Depth Completion".
<div align=center><img src="https://github.com/Anonymous1234321/GuideFormer/blob/main/images/model%20architecture.png" width = "100%" height = "100%" /></div>


## Install
Our released implementation is tested on.
+ Ubuntu 18.04
+ Python 3.8.10
+ PyTorch 1.8.1 / torchvision 0.9.1
+ NVIDIA CUDA 11.0
+ 8x NVIDIA Tesla V100 GPUs

```bash
pip install numpy matplotlib Pillow
pip install scikit-image
pip install opencv-contrib-python==3.4.2.17
pip install einops
pip install timm
```

## Data
- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset from their websites.
The overall data directory is structured as follows:
```
├── kitti_depth
|   ├── depth
|   |   ├──data_depth_annotated
|   |   |  ├── train
|   |   |  ├── val
|   |   ├── data_depth_velodyne
|   |   |  ├── train
|   |   |  ├── val
|   |   ├── data_depth_selection
|   |   |  ├── test_depth_completion_anonymous
|   |   |  |── test_depth_prediction_anonymous
|   |   |  ├── val_selection_cropped
```

## Commands
A complete list of training options is available with
```bash
python main.py -h
```

### Training
```bash
# Non-distributed GPU setting
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python main.py -b 8

# Distributed GPU setting
python -m torch.distributed.launch --nproc_per_node=8 main_disrtibuted.py -b 8
```

### Validation
```bash
CUDA_VISIBLE_DEVICES="0" python main.py -b 1 --evaluate [checkpoint-path]
# evaluate the trained model on the KITTI validation set(val_selection_cropped)
```

### Test
```bash
CUDA_VISIBLE_DEVICES="0" python main.py -b 1 --evaluate [checkpoint-path] --test
# generate and save results of the trained model on the KIITI test set (test_depth_completion_anonymous)
```

## Related Repositories
The original code framework is rendered from ["PENet: Precise and Efficient Depth Completion"](https://github.com/JUGGHM/PENet_ICRA2021) (which is also rendered from ["Self-supervised Sparse-to-Dense:  Self-supervised Depth Completion from LiDAR and Monocular Camera"](https://github.com/fangchangma/self-supervised-depth-completion)).

And the part of utils is rendered from ["Swin Transformer"](https://github.com/microsoft/Swin-Transformer).


