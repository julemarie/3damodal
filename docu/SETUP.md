# How to run the code

1. Setup conda environment
```
conda create amodal_env python=3.8
conda activate amodal_env
```

2. Install dependencies

2.1 For AISFormer:
```
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=10.2 -c pytorch
pip install ninja yacs cython matplotlib tqdm
pip install opencv-python==4.4.0.40
pip install scikit-image
pip install timm==0.4.12
pip install setuptools==59.5.0
pip install torch-dct

python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
```
For other cuda versions, check here:  [installing pytorch with cuda](https://pytorch.org/get-started/previous-versions/), [installing detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

2.2 Verify that PyTorch runs with cuda enabled (necessary):
```
python
import torch
torch.__version__
torch.cuda.is_available()
```
2.3 Download detectron2 and move into repository
Download zip folder from detectron2 repository [|Link|](https://github.com/facebookresearch/detectron2)
```
unzip detectron2.zip
cd detectron2
mv detectron2 /your-path-to/3damodal/3DAmodal
```

2.4 For PointPillars
```
cd 3DAmodal/pointpillars/ops
python setup.py develop
```

2.5 Further dependencies
```
pip install numba
...
```

## Training
Training loop is in train_aisformer.py. Track training in tensorboard.
To configure training settings, change configs/config.yaml
```
python train_aisformer.py

tensorboard --log_dir /pathto/runs_pp
```

## Testing
