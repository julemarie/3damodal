# 3D Amodal Instance Segmentation Using Sensor Fusion

The amodal instance segmentation task was defined by [Li et al](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_42) (2017) as predicting both visible and occluded parts of the segmented objects. With the growing popularity of transformers for computer vision tasks, including segmentation (see [Mask2Former](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html)), a transformer-based method has emerged as the state-of-the-art method for amodal instance segmentation in 2023, namely the [AISFormer](https://arxiv.org/abs/2210.06323). 

This repository is a working exploration of adding another modality to the AISFormer approach to achieve even better results. The idea is to generate a depth mask from LiDAR data and add it as a fourth channel to the input of the amodal instance segmentation model. 

## AISFormer model
### Original
The AISFormer is a proposal based approach, meaning that it detects segmentation masks within bounding boxes given by an object detection backbone. The architecture of the AISFormer is described below (see the [original repo](https://github.com/UARK-AICV/AISFormer) for further reference).

![Architecture of AISFormer](doc/arch.png "AISFormer architecture")

## Re-Implementation
- re-implemented Aisformer in pytorch from scratch
- model is from original repo with some adaptations
- smaller model size than in original repo due to gpu memory constraints

## Backbone and depth mask

## Dataset
### [KINS](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset)
Extending KITTI instance segmentation with amodal instance segmentation annotations. 7400 training and 7400 test images.



### [AmodalSynthDrive](http://amodalsynthdrive.cs.uni-freiburg.de)
Synthetic dataset recorded using CARLA simulator, which includes labels for various different tasks, as well as LiDAR data.
Currently (as of January 2024) unavailable due to issues with the LiDAR data.

## Training
### AISFormer
- gpu constraints didn't allow us to train the full model -> two approaches:
    - split model on gpus
    - use smaller model (less encoder/decoder layers)
- in the end decided to use smaller model, since training bigger model would take more time
- smaller model can be parallelized on multiple gpus

### AISFormer + PointPillars

## Evaluation and Results
