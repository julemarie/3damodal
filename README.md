# 3D Amodal Instance Segmentation Using Sensor Fusion

The amodal instance segmentation task was defined by [Li et al](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_42) (2017) as predicting both visible and occluded parts of the segmented objects. With the growing popularity of transformers for computer vision tasks, including segmentation (see [Mask2Former](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html)), a transformer-based method has emerged as the state-of-the-art method for amodal instance segmentation in 2023, namely the [AISFormer](https://arxiv.org/abs/2210.06323). 

This repository is a working exploration of adding another modality to the AISFormer approach to achieve even better results. The idea is to generate a depth mask from LiDAR data and add it as a fourth channel to the input of the amodal instance segmentation model. 

## AISFormer model

## Backbone and depth mask

## Training 

## Evaluation and Results