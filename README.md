# Multi-modal amodal instance segmentation

In this repository, we explore combining two modalities for the amodal instance segmentation task. We adapt existing models for amodal instance segmentation and 3D object detection to predict amodal instances using both image and LiDAR data. For the amodal instance segmentation, we use AISFormer [(Tran et al. 2022)](https://arxiv.org/pdf/2210.06323v3.pdf), and for 3D Object detection, we use a [Pytorch implementation](https://github.com/zhulf0804/PointPillars) of PointPillars [(Lang et al. 2019)](https://arxiv.org/pdf/1812.05784.pdf). As data, we combine the [KINS](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset) and [KITTI](https://www.cvlibs.net/datasets/kitti/) datasets. 

## Pipeline

First, we predict 3D bounding boxes on KITTI point clouds using PointPillars. We use these bounding boxes for two purposes: to create a depth mask feature and as input into ROI align. 
To create a depth mask, we identify all points that lie within the relevant bounding boxes. We map these points onto the 2D image space and keep the z-coordinate depth information. Then, we interpolate and upsample these partial point clouds to get a depth value for each pixel in the image space bounding box.
We add the depth mask as a fourth channel to the input KITTI image. The resulting RGB-D image is used as input for the ResNet-FPN feature extractor.
The extracted features combined with 2D bounding boxes (which we get from mapping the 3D bounding boxes to the image space) are used to perform ROI align.
The features and ROI align are then used as input to the AISFormer model, which predicts 2D amodal masks using KINS annotations.

The pipeline is visualized in the figure below.

![Visualisation of our model's pipeline](docu/img/3damodalstructure.png "Visualisation of our model's pipeline")

## Repository structure and documentation

- Source code is included in the 3DAmodal subfolder
- Documentation is available in the docu subfolder
- For information on setup and training see [SETUP](docu/SETUP.md)
- For information on the theoretical background see [BACKGROUND](docu/BACKGROUND.md)
- For more information on our implementation and results see [CONCEPT](docu/CONCEPT.md)

## Sources
We use parts of and extend existing code bases in our repository.
**AISFormer** was introduced by [(Tran et al. 2022)](https://arxiv.org/pdf/2210.06323v3.pdf). The original code can be found at their [repository](https://github.com/UARK-AICV/AISFormer?tab=readme-ov-file#citation).
**PointPillars** was introduced by [(Lang et al. 2019)](https://arxiv.org/pdf/1812.05784.pdf). The original repository can be found at this [repository](https://github.com/nutonomy/second.pytorch), but here we use a [Pytorch implementation](https://github.com/zhulf0804/PointPillars).
**KINS** is a dataset for amodal segmentation introduced by [Qi et al.](https://ieeexplore.ieee.org/document/8954364). The annotations are available [here](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset).
**KITTI** is a dataset for 3D object detection. We refer to the project [website](https://www.cvlibs.net/datasets/kitti/).

