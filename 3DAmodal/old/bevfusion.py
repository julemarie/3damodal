import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from asd_dataset import AmodalSynthDriveDataset
from open3d.ml.torch.models import PointPillars

from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork

img_size = (448, 800)
voxel_size = () # see official settings of lidar stream (transfusion)


"""
evaluation metrics
mean Average Precision (mAP) - defined by BEV center distance instead of 3D IoU, final mAP computed
    by averaging over distance thresholds of 0.5m, 1m, 2m, 4m across 10 classes 
        (car, truck, bus, trailer, construction vehicle, pedestrian, motorcycle, bicycle, barrier, traffic cone)
average Translation Error (ATE), Average Scale Error (ASE),
Average Orientation Error (AOE), Average Velocity Error (AVE), Average Attribute Error (AAE)
nuScenes detection score (NDS)
"""



class BEVFusion(nn.Module):
    """
    fusion model
    """

    def __init__(self,
                 in_channels=[96,192,384,768],
                 out_channels=256,
                 final_dim=(1080,1920),
                 downsample=8):
        super().__init__()
        ###Image-View Encoder
        self.img_view_encoder = ImageViewEncoder(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_outs=self.num_outs,
            final_dim=self.final_dim,
            downsample=self.downsample
        )

        ###View Projector Module

        self.lidar_backbone = PointPillars(name="PointPillars",
                                           voxel_size=[1,1,1], # xyz, TODO
                                           point_cloud_range=[-1, -1, -1, 1, 1, 1], # min:xyz, max:xyz TODO
                                           #voxelize="",
                                           #scatter="",
                                           #backbone="SECOND",
                                           #neck="SECOND",
                                           #head=""
                                           )
        self.lidar_backbone.eval()


    def forward(self, x):
        pts = x["lidar"]["points"]
        transform = x["lidar"]["transform"]
        horiz_angle = x["lidar"]["horizontal_angle"]

        imgs = [x["front_full_"], x["back_full_"], x["left_full_"], x["right_full_"]]
        features_2d = []
        for img in imgs:
            features_2d.append(self.img_view_encoder(img))

        return features_2d




class ImageViewEncoder(nn.Module):
    """
    encode input images into semantic information-rich deep features
    2D backbone for basic feature extraction - ResNet50
    neck module for scale variate object representation - FPN
    ADP Adaptive Module to refine upsampled features
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 final_dim,
                 downsample):
        super.__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.final_dim = final_dim
        self.target_size = (self.final_dim[0] // downsample, self.final_dim[1] //downsample)

        self.img_backbone = resnet50(pretrained=True)
        self.img_backbone.eval()

        self.img_neck = FeaturePyramidNetwork(
            in_channels_list=self.in_channels,
            out_channels=self.out_channels
        )

        adp_list = []
        for i in range(self.num_outs):
            if i == 0:
                resize = nn.AdaptiveAvgPool2d(self.target_size)
            else:
                resize = nn.Upsample(size=self.target_size,
                                     mode='bilinear')
            self.conv = nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True
            )
            
            self.transposed = self.conv.transposed
            self.output_padding = self.conv.output_padding

            self.conv = nn.utils.spectral_norm(self.conv) # TODO ??
            adp = nn.Sequential(resize, self.conv)
            adp_list.append(adp)
        self.adp = nn.ModuleList(adp_list)

        self.reduced_conv = nn.Conv2d(
            self.out_channels * self.num_outs,
            256,
            3,
            padding=1
        )
        
    def forward(self, x):
        outs = self.img_backbone(x)
        outs = self.img_neck.forward(x)

        if len(outs) > 1:
            resize_outs = []
            for i in range(len(outs)):
                feature = self.adp[i](outs[i])
                resize_outs.append(feature)
            out = torch.cat(resize_outs, dim=1)
            out = self.reduced_conv(out)
        else:
            out = outs[0]

        return [out]


class ViewProjectorModule(nn.Module):
    """
    project image features from 2D image coordinates to 3D ego-car coordinate
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class BEVEncoderModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class DynamicFusionModule(nn.Module):
    """
    
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class ObjectDetectionHead(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


def train_one_epoch(model, train_dl, optimizer, loss_fn):
    running_loss = 0

    for i, data in enumerate(train_dl):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss



    """
    hyperparameters and settings: Appendix B1
    """
def train_bevfusion(train_dl):
    # First: train image + LiDAR stream separately
    # (here: using pre-trained models)

    # BEV-space data augmentation

    # Then: train with both streams for 9 more epochs, inherit the weights from both streams
    fusion_model = BEVFusion()
    loss_fn = nn.CrossEntropyLoss() # TODO
    optimizer = torch.optim.SGD(fusion_model.parameters(), lr=0.001, momentum=0.9) # TODO

    epoch_num = 0
    for epoch in range(9):
        print(f"EPOCH {epoch_num+1}")
        fusion_model.train(True)
        avg_loss = train_one_epoch(fusion_model, train_dl, optimizer)



if __name__ == "__main__":
    data_root = "/home/jule-magnus/dd2414/Data/AmodalSynthDrive/train"
    ds = AmodalSynthDriveDataset(data_root)
    train_size = int(len(ds)*0.8)
    test_size = len(ds) - train_size
    train_set, test_set = torch.utils.data.random_split(ds, [train_size, test_size])
    train_dl = DataLoader(train_set)
    test_dl = DataLoader(test_set)
    #train_bevfusion(train_dl)

    print(train_set[0][1]["front_full_"][0]["bbox"])    
