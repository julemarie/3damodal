import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from torchvision.models import resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from torchvision.utils import save_image

import cv2
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from utils import Attention, CrossAttention


class AISFormer(nn.Module):
    """
        AISFormer using depth information.
    """
    def __init__(self, in_shape, cfg) -> None:
        # backbone
        # spatial_scaling
        
        super().__init__()
        self.H, self.W = in_shape
        self.in_chans = cfg.MODEL.PARAMS.IN_CHANS
        self.roi_out_size = cfg.MODEL.PARAMS.ROI_OUT_SIZE
        # output of backbone: list of dictionaries {boxes: tensor(4,num_boxes), labels: tensor(num_boxes), scores: tensor(num_boxes)}
        self.backbone = fasterrcnn_resnet50_fpn(pretrained=True)
        self.backbone.eval()
        # ROIAlign takes images and ROIs as input and outputs a tensor of size [K, in_chans, (OUT_SIZE)] where K = #bboxes
        self.roi_align = RoIAlign(output_size=(self.roi_out_size, self.roi_out_size), spatial_scale=1, sampling_ratio=2,aligned=True)
        # self.next = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # deconvolution 
        self.deconv = nn.ConvTranspose2d(in_channels=self.in_chans, out_channels=self.in_chans, kernel_size=(2,2), stride=2)
        self.conv = nn.Conv2d(in_channels=self.in_chans, out_channels=self.in_chans, kernel_size=(1,1), stride=1)
        # self.encoder = Encoder(img_size=self.roi_out_size*2, in_chans=3, num_heads=3)
        # self.decoder = Decoder(img_size=self.roi_out_size*2, in_chans=3, num_heads=3)
        # self.mlp = nn.ModuleList([
        #     nn.Linear(in_features=2*in_chans, out_features=in_chans),
        #     nn.Linear(in_chans, in_chans),
        #     nn.Linear(in_chans, in_chans)
        # ])
        # self.unflatten = nn.Unflatten(2, (self.roi_out_size*2, self.roi_out_size*2))

        
    def forward(self, imgs):
        B, C, H, W = imgs.shape
        thresh = 0.9
        x = self.backbone(imgs)
        roi_list = []
        for i, ann in enumerate(x):
            boxes = ann["boxes"][torch.where(ann["scores"] > thresh)]
            id = i*torch.ones((boxes.shape[0])).unsqueeze(-1)
            roi_list.append(torch.cat((id, boxes), dim=1))

        rois = torch.cat(roi_list, dim=0) # [K, 5] K: #bboxes, 5: first for id and last four for corners

        f_roi2 = self.roi_align(imgs, rois) # [K, C, H_r, W_r]
        f_roi1 = self.conv(self.deconv(f_roi2)) # [K, C, H_m, W_m]
        # f_e = self.encoder(f_roi) # [K, H_m*W_m, C]
        # Q = self.decoder(f_e) # [C, 3]
        # Q_vis_am = torch.cat((Q[:,1], Q[:,2]), dim=0)
        # I_i = Q_vis_am
        # for i, layer in enumerate(self.mlp):
        #     I_i = F.gelu(layer(I_i)) if i < len(self.mlp-1) else layer(I_i)  # [C]
        # I_i = I_i.unsqueeze(-1)
        # roi_embed = self.unflatten(f_roi.flatten(2) + f_e.permute(0,2,1)) # [K, C, H_m, W_m]
        # masks = torch.tensordot(roi_embed, torch.cat((Q, I_i), dim=1), dims=([1], [0])).permute(0,3,1,2)
        return f_roi2, f_roi1


class Encoder(nn.Module):
    """
        Transformer encoder with one block of self-attention.
    """
    def __init__(self, img_size, in_chans, num_heads):
        super().__init__()

        self.embed_dim = img_size**2
        self.pos_encod = nn.Parameter(torch.zeros(1, self.embed_dim, in_chans))
        self.attention = Attention(in_chans, num_heads)
        self.norm = nn.LayerNorm(in_chans)
        self.ffn = nn.Linear(in_chans, in_chans)

    def forward(self, x):
        # flatten feature vectors
        x = x.flatten(2).permute(0,2,1) # [K, H_m*W_m, C]

        # add positional encoding
        x = x + self.pos_encod
        # self attention layer and add x
        x = x + self.attention(x)
        # apply layer norm and linear layer
        x = self.ffn(self.norm(x))
        
        return x
    
class Decoder(nn.Module):
    """
        Transformer decoder with one block of self-attention and one block of cross-attention.
    """
    def __init__(self, img_size, in_chans, num_heads):
        super().__init__()

        self.Q = nn.Parameter(torch.ones((in_chans, 3)).unsqueeze(0))

        self.attention = Attention(in_chans, num_heads)
        self.norm_self = nn.LayerNorm(in_chans)
        self.ffn_self = nn.Linear(in_chans, in_chans)
        self.cross_attention = CrossAttention(in_chans, num_heads)
        self.norm_cross1 = nn.LayerNorm(in_chans)
        self.ffn_cross1 = nn.Linear(in_chans, in_chans)
        self.norm_cross2 = nn.LayerNorm(in_chans)
        self.ffn_cross2 = nn.Linear(in_chans, in_chans)

    def forward(self, x):
        K, D, N = x.shape
        Q = self.Q.repeat(K, 1, 1)
        attn_Q = self.Q + self.attention(Q)
        attn_Q = self.ffn_self(self.norm_self(attn_Q))

        Q = Q + attn_Q

        Q = Q + self.cross_attention(x, Q)
        Q = self.ffn_cross1(self.norm_cross1(Q))
        Q = Q + self.ffn_cross2(self.norm_cross2(Q))

        Q = Q[0, :, :].squeeze(0)

        return Q


            

if __name__ == "__main__":
    # load config
    cfg = OmegaConf.load("configs/config.yaml")

    in_shape = tuple((cfg.MODEL.PARAMS.INPUT_H, cfg.MODEL.PARAMS.INPUT_W))

    img = cv2.imread("front_full_0000_rgb.jpg")
    img2 = cv2.imread("front_full_0063_rgb.jpg")
    H, W, C = img.shape # for ASD: [1080, 1920, 3]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomCrop((int(H/1.5), int(W/1.5))),
        transforms.Resize(in_shape)
    ])
    img = transform(img).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)

    imgs = torch.cat((img, img2), dim=0)

    print(imgs.shape)


    aisformer = AISFormer(in_shape=in_shape, cfg=cfg)
    
    f_roi2, f_roi1 = aisformer(imgs)
    print(f_roi2.shape)

    save_image(f_roi2[0], "roi.png")
    save_image(f_roi1[0], "roi_deconv.png")

    
    # masks = aisformer(imgs)
    