import torch
import torch.nn as nn
from torchvision.ops import RoIAlign
from torchvision.models import resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

import cv2

from utils import Attention, CrossAttention


class AISFormer(nn.Module):
    """
        AISFormer using depth information.
    """
    def __init__(self, img_size, in_chans=3) -> None:
        super().__init__()
        # output of backbone: list of dictionaries {boxes: tensor(4,num_boxes), labels: tensor(num_boxes), scores: tensor(num_boxes)}
        self.backbone = fasterrcnn_resnet50_fpn(pretrained=True)
        self.backbone.eval()
        self.roi_out_size = int(img_size/2.5)
        self.roi_align = RoIAlign(output_size=(self.roi_out_size, self.roi_out_size), spatial_scale=1, sampling_ratio=2)
        self.deconv = nn.ConvTranspose2d(in_channels=in_chans, out_channels=in_chans, kernel_size=(2,2), stride=2)
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=(1,1), stride=1)
        self.encoder = Encoder(img_size=self.roi_out_size*2, in_chans=3, num_heads=3)
        self.decoder = Decoder(img_size=self.roi_out_size*2, in_chans=3, num_heads=3)
        self.mlp = nn.ModuleList([
            nn.Linear(in_features=2*in_chans, out_features=in_chans),
            nn.Linear(in_chans, in_chans),
            nn.Linear(in_chans, in_chans)
        ])
        self.unflatten = nn.Unflatten(2, (self.roi_out_size*2, self.roi_out_size*2))

        
    def forward(self, imgs):
        B, C, H, W = imgs.shape
        thresh = 0.25
        x = self.backbone(imgs)
        tensors = []
        for i, ann in enumerate(x):
            boxes = ann["boxes"][torch.where(ann["scores"] > thresh)]
            id = i*torch.ones((boxes.shape[0])).unsqueeze(-1)
            tensors.append(torch.cat((id, boxes), dim=1))

        rois = torch.cat(tensors, dim=0) # [K, 5] K: #bboxes, 5: first for id and last four for corners

        f_roi = self.roi_align(imgs, rois) # [K, C, H_r, W_r]
        f_roi = self.conv(self.deconv(f_roi)) # [K, C, H_m, W_m]
        f_e = self.encoder(f_roi) # [K, H_m*W_m, C]
        Q = self.decoder(f_e) # [C, 3]
        Q_vis_am = torch.cat((Q[:,1], Q[:,2]), dim=0)
        I_i = Q_vis_am
        for m in self.mlp:
            I_i = m(I_i) # [C]
        I_i = I_i.unsqueeze(-1)
        roi_embed = self.unflatten(f_roi.flatten(2) + f_e.permute(0,2,1)) # [K, C, H_m, W_m]
        masks = torch.tensordot(roi_embed, torch.cat((Q, I_i), dim=1), dims=([1], [0])).permute(0,3,1,2)
        return x, f_roi, f_e, Q, I_i, roi_embed, masks





# class Backbone(nn.Module):
#     """
#         Backbone network for obtaining region of interest features (ROI).
#     """
#     def __init__(self) -> None:
#         super().__init__()

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
    img = cv2.imread("n01443537_275.JPEG")
    img2 = cv2.imread("n01443537_269.JPEG")
    h,w,c = img.shape
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        # transforms.RandomCrop(h//1.5),
        transforms.Resize(h)
    ])
    img = transform(img).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)
    imgs = torch.cat((img, img2), dim=0)

    aisformer = AISFormer(img_size=h)

    
    x, f_roi, f_e, Q, I, roi_embed, masks = aisformer(imgs)
    # print(x)
    print(imgs.shape)
    print(f_roi.shape)
    print(f_e.shape)
    print(Q.shape)
    print(I.shape)
    print(roi_embed.shape)
    print(masks)


    
