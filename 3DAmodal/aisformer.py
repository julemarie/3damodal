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

from utils import MultiHeadAttention


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
        self.Q_dim = cfg.MODEL.PARAMS.Q_DIM
        self.embed_dim = cfg.MODEL.PARAMS.EMBED_DIM
        self.encoder_depth = cfg.MODEL.PARAMS.ENCODER_DEPTH
        self.encoder_num_heads = cfg.MODEL.PARAMS.ENCODER_NUM_HEADS
        self.decoder_depth = cfg.MODEL.PARAMS.DECODER_DEPTH
        self.decoder_num_heads = cfg.MODEL.PARAMS.DECODER_NUM_HEADS
        if cfg.DEVICE == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'

        # output of backbone: list of dictionaries {boxes: tensor(4,num_boxes), labels: tensor(num_boxes), scores: tensor(num_boxes)}
        self.backbone = fasterrcnn_resnet50_fpn(pretrained=True)
        self.backbone.eval()
        # ROIAlign takes images and ROIs as input and outputs a tensor of size [K, in_chans, (OUT_SIZE)] where K = #bboxes
        self.roi_align = RoIAlign(output_size=(self.roi_out_size, self.roi_out_size), spatial_scale=1, sampling_ratio=2,aligned=True)
        # deconvolution 
        self.deconv = nn.ConvTranspose2d(in_channels=self.in_chans, out_channels=self.in_chans, kernel_size=(2,2), stride=2)
        self.conv = nn.Conv2d(in_channels=self.in_chans, out_channels=self.in_chans, kernel_size=(1,1), stride=1)
        self.encoder = Encoder(self.embed_dim, (self.roi_out_size*2)**2, num_heads=self.encoder_num_heads)
        self.decoder = Decoder(self.embed_dim, Q_dim=self.Q_dim, num_heads=self.decoder_num_heads)
        self.mlp = nn.ModuleList([
            nn.Linear(in_features=2*self.embed_dim, out_features=self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim)
        ])
        self.unflatten = nn.Unflatten(2, (self.roi_out_size*2, self.roi_out_size*2))

    def patchify(self, x):
        """
            Patchify a batch of feature tensors.
        """
        K, C, H, W = x.shape
        
        x = x.reshape(K, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)    # [K, H', W', p_H, p_W, C]
        x = x.reshape(K, -1, *x.shape[3:])   # [K, H'*W', p_H, p_W, C]
        x = x.reshape(K, x.shape[1], -1) # [K, H'*W', p_H*p_W*C]

        return x
    
    def unpatchify(self, x):
        """
            Unpatchify feature tensors.
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        c = self.in_chans
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
        
    def forward(self, imgs):
        B, C, H, W = imgs.shape
        thresh = 0.9
        x = self.backbone(imgs)
        roi_list = []
        for i, ann in enumerate(x):
            boxes = ann["boxes"][torch.where(ann["scores"] > thresh)]
            id = i*torch.ones((boxes.shape[0]), dtype=torch.int, device=self.device).unsqueeze(-1)
            roi_list.append(torch.cat((id, boxes), dim=1))

        if roi_list:
            rois = torch.cat(roi_list, dim=0) # [K, 5] K: #bboxes, 5: first for id and last four for corners
        else:
            return None, roi_list

        f_roi2 = self.roi_align(imgs, rois) # [K, C, H_r, W_r]
        f_roi1 = self.conv(self.deconv(f_roi2)) # [K, C, H_r', W_r']
        f_e = self.encoder(f_roi1) # [K, H_m*W_m, C]
        Q = self.decoder(f_e) # [C, 3]
        Q_vis_am = torch.cat((Q[:,1], Q[:,2]), dim=0)
        I_i = Q_vis_am
        for i, layer in enumerate(self.mlp):
            I_i = F.gelu(layer(I_i)) if i < len(self.mlp)-1 else layer(I_i)  # [C]
        I_i = I_i.unsqueeze(-1)
        roi_embed = self.unflatten(f_roi1.flatten(2) + f_e.permute(0,2,1)) # [K, C, H_m, W_m]
        masks = torch.tensordot(roi_embed, torch.cat((Q, I_i), dim=1), dims=([1], [0])).permute(0,3,1,2)
        masks = masks.reshape(masks.shape[0]*masks.shape[1], -1)
        masks -= masks.min(1, keepdim=True)[0]
        masks /= masks.max(1, keepdim=True)[0]
        masks = masks.reshape(rois.shape[0], 4, self.roi_out_size*2, self.roi_out_size*2)
        masks[masks < 0.5] = 0
        masks[masks >= 0.5] = 1
        return masks, rois # [K, 4, H_m, W_m]


class Encoder(nn.Module):
    """
        Transformer encoder with one block of self-attention.
    """
    def __init__(self, embed_dim, seq_len, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.pos_encod = nn.Parameter(torch.zeros(1, seq_len, self.embed_dim))
        self.attention = MultiHeadAttention(embed_dim, embed_dim, num_heads)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        # flatten feature vectors
        x = x.flatten(2).permute(0,2,1) # [K, H_m*W_m, C]
        
        # x shape: [K, N, embed_dim], N: num_patches

        # add positional encoding
        x = x + self.pos_encod
        # self attention layer and add x
        x = x + self.attention(input_q=x, input_kv=x)
        # apply layer norm and linear layer
        x = self.ffn(self.norm(x))
        
        return x
    
class Decoder(nn.Module):
    """
        Transformer decoder with one block of self-attention and one block of cross-attention.
    """
    def __init__(self, embed_dim, Q_dim, num_heads):
        super().__init__()

        self.Q = nn.Parameter(torch.ones((embed_dim, Q_dim)).unsqueeze(0))

        self.attention = MultiHeadAttention(Q_dim, Q_dim, num_heads)
        self.norm_self = nn.LayerNorm(Q_dim)
        self.ffn_self = nn.Linear(Q_dim, Q_dim)
        self.cross_attention = MultiHeadAttention(Q_dim,embed_dim, num_heads)
        self.norm_cross1 = nn.LayerNorm(Q_dim)
        self.ffn_cross1 = nn.Linear(Q_dim, Q_dim)
        self.norm_cross2 = nn.LayerNorm(Q_dim)
        self.ffn_cross2 = nn.Linear(Q_dim, Q_dim)

    def forward(self, x):
        K, N, D = x.shape
        Q = self.Q.repeat(K, 1, 1)
        attn_Q = Q + self.attention(input_q=Q, input_kv=Q)
        attn_Q = self.ffn_self(self.norm_self(attn_Q))

        Q = Q + attn_Q

        Q = Q + self.cross_attention(input_q=Q, input_kv=x)
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
    
    masks = aisformer(imgs) 
    
    print(masks.shape)
    