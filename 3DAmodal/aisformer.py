import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
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
        # self.backbone = fasterrcnn_resnet50_fpn(pretrained=True)
        # params = self.backbone.state_dict()
        # for name, param in self.backbone.state_dict().items():
        #     print(name)
        # self.backbone.eval()
        # # ROIAlign takes images and ROIs as input and outputs a tensor of size [K, in_chans, (OUT_SIZE)] where K = #bboxes
        # self.roi_align = RoIAlign(output_size=(self.roi_out_size, self.roi_out_size), spatial_scale=3, sampling_ratio=2,aligned=True)
        # deconvolution 
        self.deconv = nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(2,2), stride=2)
        self.conv = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(1,1), stride=1)
        self.encoder = Encoder(self.embed_dim, (self.roi_out_size*2)**2, num_heads=self.encoder_num_heads)
        self.decoder = Decoder(self.embed_dim, Q_dim=self.Q_dim, num_heads=self.decoder_num_heads)
        self.mlp = nn.ModuleList([
            nn.Linear(in_features=2*self.embed_dim, out_features=self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim)
        ])
        self.unflatten = nn.Unflatten(2, (self.roi_out_size*2, self.roi_out_size*2))


    def init_weights(self):
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.deconv.bias, 0)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)


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
    
    def min_max_normalize(self, vector, dim=1):
        min_vals, _ = torch.min(vector, dim=dim, keepdim=True)
        max_vals, _ = torch.max(vector, dim=dim, keepdim=True)

        # Check if the range is zero
        range_nonzero = (max_vals - min_vals) != 0

        # Apply normalization only when the range is nonzero
        normalized_vector = torch.where(range_nonzero, (vector - min_vals) / (max_vals - min_vals), vector)

        # If the range is zero, set the normalized vector to one
        normalized_vector = torch.where(~range_nonzero, torch.ones_like(vector), normalized_vector)

        return normalized_vector
        
    def forward(self, x):
        B, C, H, W = x.shape

        f_roi1 = self.conv(self.deconv(x)) # [K, C, H_m, W_m]
        f_roi1 = f_roi1.flatten(2).permute(0,2,1) # [K, H_m*W_m, C]
        f_e = self.encoder(f_roi1) # [K, H_m*W_m, C]
        Q = self.decoder(f_e) # [3, C]
        Q = Q.moveaxis(0,1) # [C, 3]
        Q_vis_am = torch.cat((Q[:,1], Q[:,2]), dim=0)
        I_i = Q_vis_am
        for i, layer in enumerate(self.mlp):
            I_i = F.gelu(layer(I_i)) if i < len(self.mlp)-1 else layer(I_i)  # [C]
        I_i = I_i.unsqueeze(-1)
        roi_embed = f_roi1 + f_e # [K, H_m*W_m, C]
        roi_embed = self.unflatten(roi_embed.permute(0,2,1)) # [K, C, H_m, W_m]
        masks = torch.tensordot(roi_embed, torch.cat((Q, I_i), dim=1), dims=([1], [0])).permute(0,3,1,2)
        # min-max normalization between 0 and 1
        masks = masks.reshape(masks.shape[0]*masks.shape[1], -1)
        masks = self.min_max_normalize(masks, dim=1)
        masks = masks.reshape(x.shape[0], 4, self.roi_out_size*2, self.roi_out_size*2)
        # binary thresholding
        masks[masks < 0.5] = 0
        masks[masks >= 0.5] = 1
        return masks # [K, 4, H_m, W_m]


class Encoder(nn.Module):
    """
        Transformer encoder with one block of self-attention.
    """
    def __init__(self, embed_dim, seq_len, num_heads):
        super().__init__()

        self.pos_encod = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.attention = MultiHeadAttention(embed_dim, embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # flatten feature vectors
        # x = x.flatten(2).permute(0,2,1) # [K, H_m*W_m, C]
        
        # x shape: [K, N, embed_dim], N: num_patches

        # add positional encoding
        x = x + self.pos_encod
        # self attention layer and add x
        x = x + self.attention(input_q=x, input_kv=x)
        # apply layer norm and linear layer
        x = self.ffn(self.norm1(x))
        # apply layer norm
        x = self.norm2(x)
        
        return x
    
class Decoder(nn.Module):
    """
        Transformer decoder with one block of self-attention and one block of cross-attention.
    """
    def __init__(self, embed_dim, Q_dim, num_heads):
        super().__init__()

        self.Q = nn.Parameter(torch.ones((Q_dim, embed_dim)).unsqueeze(0))

        self.attention = MultiHeadAttention(embed_dim, embed_dim, num_heads)
        self.norm_self = nn.LayerNorm(embed_dim)
        
        self.cross_attention = MultiHeadAttention(embed_dim,embed_dim, num_heads)
        self.norm_cross1 = nn.LayerNorm(embed_dim)
        self.ffn_cross1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm_cross2 = nn.LayerNorm(embed_dim)
        self.ffn_cross2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        K, N, D = x.shape
        Q = self.Q.repeat(K, 1, 1)
        Q = Q + self.attention(input_q=Q, input_kv=Q)
        Q = self.norm_self(Q)

        Q = Q + self.cross_attention(input_q=Q, input_kv=x)
        Q = self.norm_cross1(Q)
        Q = Q + self.ffn_cross1(Q)
        Q = self.ffn_cross2(self.norm_cross2(Q))

        Q = Q[0, :, :].squeeze(0)

        return Q



def test():
    encoder = Encoder(256, 28*28, 8)
    decoder = Decoder(256, 3, 8)
    x = torch.randn((1, 28*28, 256))
    x = encoder(x)
    Q = decoder(x)


if __name__ == "__main__":
    test()
    exit()
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
    backbone = fasterrcnn_resnet50_fpn(pretrained=True)
    backbone.eval()

    # img = img.unsqueeze(0)
    thresh = 0.9
    x = backbone(img)
    roi_list = []
    for i, ann in enumerate(x):
        boxes = ann["boxes"][torch.where(ann["scores"] > thresh)]
        id = i*torch.ones((boxes.shape[0]), dtype=torch.int).unsqueeze(-1)
        roi_list.append(torch.cat((id, boxes), dim=1))

    if roi_list:
        rois = torch.cat(roi_list, dim=0) # [K, 5] K: #bboxes, 5: first for id and last four for corners
    else:
        pass



    resnet = resnet50(pretrained=True)
    resnet.eval()
    
    out = resnet.conv1(img)
    out = resnet.bn1(out)
    out = resnet.relu(out)
    out = resnet.maxpool(out)
    out = resnet.layer1(out)

    out = roi_align(out, rois, output_size=(cfg.MODEL.PARAMS.ROI_OUT_SIZE, cfg.MODEL.PARAMS.ROI_OUT_SIZE), spatial_scale=0.25, aligned=True)



    out = resnet.layer2(out)
    out = resnet.layer3(out)
    out = resnet.layer4(out)
    out = resnet.avgpool(out)
    out = torch.flatten(out, 1)
    out = resnet.fc(out)
    print(out.shape)




    aisformer = AISFormer(in_shape=in_shape, cfg=cfg)
    
    masks = aisformer(imgs) 
    
    print(masks.shape)
    