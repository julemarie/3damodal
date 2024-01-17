from aisformer_orig import AISFormer
from pointpillars.model import PointPillars
from pointpillars.utils import keep_bbox_from_image_range
import torch
import torch.nn as nn
from collections import OrderedDict

from torchvision.ops import roi_align
from torchvision.models import resnet50


class AISFormerLiDAR(nn.Module):
    def __init__(self, devices, cfg):
        super(AISFormerLiDAR, self).__init__()

        self.cfg = cfg
        self.gpu_id = devices[0]


        self.aisformer = AISFormer(devices=devices, cfg=cfg)
        self.load_aisformer()
        self.aisformer.train()

    def load_aisformer(self):
        if self.cfg.START_FROM_CKPT:
            state_dict = torch.load(self.cfg.CKPT_PATH)
            new_state_dict = OrderedDict()
            for key in state_dict['model'].keys():
                if "mask_head_model" in key:
                    if "predictor" not in key:
                        new_state_dict[key.replace("roi_heads.mask_head.mask_head_model.", "")] = state_dict['model'][key]
            self.aisformer.load_state_dict(new_state_dict)

    def forward(self, x):
        return self.aisformer(x)
    

class BackbonePP(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.roi_out_size = cfg.MODEL.PARAMS.ROI_OUT_SIZE
        self.img_height = cfg.MODEL.PARAMS.INPUT_H
        self.img_width = cfg.MODEL.PARAMS.INPUT_W
        self.cfg = cfg

        if self.cfg.DEVICE == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else: self.device = 'cpu'

        # setup feature encoder
        resnet = resnet50(pretrained=True)
        self.feat_encoder = nn.ModuleList([
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        ])
        if self.device == "cuda":
            self.feat_encoder = self.feat_encoder.cuda() 
        self.feat_encoder.eval()
        for param in self.feat_encoder.parameters():
            param.requires_grad = False

        # setup backbone
        self.backbone = PointPillars()
        if self.device == "cuda":
            self.backbone = self.backbone.cuda()
        self.backbone.load_state_dict(torch.load(cfg.MODEL.PATH))
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        lidar = x["lidar"]
        features, _ = x["amodal"]
        if self.device == "cuda":
            features = features.to("cuda")
        # predict 3d bboxes
        x_pts = lidar["batched_pts"]
        if self.device == "cuda":
            x_cuda = []
            for batch in x_pts:
                batch = torch.tensor(batch).to('cuda')
                x_cuda.append(batch)
            x_pts = x_cuda
        bbox_3d = self.backbone(x_pts)

        # roi align
        thresh = 0.9
        roi_list = []
        for i, result in enumerate(bbox_3d):
            # transform bboxes to correct format
            calib = lidar['batched_calib_info'][i]
            result = keep_bbox_from_image_range(result,
                                                calib['Tr_velo_to_cam'],
                                                calib['R0_rect'],
                                                calib['P2'],
                                                (self.cfg.MODEL.PARAMS.INPUT_H, self.cfg.MODEL.PARAMS.INPUT_W))
            bbox_2d = torch.tensor(result['bboxes2d'], device=self.device)
            score = torch.tensor(result['scores'])
            boxes = bbox_2d[torch.where(score > thresh)]
            
            id = i*torch.ones((boxes.shape[0]), dtype=torch.int, device=self.device).unsqueeze(-1)
            roi_list.append(torch.cat((id, boxes), dim=1))

        if roi_list:
            # pred bbs: [xmin, ymin, xmax, ymax]
            rois = torch.cat(roi_list, dim=0).float() # [K, 5] K: #bboxes, 5: first for img id and last four for corners
        else:
            return None, None
        
        for l in self.feat_encoder:
            features = l(features)

        f_roi = roi_align(features, rois, output_size=(self.cfg.MODEL.PARAMS.ROI_OUT_SIZE, self.cfg.MODEL.PARAMS.ROI_OUT_SIZE), spatial_scale=0.25, sampling_ratio=-1,aligned=True) # [K, C, H_r, W_r]
        if f_roi.shape[0] == 0:
            return None, None

        return f_roi, rois
