from aisformer_orig import AISFormer
from pointpillars.model import PointPillars
from pointpillars.utils import keep_bbox_from_image_range, bbox3d2corners
from pointpillars.utils.process import group_rectangle_vertexs, group_plane_equation
import torch
from torch.nn import Module, Sequential, Conv2d
from collections import OrderedDict

from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import LastLevelMaxPool

from omegaconf import OmegaConf

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from upsampling import pcd_limit_range, points_in_bboxes, map_pc_to_img_space


class AISFormerLiDAR(Module):
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
    

class BackbonePP(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.feature_names = self.cfg.MODEL.BACKBONE.FPN.IN_FEATURES
        self.fpn_out_channels = self.cfg.MODEL.BACKBONE.FPN.OUT_CHANNELS
        self.bb_threshold = self.cfg.MODEL.BACKBONE.BB_THRESHOLD
        self.img_size = (self.cfg.MODEL.PARAMS.INPUT_H, self.cfg.MODEL.PARAMS.INPUT_W)
        self.feature_dims = self.cfg.MODEL.BACKBONE.FEATURE_DIMS
        self.roi_size = self.cfg.MODEL.PARAMS.ROI_OUT_SIZE

        # setup feature encoder
        resnet = resnet50(pretrained=True)

        self.res1 = Sequential(*list(resnet.children())[:5])
        self.res1[0] = Conv2d(4,64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.res2 = Sequential(*list(resnet.children())[5:6])
        self.res3 = Sequential(*list(resnet.children())[6:7])
        self.res4 = Sequential(*list(resnet.children())[7:8])

        self.fpn = FeaturePyramidNetwork(self.feature_dims, self.fpn_out_channels, extra_blocks=LastLevelMaxPool())

        self.multi_roi_align = MultiScaleRoIAlign(self.feature_names, self.roi_size, 2)

        self.init_fpn()

        # setup backbone
        self.bb_predictor = PointPillars()
        self.bb_predictor.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PREDICTOR.WEIGHTS))

    def init_fpn(self):
        state_dict = torch.load(self.cfg.CKPT_PATH)
        with torch.no_grad():
            self.fpn.inner_blocks[0].weight.copy_(state_dict['model']['backbone.fpn_lateral2.weight'])
            self.fpn.inner_blocks[0].bias.copy_(state_dict['model']['backbone.fpn_lateral2.bias'])
            self.fpn.inner_blocks[1].weight.copy_(state_dict['model']['backbone.fpn_lateral3.weight'])
            self.fpn.inner_blocks[1].bias.copy_(state_dict['model']['backbone.fpn_lateral3.bias'])
            self.fpn.inner_blocks[2].weight.copy_(state_dict['model']['backbone.fpn_lateral4.weight'])
            self.fpn.inner_blocks[2].bias.copy_(state_dict['model']['backbone.fpn_lateral4.bias'])
            self.fpn.inner_blocks[3].weight.copy_(state_dict['model']['backbone.fpn_lateral5.weight'])
            self.fpn.inner_blocks[3].bias.copy_(state_dict['model']['backbone.fpn_lateral5.bias'])

            self.fpn.layer_blocks[0].weight.copy_(state_dict['model']['backbone.fpn_output2.weight'])
            self.fpn.layer_blocks[0].bias.copy_(state_dict['model']['backbone.fpn_output2.bias'])
            self.fpn.layer_blocks[1].weight.copy_(state_dict['model']['backbone.fpn_output3.weight'])
            self.fpn.layer_blocks[1].bias.copy_(state_dict['model']['backbone.fpn_output3.bias'])
            self.fpn.layer_blocks[2].weight.copy_(state_dict['model']['backbone.fpn_output4.weight'])
            self.fpn.layer_blocks[2].bias.copy_(state_dict['model']['backbone.fpn_output4.bias'])
            self.fpn.layer_blocks[3].weight.copy_(state_dict['model']['backbone.fpn_output5.weight'])
            self.fpn.layer_blocks[3].bias.copy_(state_dict['model']['backbone.fpn_output5.bias'])

    def create_depthmask(self, pc, result, calib):
        pc = pc.cpu().numpy()
        # initialize depth mask the size of the image
        depth_mask = np.zeros((self.cfg.MODEL.PARAMS.INPUT_H, self.cfg.MODEL.PARAMS.INPUT_W))
        # retrieve bboxes, use those with high enough confidence
        bboxes3d = result['lidar_bboxes']
        scores = result['scores']
        valid_idx = torch.where(torch.tensor(scores) > self.bb_threshold)

        # identify points within bboxes
        masks = points_in_bboxes(pc[:, :3],                # (N, n), bool; N=#points, n=#bboxes
                                    group_plane_equation(              # plane equation parameters
                                        group_rectangle_vertexs(       # (n, 6, 4, 3)
                                            bbox3d2corners(bboxes3d)   # (n, 8, 3)
                                        )))
        
        # iterate over valid bboxes
        for idx in valid_idx[0]:
            points_in_box = np.array([pc[j] for j in range(len(pc)) if masks[j, idx] == True])
            points_in_box_img = map_pc_to_img_space(points_in_box, calib)

            # map x, y to pixels and resize to resized image shape
            x_img = np.floor(points_in_box_img[:, 0] * (self.cfg.MODEL.PARAMS.INPUT_W / self.cfg.MODEL.PARAMS.ORIG_W)).astype(np.int32)
            y_img = np.floor(points_in_box_img[:, 1] * (self.cfg.MODEL.PARAMS.INPUT_H / self.cfg.MODEL.PARAMS.ORIG_H)).astype(np.int32)

            z = points_in_box[:, 2] # z coords from pc space to keep depth info
            z_normed = (z - pcd_limit_range[2]) / (pcd_limit_range[5] - pcd_limit_range[2])

            # map x, y coordinates to 
            x_min, x_max = np.floor(np.min(x_img)).astype(np.int32), np.floor(np.max(x_img)).astype(np.int32)
            y_min, y_max = np.floor(np.min(y_img)).astype(np.int32), np.floor(np.max(y_img)).astype(np.int32)

            # upsample point cloud
            try:
                X = np.linspace(x_min, x_max, num=x_max-x_min)
                Y = np.linspace(y_min, y_max, num=y_max-y_min)
                Y, X = np.meshgrid(Y, X)
                interp = LinearNDInterpolator(list(zip(y_img, x_img)), z_normed)
                Z = interp(Y, X)
                Z = np.nan_to_num(Z, nan=0.0)
            except:
                continue

            if depth_mask[y_min:y_max, x_min:x_max].shape != Z.T.shape:
                continue # workarount for edge cases that do not align, not sure why

            depth_mask[y_min:y_max, x_min:x_max] = Z.T

        return np.array(depth_mask)
    
    def forward(self, x):
        lidar = x["lidar"]
        img, _ = x["amodal"]
        # predict 3d bboxes
        x_pts = lidar["batched_pts"]
        if self.cfg.DEVICE == "cuda":
            x_cuda = []
            for batch in x_pts:
                batch = torch.tensor(batch).to('cuda')
                x_cuda.append(batch)
            x_pts = x_cuda
        img = img.to(self.cfg.DEVICE)
        bbox_3d = self.bb_predictor(x_pts)

        if self.cfg.USE_DEPTH_FEATURE:
            depth_masks = []
        roi_list = []
        for i, result in enumerate(bbox_3d):
            # transform bboxes to correct format
            calib = lidar['batched_calib_info'][i]
            result = keep_bbox_from_image_range(result,
                                                calib['Tr_velo_to_cam'],
                                                calib['R0_rect'],
                                                calib['P2'],
                                                (self.cfg.MODEL.PARAMS.INPUT_H, self.cfg.MODEL.PARAMS.INPUT_W))
            bbox_2d = torch.tensor(result['bboxes2d'])
            score = torch.tensor(result['scores'])
            boxes = bbox_2d[torch.where(score > self.bb_threshold)]
            
            id = i*torch.ones((boxes.shape[0]), dtype=torch.int).unsqueeze(-1)
            roi_list.append(torch.cat((id, boxes), dim=1))
            if self.cfg.USE_DEPTH_FEATURE:
                depth_mask = self.create_depthmask(x_pts[i], result, calib)
                depth_masks.append(depth_mask)
        if self.cfg.USE_DEPTH_FEATURE:
            depth_masks = torch.tensor(np.array(depth_masks))

        
        # pred bbs: [xmin, ymin, xmax, ymax]
        rois = torch.cat(roi_list, dim=0).float() # [K, 5] K: #bboxes, 5: first for img id and last four for corners
        if rois.shape[0] == 0:
            return None, None
        
        rois = rois.to(self.cfg.DEVICE)
        bbs = rois[:,1:]

        # get depth mask as extra input channel
        #depth_masks = torch.tensor(self.create_depthmasks(x_pts, bbox_3d, calib))

        # resnet -- feature maps 
        feature_maps = {}
        if self.cfg.USE_DEPTH_FEATURE:
            feature_maps[self.feature_names[0]] = self.res1(torch.cat((img, depth_masks.unsqueeze(0).to(device=self.cfg.DEVICE, dtype=torch.float32)), dim=1))
        else:
            feature_maps[self.feature_names[0]] = self.res1(img)
        feature_maps[self.feature_names[1]] = self.res2(feature_maps[self.feature_names[0]])
        feature_maps[self.feature_names[2]] = self.res3(feature_maps[self.feature_names[1]])
        feature_maps[self.feature_names[3]] = self.res4(feature_maps[self.feature_names[2]])

        feature_maps = self.fpn(feature_maps)

        final_features = self.multi_roi_align(feature_maps, [bbs], [self.img_size])
        
        return final_features, rois
    


def test():
    cfg = OmegaConf.load('3DAmodal/configs/config.yaml')

    
