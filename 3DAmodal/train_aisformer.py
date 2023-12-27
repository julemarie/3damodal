from aisformer import AISFormer
# from datasets.asd_dataset import AmodalSynthDriveDataset
# from datasets.KINS_dataset import KINS
import sys
sys.path.append('/Midgard/home/tibbe/3damodal/3DAmodal/datasets')
from dataloader import get_dataloader
from utils import get_obj_from_str
import datasets.KINS_dataset
from torchvision.ops import roi_align
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from pycocotools import mask as coco_mask
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import cv2
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from scipy.optimize import linear_sum_assignment

# distributed training
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
 

def ddp_setup(rank, world_size):
    """
    Args:
        rank: uique identifier of each process
        world_size: total number of processes
    """
    # master is in charge of communication between the different processes
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "12355"

    # nccl: nvidia collective communications library -> distr. communication across cuda gpus
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer():
    def __init__(self, gpu_id, cfg, train=True):
        self.gpu_id = 'cpu' if gpu_id == -1 else 'cuda:{}'.format(gpu_id)
        self.cfg = cfg
        self.in_shape = tuple((self.cfg.MODEL.PARAMS.INPUT_H, self.cfg.MODEL.PARAMS.INPUT_W))

        # if self.cfg.DEVICE == 'cuda':
        #     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # else:
        #     self.device = 'cpu'

        self.save_interval = self.cfg.SAVE_INTERVAL

        self.backbone = fasterrcnn_resnet50_fpn(pretrained=True)
        # params = self.backbone.state_dict()
        # for name, param in self.backbone.state_dict().items():
        #     print(name)
        self.backbone.eval()
        # ROIAlign takes images and ROIs as input and outputs a tensor of size [K, in_chans, (OUT_SIZE)] where K = #bboxes
        # self.roi_align = RoIAlign(output_size=(self.cfg.MODEL.PARAMS.ROI_OUT_SIZE, self.cfg.MODEL.PARAMS.ROI_OUT_SIZE), spatial_scale=3, sampling_ratio=2,aligned=True)
        
        # init model and move it to device
        self.model = AISFormer(self.in_shape, self.cfg)
        self.model.to(self.gpu_id)
        if self.cfg.DISTRIBUTED:
            self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        # init dataloader
        if train:
            data_root = self.cfg.DATASET.TRAIN
            dataset = get_obj_from_str(self.cfg.DATASET.NAME)(data_root)
            # self.img_views = dataset.img_settings
        else:
            raise NotImplementedError

        self.dataloader = get_dataloader(dataset, batch_size=1, num_workers=5, partition="image", distributed=self.cfg.DISTRIBUTED)

        # self.resize = transforms.Resize((64, 64))
        self.roi_size = tuple((self.cfg.MODEL.PARAMS.ROI_OUT_SIZE*2, self.cfg.MODEL.PARAMS.ROI_OUT_SIZE*2))

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.LEARNING_RATE)

        self.epochs = self.cfg.EPOCHS

        if isinstance(dataset, datasets.KINS_dataset.KINS):
            self.mask_tf = transforms.Resize((250,828))

        # self.viewstr2int = {"front_full": 0,
        #                     "back_full": 1,
        #                     "left_full": 2,
        #                     "righ_full": 3,
        #                     "bev_full": 4}


    def crop_and_resize_mask(self, mask, roi):
        """
            Crop and resize a mask to match the format of the prediciton masks.

            Inputs:
                mask: Tensor [H, W]
                roi: Tensor [5]
        """
        x1, y1, x2, y2 = roi[1], roi[2], roi[3], roi[4]
        # draw bounding box on mask
        # pil_mask = Image.fromarray(mask.numpy())
        # ImageDraw.Draw(pil_mask).rectangle([x1, y1, x2, y2], outline=255)
        # mask_w_bb = torch.tensor(np.array(pil_mask), dtype=torch.float).unsqueeze(0)
        # save_image(mask_w_bb, "test_maskwbb.png")
        # crop
        cropped_mask = mask[y1:y2, x1:x2].unsqueeze(0)
        save_image(cropped_mask, "test_crop.png")
        # resize
        resized_mask = transforms.Resize(self.roi_size)(cropped_mask)
        save_image(resized_mask, "test_resizecrop.png")

        resized_mask = resized_mask.squeeze(0)

        return resized_mask
    

    def compute_loss(self, pred_mask, gt_mask):
        # compute binnary cross entropy loss
        loss = torch.nn.functional.binary_cross_entropy(pred_mask, gt_mask)
        # compute dice loss
        # intersection = (pred_mask.int() & gt_mask.int()).float().sum((0,1))
        # union = (pred_mask.int() | gt_mask.int()).float().sum((0,1))
        # loss = 1 - (2*intersection + 1e-6) / (union + 1e-6)                                    lm
        return loss

    def decode_mask_rle(self, mask_info):
        # Decode RLE-encoded mask
        decoded_mask = coco_mask.decode(mask_info)

        # Add channel dimension if needed
        binary_mask = torch.tensor(decoded_mask, dtype=torch.float).unsqueeze(0)

        return binary_mask
    
    def decode_mask_lst(self, polys, height, width):
        mask = np.zeros((height, width), dtype=np.int32)
        # mask = Image.new('L', (width, height), 0)
        
        for j ,poly in enumerate(polys):
            poly = [[poly[i], poly[i+1]] for i in range(0, len(poly), 2)]
            poly = np.array(poly, np.int32)
            poly = poly.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [poly], color=(255, 255, 255), lineType=cv2.LINE_AA)
    
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
        return mask


    def __get_mask_kins(self, ann):
        ann['amodal_full']['counts'] = ann['amodal_full']['counts']
        # 1) fetch necessary data from data and anns
        h, w = ann['img_height'], ann['img_width']
        amodal_mask = self.decode_mask_rle(ann['amodal_full'])
        seg_masks = []
        inmodal_masks = []
        for key in ann.keys():
            if key == "amodal_full" or key == "img_height" or key == "img_width":
                continue
            segmentation = self.decode_mask_lst(ann[key]['segmentation'], h, w)
            ann[key]['inmodal_seg']['counts'] = ann[key]['inmodal_seg']['counts']
            inmodal_seg = self.decode_mask_rle(ann[key]['inmodal_seg'])
            inmodal_masks.append(inmodal_seg)
            seg_masks.append(segmentation)
        seg_masks = torch.stack(seg_masks)
        inmodal_masks = torch.stack(inmodal_masks)
        amodal_mask = self.mask_tf(amodal_mask)
        seg_masks = self.mask_tf(seg_masks)
        inmodal_masks = self.mask_tf(inmodal_masks)
        save_image(inmodal_seg[0], "test_inmod.png")
        save_image(amodal_mask, "test_amod.png")
        save_image(segmentation[0], "test_seg.png")
        # combined_seg = torch.einsum('nchw->chw', seg_masks)
        # combined_inmodal = torch.einsum('nchw->chw', inmodal_masks)

        return inmodal_masks, seg_masks, amodal_mask
    
    def get_masks_kins(self, anns):
        inmodal_masks_all = []
        seg_masks_all = []
        amodal_masks_all = []
        for ann in anns:
            inmodal_masks, seg_masks, amodal_mask = self.__get_mask_kins(ann)
            inmodal_masks_all.append(inmodal_masks)
            seg_masks_all.append(seg_masks)
            amodal_masks_all.append(amodal_mask)

        return inmodal_masks_all, seg_masks_all, amodal_masks_all
    
    def __get_mask_kins2020(self, ann):
        # 1) fetch necessary data from data and anns
        h, w = ann['img_height'], ann['img_width']
        amodal_masks = []
        inmodal_masks = []
        all_amodal = torch.zeros((1, h, w))
        all_inmodal = torch.zeros((1, h, w))
        for key in ann.keys():
            if key == "img_height" or key == "img_width":
                continue
            amodal_mask = self.decode_mask_lst(ann[key]['a_segm'], h, w)
            inmodal_mask = self.decode_mask_lst(ann[key]['i_segm'], h, w)
            inmodal_masks.append(inmodal_mask)
            amodal_masks.append(amodal_mask)
            all_amodal += amodal_mask
            all_inmodal += inmodal_mask
        amodal_masks = torch.stack(amodal_masks)
        inmodal_masks = torch.stack(inmodal_masks)
        amodal_masks = self.mask_tf(amodal_masks)
        inmodal_masks = self.mask_tf(inmodal_masks)
        
        # for i in range(amodal_masks.shape[0]):
        #     save_image(amodal_masks[i], "test_amod_{}.png".format(i))
        #     save_image(inmodal_masks[i], "test_inmod_{}.png".format(i))
        # combined_seg = torch.einsum('nchw->chw', seg_masks)
        # combined_inmodal = torch.einsum('nchw->chw', inmodal_masks)

        return inmodal_masks, amodal_masks
    
    def get_masks_kins2020(self, anns):
        inmodal_masks_all = []
        amodal_masks_all = []
        for ann in anns:
            inmodal_masks, amodal_masks = self.__get_mask_kins2020(ann)
            # min max normalize between 0 and 1
            inmodal_masks = inmodal_masks.view(inmodal_masks.shape[0]*inmodal_masks.shape[1], -1)
            inmodal_masks -= inmodal_masks.min(1, keepdim=True)[0]
            inmodal_masks /= inmodal_masks.max(1, keepdim=True)[0]
            inmodal_masks = inmodal_masks.view(inmodal_masks.shape[0], 1, self.in_shape[0], self.in_shape[1])
            amodal_masks = amodal_masks.view(amodal_masks.shape[0]*amodal_masks.shape[1], -1)
            amodal_masks -= amodal_masks.min(1, keepdim=True)[0]
            amodal_masks /= amodal_masks.max(1, keepdim=True)[0]
            amodal_masks = amodal_masks.view(amodal_masks.shape[0], 1, self.in_shape[0], self.in_shape[1])
            # amodal_masks = (amodal_masks - amodal_masks.min()) / (amodal_masks.max() - amodal_masks.min())            
            inmodal_masks_all.append(inmodal_masks)
            amodal_masks_all.append(amodal_masks)

        return inmodal_masks_all, amodal_masks_all
    
    def calculate_IOU(self, mask1, mask2):
        """
            Calculate IOU between two given masks.
        """
        # binarize masks
        mask2[mask2 < 0.5] = 0
        mask2[mask2 >= 0.5] = 1
        save_image(mask1, "test_mask1_pred.png")
        save_image(mask2, "test_mask2_gt.png")
        intersection = (mask1.int() & mask2.int()).float().sum((0,1))
        union = (mask1.int() | mask2.int()).float().sum((0,1))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou

    def calculate_IOU_matrix(self, pred_masks, gt_masks, bbs):
        """
            Calculate IOU matrix between prediction masks and GT masks.
            Input:
                pred_masks: Tensor [K, 1, 64, 64]
                gt_masks: Tensor [K_gt, 1, 540, 960] (KINS: [250,  828])
                bbs: Tensor [K, 5]
                full_mask: bool (if True, shape of gt_masks: [4, 3, 540, 960])
            Output:
                iou_matrix: Tensor [K, K_gt]
        """
        num_pred = pred_masks.shape[0]
        num_gt = gt_masks.shape[0]
        iou_matrix = torch.zeros((num_pred, num_gt))
        for i in range(num_pred):
            for j in range(num_gt):
                # if gt_masks[j, 1, 0, 0] == bbs[i, 0]: # check if gt mask is in same image as prediction
                gt_mask_roi = self.crop_and_resize_mask(gt_masks[j,0], bbs[i])
                # save_image(pred_masks[i], "test_pred.png")
                iou_matrix[i, j] = self.calculate_IOU(pred_masks[i], gt_mask_roi)
                # test =  self.calculate_IOU(gt_mask_roi, torch.ones_like(gt_mask_roi))
                # else:
                #     iou_matrix[i, j] = 0.0

        return iou_matrix
    
    def assign_iou(self, pred_masks, gt_masks,  bbs, iou_threshold=0.5):
        """
            Assign pred masks to GT masks for loss calculation using IOU matrix and the Hungarian algorithm for optimal assignment.
        """
        iou_matrix = self.calculate_IOU_matrix(pred_masks, gt_masks, bbs)

        cost_matrix = 1 - iou_matrix
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        assigned_masks = {}
        for i, j in zip(row_indices, col_indices):
            if iou_matrix[i, j] >= iou_threshold:
                assigned_masks[i] = j # assign pred i to gt j
            else:
                assigned_masks[i] = 'fp' # false positive if not above iou_thresh
            
        return assigned_masks
    
    def compute_losses(self, assignment, pred_masks, gt_masks, bbs):
        """
            Compute the loss for one type of mask (occluder, invisible, amodal).

            Inputs:
                assignment: dict (dict containing mapping between corresponding masks)
                pred_masks: Tensor [K, 1, 64, 64]
                gt_masks: Tensor [K_gt, 1, 540, 960] (KINS: [250, 828])
                bbs: Tensor [K, 5]
        """
        loss = 0
        for i, j in assignment.items():
            if j == 'fp': # if false positive
                zero_mask = torch.zeros((pred_masks[i].shape), device=self.gpu_id)
                loss += self.compute_loss(pred_masks[i], zero_mask)
            else:
                gt_mask = self.crop_and_resize_mask(gt_masks[j,0], bbs[i])
                # compare = torch.cat((pred_masks[i], gt_mask), dim=0).unsqueeze(0)
                # save_image(compare, "test_compare.png")
                loss += self.compute_loss(pred_masks[i], gt_mask)
        return loss
    
    def compute_loss_vis(self, pred_masks, gt_masks, bbs):
        """
            Compute the loss for the visible masks (need to be treated differently because the input is different).

            Inputs:
                pred_masks: Tensor [K, 1, 64, 64]
                gt_masks: Tensor [4, 3, 540, 960]
                bbs: Tensor [K, 5]
        """
        loss = 0
        for i in range(bbs.shape[0]):
            gt_mask = self.crop_and_resize_mask(gt_masks[bbs[i, 0],0], bbs[i])
            loss += self.compute_loss(pred_masks[i], gt_mask)

        return loss

    def train(self):
        print("Starting training on GPU:{}...".format(self.gpu_id))
        print(torch.cuda.current_device())
        # clear gpu memory
        # torch.cuda.empty_cache()
        for epoch_idx in tqdm(range(self.epochs)):
            # for data, anns in self.dataloader:
            for img_batch, anns in self.dataloader:
                print("Loaded se data")
                # 1) Extract masks from annotations
                gt_vis_masks, gt_amod_masks, gt_occl_masks, gt_invis_masks = anns[0]['visible'], anns[0]['amodal'], anns[0]['occluder'], anns[0]['invisible']
                gt_vis_masks = gt_vis_masks.to(self.gpu_id)
                gt_amod_masks = gt_amod_masks.to(self.gpu_id)
                gt_occl_masks = gt_occl_masks.to(self.gpu_id)
                gt_invis_masks = gt_invis_masks.to(self.gpu_id)
                # img_batch size: [B, 5, 3, 540, 960]
                img_batch = img_batch[:,:4] # we don't care about the BEV image here
                B, V, C, H, W = img_batch.shape
                print(img_batch.shape)
                img_batch = img_batch.reshape((B*V, C, H, W))
                img_batch = img_batch.to(self.gpu_id)
                
                # 2) pass images through model (output will be the four masks)
                masks, rois = self.model(img_batch) # [K, 4, 64, 64], [K, 5]
                # torch.cuda.empty_cache()
                rois = rois.to(torch.int).to('cpu')
                K, num_masks, H_m, W_m = masks.shape
                m_o = masks[:, 0]
                m_v = masks[:, 1]
                m_a = masks[:, 2]
                m_i = masks[:, 3]
                
                # 3) calculate the loss
                # 3.1) assign predicted masks to gt masks using IOU matrix and Hungarian algorithm
                # for occluder masks:
                assigned_masks_o = self.assign_iou(m_o, gt_occl_masks, bbs=rois)
                assigned_masks_a = self.assign_iou(m_a, gt_amod_masks, bbs=rois)
                assigned_masks_i = self.assign_iou(m_i, gt_invis_masks, bbs=rois)

                # 3.2) calculate the losses
                loss_o = self.compute_losses(assigned_masks_o, m_o, gt_occl_masks, bbs=rois)
                loss_a = self.compute_losses(assigned_masks_a, m_a, gt_amod_masks, bbs=rois)
                loss_i = self.compute_losses(assigned_masks_i, m_i, gt_invis_masks, bbs=rois)
                loss_v = self.compute_loss_vis(m_v, gt_vis_masks, rois)
                
                # 3.3) accumulate the losses
                total_loss = loss_o + loss_a + loss_i + loss_v

                # torch.cuda.empty_cache()

                # 4) update the weights (for this we need to define an optimizer -> check paper)
                self.optimizer.zero_grad()
                print("BACKWARD")
                total_loss.backward()
                print("BACHWARD DONE")
                self.optimizer.step()

                # torch.cuda.empty_cache()

                # 5) start training and save masks, feature tensors, and attention maps for visualisation (check paper what they used)

            print("Loss after epoch {}: {}".format(epoch_idx, total_loss))
    
    def train_kins(self):
        self.model.train()
        # model_dict = self.model.state_dict()
        # for param, name in zip(self.model.parameters(), model_dict.keys()):
        #     print("Before {}: {}".format(name, param.requires_grad))
        #     param.requires_grad = True
        #     print("After {}: {}".format(name, param.grad_fn))
        # for param, name in zip(self.backbone.parameters(), self.backbone.state_dict().keys()):
        #     print("{}: {}".format(name, param.requires_grad))
        # model_dict = self.model.state_dict()
        print("Starting training on GPU:{}...".format(self.gpu_id))
        # print(torch.cuda.current_device())
        # clear gpu memory
        # torch.cuda.empty_cache()
        for epoch_idx in tqdm(range(self.epochs)):
            # for data, anns in self.dataloader:
            for img_batch, anns in tqdm(self.dataloader):
                # 1) Extract masks from annotations
                """
                anns: list[dict]
                dict: {inst: {Segmentation(polygons),
                              Area,
                              inmodal_seg({counts, size}),
                              iscrowd,
                              image_id,
                              bbox,
                              inmodal_bbox,
                              catgory_id,
                              id},
                        amodal_full({counts, size}),
                        img_height,
                        img_width}
                """
                gt_inmodal_masks, gt_amodal_masks = self.get_masks_kins2020(anns)
                gt_invis_masks = [a-i for a, i in zip(gt_amodal_masks, gt_inmodal_masks)]

                # move gt masks to gpu
                gt_inmodal_masks = [m.to(self.gpu_id) for m in gt_inmodal_masks]
                gt_amodal_masks = [m.to(self.gpu_id) for m in gt_amodal_masks]
                gt_invis_masks = [m.to(self.gpu_id) for m in gt_invis_masks]

                # img_batch size: [B, 3, 540, 960] (ASD), [B, 3, 250, 828] (KINS)
                img_batch = img_batch.to(self.gpu_id)
                save_image(img_batch[0], "img_0.png")

                # self.optimizer.zero_grad()
                
                # 2) pass images through model (output will be the four masks)
                thresh = 0.9
                x = self.backbone(img_batch)
                roi_list = []
                for i, ann in enumerate(x):
                    boxes = ann["boxes"][torch.where(ann["scores"] > thresh)]
                    id = i*torch.ones((boxes.shape[0]), dtype=torch.int, device=self.gpu_id).unsqueeze(-1)
                    roi_list.append(torch.cat((id, boxes), dim=1))

                if roi_list:
                    rois = torch.cat(roi_list, dim=0) # [K, 5] K: #bboxes, 5: first for id and last four for corners
                else:
                    continue
                
                # draw bounding box on image
                # img = img_batch[0].permute(1,2,0).cpu().numpy()
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # for roi in rois:
                #     x1, y1, x2, y2 = roi[1].item(), roi[2].item(), roi[3].item(), roi[4].item()
                #     start_point = (int(x1), int(y1))
                #     end_point = (int(x2), int(y2))
                #     color = (255, 0, 0)
                #     thickness = 1
                #     img = cv2.rectangle(img, start_point, end_point, color, thickness)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = transforms.ToTensor()(img)
                # save_image(img, "img_2.png")

                f_roi = roi_align(img_batch, rois, output_size=(self.cfg.MODEL.PARAMS.ROI_OUT_SIZE, self.cfg.MODEL.PARAMS.ROI_OUT_SIZE), spatial_scale=1, sampling_ratio=-1,aligned=True) # [K, C, H_r, W_r]
                # normalize
                # mean = [0.485, 0.456, 0.406]
                # std = [0.229, 0.224, 0.225]
                # f_roi = transforms.Normalize(mean, std)(f_roi)
                save_image(f_roi[0], "img_1.png")
                masks = self.model(f_roi) # [K, 4, 64, 64], [K, 5]
                if masks is None:
                    print("No masks")
                    continue
                
                # torch.cuda.empty_cache()
                rois = rois.to(torch.int).to('cpu')
                K, num_masks, H_m, W_m = masks.shape
                m_o = masks[:, 0] # occluder mask (no gt in KINS)
                m_v = masks[:, 1] # visible mask (inmodal)
                m_a = masks[:, 2] # amodal mask
                m_i = masks[:, 3] # invisible mask (amodal - inmodal)

                save_image(m_v[0], "test_v.png")
                save_image(m_v[1], "test_v1.png")
                
                # 3) calculate the loss
                # 3.1) assign predicted masks to gt masks using IOU matrix and Hungarian algorithm
                assigned_masks_a = self.assign_iou(m_a, gt_amodal_masks[0], bbs=rois)
                assigned_masks_v = self.assign_iou(m_v, gt_inmodal_masks[0], bbs=rois)
                assigned_masks_i = self.assign_iou(m_i, gt_invis_masks[0], bbs=rois)

                # 3.2) calculate the losses
                # loss_o = self.compute_losses(assigned_masks_o, m_o, gt_occl_masks, bbs=rois)
                loss_a = self.compute_losses(assigned_masks_a, m_a, gt_amodal_masks[0], bbs=rois)
                loss_v = self.compute_losses(assigned_masks_v, m_v, gt_inmodal_masks[0], bbs=rois)
                loss_i = self.compute_losses(assigned_masks_i, m_i, gt_invis_masks[0], bbs=rois)
                
                # 3.3) accumulate the losses
                total_loss = loss_a + loss_v + loss_i

                # torch.cuda.empty_cache()

                # 4) backpropagate the loss
                self.optimizer.zero_grad()
                total_loss.backward()
                # print gradients
                for param in self.model.parameters():
                    print(param.grad)
                self.optimizer.step()

                # torch.cuda.empty_cache()

                # 5) start training and save masks, feature tensors, and attention maps for visualisation (check paper what they used)

            print("Loss after epoch {}: {}".format(epoch_idx, total_loss))

def main(rank: int, world_size: int, cfg: OmegaConf):
    # ddp_setup(rank, world_size)
    trainer = Trainer(gpu_id=rank, cfg=cfg)
    trainer.train_kins()
    # destroy_process_group()


if __name__ == "__main__":
    cfg_path="/Midgard/home/tibbe/3damodal/3DAmodal/configs/config.yaml"
    cfg = OmegaConf.load(cfg_path)
    world_size = torch.cuda.device_count()

    main(-1, world_size, cfg)

    # mp.spawn(main, args=(world_size, cfg), nprocs=world_size)