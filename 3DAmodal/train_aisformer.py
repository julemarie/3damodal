from aisformer_pp import AISFormerLiDAR, BackbonePP
# from datasets.asd_dataset import AmodalSynthDriveDataset
# from datasets.KINS_dataset import KINS
import sys
#sys.path.append('/Midgard/home/tibbe/3damodal/3DAmodal/datasets')
from datasets.dataloader import get_dataloader
from datasets.KINS_kitti_dataset import AmodalLiDAR
from utils import get_obj_from_str
import datasets.KINS_dataset
from torchvision.ops import roi_align
from torchvision.models import resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from pycocotools import mask as coco_mask
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import cv2
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from scipy.optimize import linear_sum_assignment

# distributed training
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from collections import OrderedDict

from pointpillars.utils import keep_bbox_from_image_range
import logging
import traceback

# logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[logging.StreamHandler()]
#     )
# logger = logging.getLogger(__name__)

cfg_path="/home/jule-magnus/dd2414/merge_repo/3damodal/3DAmodal/configs/config.yaml"
cfg = OmegaConf.load(cfg_path)

LOG_LEVEL = cfg.LOG_LEVEL

def log(s):
    if LOG_LEVEL == "DEBUG":
        print(s)

# logging
from torch.utils.tensorboard import SummaryWriter
import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

RUNS_FOLDER = cfg.RUNS_FOLDER
CKP_FOLDER = cfg.SAVE_CKPT_FOLDER

if not os.path.exists(RUNS_FOLDER):
    os.mkdir(RUNS_FOLDER)
if not os.path.exists(CKP_FOLDER):
    os.mkdir(CKP_FOLDER)


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
    def __init__(self, cfg, device, multi_gpu=False, writer=SummaryWriter(), train=True):
    #def __init__(self, cfg, device, multi_gpu=False, train=True):
        self.writer = writer
        if device == -1:
            device = torch.device("cpu")
        self.cfg = cfg
        self.in_shape = tuple((self.cfg.MODEL.PARAMS.INPUT_H, self.cfg.MODEL.PARAMS.INPUT_W))

        self.save_interval = self.cfg.SAVE_INTERVAL

        self.backbone = BackbonePP(cfg)

        self.backbone.eval()
        self.backbone.to(device)
        # ROIAlign takes images and ROIs as input and outputs a tensor of size [K, in_chans, (OUT_SIZE)] where K = #bboxes
        
        # init model and move it to device
        log("BEFORE MODEL: {}".format(torch.cuda.memory_allocated(device)/1.074e+9))
        if multi_gpu:
            if cfg.DISTRIBUTED.MODEL:
                self.devices = [device*2, device*2+1]
            elif cfg.DISTRIBUTED.DATA:
                self.devices = [device, device]
        else:
            self.devices = [device, device]

        self.model = AISFormerLiDAR(self.devices, cfg)        

        if multi_gpu:
            if cfg.DISTRIBUTED.MODEL:
                self.model = DDP(self.model)
            elif cfg.DISTRIBUTED.DATA:
                self.model = DDP(self.model, device_ids=[device], output_device=device)

        log("AFTER MODEL: {}".format(torch.cuda.memory_allocated(device)/1.074e+9))
        
        # init dataloader
        if train:
            data_root = self.cfg.DATASET.TRAIN
            dataset = AmodalLiDAR(root_dir=data_root, mode="train")
        else:
            raise NotImplementedError

        log("BEFORE DATLOADER: {}".format(torch.cuda.memory_allocated(device)/1.074e+9))
        if multi_gpu:
            self.dataloader = get_dataloader(dataset, batch_size=self.cfg.BATCH_SIZE, num_workers=1, partition="combined", distributed=self.cfg.DISTRIBUTED.DATA)
        else:
            self.dataloader = get_dataloader(dataset, batch_size=self.cfg.BATCH_SIZE, num_workers=1, partition="combined", distributed=False)
        log("AFTER DATLOADER: {}".format(torch.cuda.memory_allocated(device)/1.074e+9))
        # self.resize = transforms.Resize((64, 64))
        self.roi_size = tuple((self.cfg.MODEL.PARAMS.ROI_OUT_SIZE*2, self.cfg.MODEL.PARAMS.ROI_OUT_SIZE*2))

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.LEARNING_RATE)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.cosine_decay_lr(epoch, self.cfg.LEARNING_RATE, 0.0025, self.cfg.EPOCHS))

        self.epochs = self.cfg.EPOCHS

        self.mask_tf = transforms.Resize(self.in_shape)


    def cosine_decay_lr(self, epoch, start_lr, end_lr, total_epochs):
        """
        Compute the learning rate using a cosine decay schedule.

        Inputs:
            epoch: Current epoch
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            total_epochs: Total number of epochs

        Returns:
            lr: Computed learning rate
        """
        cos_decay = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        lr = end_lr + 0.5 * (start_lr - end_lr) * cos_decay
        return lr

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
        # save_image(cropped_mask, "test_crop.png")
        # resize
        resized_mask = transforms.Resize(self.roi_size)(cropped_mask)
        # save_image(resized_mask, "test_resizecrop.png")

        resized_mask = resized_mask.squeeze(0)

        return resized_mask
    

    def compute_loss(self, pred_mask, gt_mask):
        """
            Compute the loss between a prediction mask and a ground truth mask.

            Inputs:
                pred_mask: Tensor [H, W]
                gt_mask: Tensor [H, W]
        """
        # compute binnary cross entropy loss
        loss = torch.nn.functional.binary_cross_entropy(pred_mask, gt_mask)
        # define weight as avg number of black pixels / avg number of all pixels
        # all_pixels = gt_mask.shape[0]*gt_mask.shape[1]
        # black_pixels = all_pixels - gt_mask.sum()
        # weight = black_pixels / all_pixels
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask, gt_mask, weight=weight)
        # compute cross entropy loss
        # loss = torch.nn.functional.cross_entropy(pred_mask, gt_mask)
        # compute dice loss
        # intersection = (pred_mask.int() & gt_mask.int()).float().sum((0,1))
        # union = (pred_mask.int() | gt_mask.int()).float().sum((0,1))
        # loss = 1 - (2*intersection + 1e-6) / (union + 1e-6)                                    lm
        return loss

    
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

    
    def __get_mask_kins2020(self, ann):
        # 1) fetch necessary data from data and anns
        h, w = ann['img_height'], ann['img_width']
        amodal_masks = []
        inmodal_masks = []
        bbs = []
        all_amodal = torch.zeros((1, h, w))
        all_inmodal = torch.zeros((1, h, w))
        for key in ann.keys():
            if key == "img_height" or key == "img_width":
                continue
            if ann[key]['a_segm'] != "":
                amodal_mask = self.decode_mask_lst(ann[key]['a_segm'], h, w)
                amodal_masks.append(amodal_mask)
                all_amodal += amodal_mask
            else:
                amodal_masks.append(torch.zeros((1, h, w)))
            if ann[key]['i_segm'] != "":
                inmodal_mask = self.decode_mask_lst(ann[key]['i_segm'], h, w)
                inmodal_masks.append(inmodal_mask)
                all_inmodal += inmodal_mask
            else:
                inmodal_masks.append(torch.zeros((1, h, w)))
            bbs.append(ann[key]['i_bbox'])
        
        amodal_masks = torch.stack(amodal_masks)
        inmodal_masks = torch.stack(inmodal_masks)
        amodal_masks = self.mask_tf(amodal_masks)
        inmodal_masks = self.mask_tf(inmodal_masks)
        
        # for i in range(amodal_masks.shape[0]):
        #     save_image(amodal_masks[i], "test_amod_{}.png".format(i))
        #     save_image(inmodal_masks[i], "test_inmod_{}.png".format(i))
        # combined_seg = torch.einsum('nchw->chw', seg_masks)
        # combined_inmodal = torch.einsum('nchw->chw', inmodal_masks)

        return inmodal_masks, amodal_masks, bbs
    
    def get_masks_kins2020(self, anns):
        inmodal_masks_all = []
        amodal_masks_all = []
        bbs_all = []
        for ann in anns:
            inmodal_masks, amodal_masks, bbs = self.__get_mask_kins2020(ann)
            bbs = torch.tensor(bbs)
            bbs_all.append(bbs)
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

        return inmodal_masks_all, amodal_masks_all, bbs_all
    
    def calculate_IOU(self, mask1, mask2):
        """
            Calculate IOU between two given masks.
        """
        # binarize masks
        mask2[mask2 < 0.5] = 0
        mask2[mask2 >= 0.5] = 1
        # save_image(mask1, "test_mask1_pred.png")
        # save_image(mask2, "test_mask2_gt.png")
        intersection = (mask1.int() & mask2.int()).float().sum((0,1))
        union = (mask1.int() | mask2.int()).float().sum((0,1))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou
    
    def calculate_IOU_bbs(self, bb1, bb2):
        """
            Calculate IOU between two given bounding boxes.

            Inputs:
                bb1: Tensor [4] (xmin, ymin, xmax, ymax)
                bb2: Tensor [4] (xmin, ymin, xmax, ymax)   
        """
        # calculate intersection
        x1_1, y1_1, x1_2, y1_2 = bb1[0], bb1[1], bb1[2], bb1[3]
        x2_1, y2_1, x2_2, y2_2 = bb2[0], bb2[1], bb2[2], bb2[3]
        x1 = max(x1_1, x2_1)
        y1 = max(y1_1, y2_1)
        x2 = min(x1_2, x2_2)
        y2 = min(y1_2, y2_2)
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # calculate union
        area1 = (x1_2 - x1_1 + 1) * (y1_2 - y1_1 + 1)
        area2 = (x2_2 - x2_1 + 1) * (y2_2 - y2_1 + 1)
        union = area1 + area2 - intersection

        iou = intersection / union

        return iou
        

    def calculate_IOU_matrix(self, pred_masks, gt_masks, pred_bbs, gt_bbs):
        """
            Calculate IOU matrix between prediction masks and GT masks.
            Input:
                pred_masks: Tensor [K, 1, 64, 64]
                gt_masks: Tensor [K_gt, 1, 540, 960] (KINS: [250,  828])
                pred_bbs: Tensor [K, 5] (img_id, xmin, ymin, xmax, ymax)
                gt_bbs: Tensor [K_gt, 4] (xmin, ymin, diffx, diffy)
            Output:
                iou_matrix: Tensor [K, K_gt]
        """
        num_pred = pred_masks.shape[0]
        num_gt = gt_masks.shape[0]
        iou_matrix = torch.zeros((num_pred, num_gt))
        for i in range(num_pred):
            for j in range(num_gt):
                # if gt_masks[j, 1, 0, 0] == bbs[i, 0]: # check if gt mask is in same image as prediction
                # gt_mask_roi = self.crop_and_resize_mask(gt_masks[j,0], bbs[i])
                # save_image(gt_mask_roi, "test_gt.png")
                # save_image(pred_masks[i], "test_pred.png")
                if pred_bbs is not None:
                    xmin, ymin, diffx, diffy = gt_bbs[j]
                    gt_bb = torch.tensor([xmin, ymin, xmin+diffx, ymin+diffy])
                    iou_matrix[i, j] = self.calculate_IOU_bbs(pred_bbs[i, 1:], gt_bb)
                else:
                    iou_matrix[i, j] = self.calculate_IOU(pred_masks[i], gt_masks[j,0])
                # iou_matrix[i, j] = self.calculate_IOU(pred_masks[i], gt_mask_roi)
                # test =  self.calculate_IOU(gt_mask_roi, torch.ones_like(gt_mask_roi))
                # else:
                #     iou_matrix[i, j] = 0.0

        return iou_matrix
    
    def assign_iou(self, pred_masks, gt_masks, pred_bbs, gt_bbs, iou_threshold=0.0):
        """
            Assign pred masks to GT masks for loss calculation using IOU matrix and the Hungarian algorithm for optimal assignment.
        """
        iou_matrix = self.calculate_IOU_matrix(pred_masks, gt_masks, pred_bbs, gt_bbs)

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
    
    def compute_losses(self, assignment, pred_masks, gt_masks, bbs, type):
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
                zero_mask = torch.zeros((pred_masks[i].shape), device=self.devices[1])
                loss += self.compute_loss(pred_masks[i], zero_mask)
            else:
                gt_mask = self.crop_and_resize_mask(gt_masks[j,0], bbs[i])
                compare = torch.cat((pred_masks[i], gt_mask), dim=0).unsqueeze(0)
                save_image(compare, "test_compare_{}.png".format(type))
                loss += self.compute_loss(pred_masks[i], gt_mask)
        return loss
 
    
    def normalize(self, x):
        """
            Normalize the input tensor between 0 and 1.

            Inputs:
                x: Tensor [K, H, W]
        """
        x = x.reshape(x.shape[0], -1)
        x -= x.min(1, keepdim=True)[0]
        x /= x.max(1, keepdim=True)[0]
        # x[x >= 0.5] = 1
        # x[x < 0.5] = 0
        x = x.reshape(x.shape[0], self.cfg.MODEL.PARAMS.ROI_OUT_SIZE*2, self.cfg.MODEL.PARAMS.ROI_OUT_SIZE*2)

        return x

    
    def train(self):
        self.model.train()
        self.backbone.train()
        self.backbone.bb_predictor.eval()
        print("Starting training on GPU:{}...".format(self.devices))
        # print(torch.cuda.current_device())
        # clear gpu memory
        # torch.cuda.empty_cache()
        log("BEFORE TRAIN: {}".format(torch.cuda.memory_allocated(self.devices[0])/1.074e+9))
        for epoch_idx in tqdm(range(self.epochs)):
            # for data, anns in self.dataloader:
            running_loss = 0.0
            running_loss_a = 0.0
            running_loss_v = 0.0
            running_loss_i = 0.0
            step = 0
            for batch in tqdm(self.dataloader):
                _, anns = batch["amodal"]
                # print("EFFECTIVE BATCH SIZE GPU{}: {}".format(self.devices[0],img_batch.shape[0]))
                log("AFTER DATA: {}".format(torch.cuda.memory_allocated(self.devices[0])/1.074e+9))
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
                out_dev = self.devices[1]
                gt_inmodal_masks, gt_amodal_masks, gt_bbs = self.get_masks_kins2020(anns)
                gt_invis_masks = [a-i for a, i in zip(gt_amodal_masks, gt_inmodal_masks)]

                # move gt masks to gpu
                # gt_bbs: [xmin, ymin, diffx, diffy]
                gt_bbs = [((b*2)//3).to('cpu') for b in gt_bbs] # resize bbs to match resized image
                gt_inmodal_masks = [m.to(out_dev) for m in gt_inmodal_masks]
                gt_amodal_masks = [m.to(out_dev) for m in gt_amodal_masks]
                gt_invis_masks = [m.to(out_dev) for m in gt_invis_masks]

                # img_batch size: [B, 3, 540, 960] (ASD), [B, 3, 250, 828] (KINS)
                
                # 2) pass images through model (output will be the four masks)

                log("AFTER ROI: {}".format(torch.cuda.memory_allocated(self.devices[0])/1.074e+9))
                torch.cuda.empty_cache()

                f_roi, rois = self.backbone(batch)
                if f_roi == None or rois == None:
                    continue
                m_v, m_o, m_a, m_i = self.model(f_roi) # [K, 1, 56, 56], [K, 1, 56, 56], [K, 1, 56, 56], [K, 1, 56, 56]
                log("BEFORE FW: {}".format(torch.cuda.memory_allocated(self.devices[0])/1.074e+9))
                #m_v, m_o, m_a, m_i = self.model(f_roi) # [K, 1, 56, 56], [K, 1, 56, 56], [K, 1, 56, 56], [K, 1, 56, 56]
                log("AFTER FW: {}".format(torch.cuda.memory_allocated(self.devices[0])/1.074e+9))
                m_v = m_v.squeeze(1)
                m_o = m_o.squeeze(1)
                m_a = m_a.squeeze(1)
                m_i = m_i.squeeze(1)

                # min max normalize between 0 and 1
                m_v = self.normalize(m_v)
                m_o = self.normalize(m_o)
                m_a = self.normalize(m_a)
                m_i = self.normalize(m_i)
                
                # torch.cuda.empty_cache()
                rois = rois.to(torch.int).to('cpu')
                
                # 3) calculate the loss
                # 3.1) assign predicted masks to gt masks using IOU matrix and Hungarian algorithm
                assigned_masks_a = self.assign_iou(m_a, gt_amodal_masks[0], pred_bbs=rois, gt_bbs=gt_bbs[0], iou_threshold=0.5)
                assigned_masks_v = self.assign_iou(m_v, gt_inmodal_masks[0], pred_bbs=rois, gt_bbs=gt_bbs[0], iou_threshold=0.5)
                assigned_masks_i = self.assign_iou(m_i, gt_invis_masks[0], pred_bbs=rois, gt_bbs=gt_bbs[0], iou_threshold=0.5)

                # 3.2) calculate the losses
                # loss_o = self.compute_losses(assigned_masks_o, m_o, gt_occl_masks, bbs=rois)
                loss_a = self.compute_losses(assigned_masks_a, m_a, gt_amodal_masks[0], bbs=rois, type="amodal")
                loss_v = self.compute_losses(assigned_masks_v, m_v, gt_inmodal_masks[0], bbs=rois, type="visible")
                loss_i = self.compute_losses(assigned_masks_i, m_i, gt_invis_masks[0], bbs=rois, type="invisible")
                
                # 3.3) accumulate the losses
                total_loss = loss_a + loss_v + loss_i

                # 4) backpropagate the loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item()
                running_loss_a += loss_a.item()
                running_loss_v += loss_v.item()
                running_loss_i += loss_i.item()

                self.writer.add_scalar("Training loss [batch]", total_loss, (epoch_idx)*len(self.dataloader)+step)
                loss_dict = {"Loss amodal [batch]": loss_a,
                             "Loss visible [batch]": loss_v,
                             "Loss invisible [batch]": loss_i}
                self.writer.add_scalars("Individual losses [batch]", loss_dict, (epoch_idx)*len(self.dataloader)+step)

                step += 1


            pseudo_mask_o = torch.zeros_like(m_o[0][None])
            if assigned_masks_a[0] != 'fp':
                gt_mask_a = self.crop_and_resize_mask(gt_amodal_masks[0][assigned_masks_a[0]].squeeze(0), rois[0])[None]
            else:
                gt_mask_a = pseudo_mask_o
            if assigned_masks_v[0] != 'fp':
                gt_mask_v = self.crop_and_resize_mask(gt_inmodal_masks[0][assigned_masks_v[0]].squeeze(0), rois[0])[None]
            else:
                gt_mask_v = pseudo_mask_o
            if assigned_masks_i[0] != 'fp':
                gt_mask_i = self.crop_and_resize_mask(gt_invis_masks[0][assigned_masks_i[0]].squeeze(0), rois[0])[None]
            else:
                gt_mask_i = pseudo_mask_o
            gt_masks = torch.stack((gt_mask_v, gt_mask_a, gt_mask_i, pseudo_mask_o), dim=0)
            masks = torch.cat((m_v[0][None,None], m_a[0][None,None], m_i[0][None,None], m_o[0][None,None]), dim=0)

            all_masks = torch.cat((gt_masks, masks), dim=2)
            self.writer.add_images("masks [v, a, i, o(no gt)]", all_masks, (epoch_idx+1)*len(self.dataloader), dataformats="NCHW")

            self.writer.add_scalar("Training loss [epoch]", running_loss/len(self.dataloader), (epoch_idx+1))
            loss_dict = {"Loss amodal [epoch]": running_loss_a/len(self.dataloader),
                            "Loss visible [epoch]": running_loss_v/len(self.dataloader),
                            "Loss invisible [epoch]": running_loss_i/len(self.dataloader)}
            self.writer.add_scalars("Individual losses [epoch]", loss_dict, (epoch_idx+1))

            self.scheduler.step()

            print("\nLoss after epoch {}: {:.2f}+{:.2f}+{:.2f} = {:.2f}\n".format(epoch_idx, running_loss_a/len(self.dataloader),
                                                                   running_loss_v/len(self.dataloader),
                                                                   running_loss_i/len(self.dataloader),
                                                                   running_loss/len(self.dataloader)))

            if self.devices[0] == 0: # only save model on gpu 0
                if epoch_idx % self.save_interval == 0 or epoch_idx == self.epochs-1:
                    torch.save(self.model.state_dict(), "{}/aisformer_{}.pth".format(CKP_FOLDER, epoch_idx))

        
    """def test_kins(self):
        self.model.eval()

        TP = 0
        FP = 0
        precision_lst = []
        recall_lst = []

        for n, (img_batch, anns) in enumerate(self.dataloader):
            out_dev = self.devices[1]
            gt_inmodal_masks, gt_amodal_masks, gt_bbs = self.get_masks_kins2020(anns)
            gt_invis_masks = [a-i for a, i in zip(gt_amodal_masks, gt_inmodal_masks)]

            # move gt masks to gpu
            # gt_bbs: [xmin, ymin, diffx, diffy]
            gt_bbs = [((b*2)//3).to('cpu') for b in gt_bbs] # resize bbs to match resized image
            gt_inmodal_masks = [m.to(out_dev) for m in gt_inmodal_masks]
            gt_amodal_masks = [m.to(out_dev) for m in gt_amodal_masks]
            gt_invis_masks = [m.to(out_dev) for m in gt_invis_masks]

            thresh = 0.9
            x = self.backbone(img_batch)
            roi_list = []
            for i, ann in enumerate(x):
                boxes = ann["boxes"][torch.where(ann["scores"] > thresh)]
                id = i*torch.ones((boxes.shape[0]), dtype=torch.int, device='cpu').unsqueeze(-1)
                roi_list.append(torch.cat((id, boxes), dim=1))

            if roi_list:
                # pred bbs: [xmin, ymin, xmax, ymax]
                rois = torch.cat(roi_list, dim=0) # [K, 5] K: #bboxes, 5: first for img id and last four for corners
            else:
                continue
            
            feats = img_batch
            for l in self.feat_encoder:
                feats = l(feats)

            f_roi = roi_align(feats, rois, output_size=(self.cfg.MODEL.PARAMS.ROI_OUT_SIZE, self.cfg.MODEL.PARAMS.ROI_OUT_SIZE), spatial_scale=0.25, sampling_ratio=-1,aligned=True) # [K, C, H_r, W_r]
            if f_roi.shape[0] == 0:
                continue
            
            m_v, m_o, m_a, m_i = self.model(f_roi) # [K, 1, 56, 56], [K, 1, 56, 56], [K, 1, 56, 56], [K, 1, 56, 56]
            
            m_v = m_v.squeeze(1)
            m_o = m_o.squeeze(1)
            m_a = m_a.squeeze(1)
            m_i = m_i.squeeze(1)

            # min max normalize between 0 and 1
            m_v = self.normalize(m_v)
            m_o = self.normalize(m_o)
            m_a = self.normalize(m_a)
            m_i = self.normalize(m_i)

            iou_matrix = self.calculate_IOU_matrix(m_a, gt_amodal_masks[0], pred_bbs=None, gt_bbs=None)
            iou_threshold = 0.5
            # calculate TP and FP
            for i in range(iou_matrix.shape[0]):
                col = iou_matrix[i].argmax()
                if iou_matrix[i].max() >= iou_threshold:
                    if iou_matrix[i, col] == iou_matrix[:, col].max():
                        TP += 1
                    else:
                        FP += 1
                else:
                    FP += 1
                
                recall = TP/gt_amodal_masks[0].shape[0]
                precision = TP/(TP+FP)
                precision_lst.append(precision)
                recall_lst.append(recall)

        # calculate average precision using 11-point interpolation
        # sort precision and recall lists
        precision_lst = np.array(precision_lst)
        recall_lst = np.array(recall_lst)
        idx = np.argsort(recall_lst)
        precision_lst = precision_lst[idx]
        recall_lst = recall_lst[idx]
        # calculate average precision
        ap = 0
        for r in np.linspace(0, 1, 11):
            p = precision_lst[recall_lst >= r].max()
            if p is None:
                p = 0
            ap += p/11

        # calculate average recall
        ar = recall_lst.max()

        
        print("Average precision: {}".format(ap))"""

        
def main(rank: int, world_size: int, cfg: OmegaConf):
    try: 
        if world_size > 1:
            ddp_setup(rank, world_size)
            writer = SummaryWriter("{}/{}".format(RUNS_FOLDER, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            print("Init the Trainer...")
            #trainer = Trainer(cfg=cfg, device=rank, multi_gpu=True)
            trainer = Trainer(cfg=cfg, device=rank, multi_gpu=True, writer=writer)
        else:
            writer = SummaryWriter("{}/{}".format(RUNS_FOLDER, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            print("Init the Trainer...")
            #trainer = Trainer(cfg=cfg, device=rank, multi_gpu=False)
            trainer = Trainer(cfg=cfg, device=rank, multi_gpu=False, writer=writer)
            
        trainer.train()

        writer.close()
        if world_size > 1:
            destroy_process_group()
    
    except Exception as e:
        logging.error(traceback.format_exc())
        writer.close()

        if world_size > 1:
            destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.cuda.empty_cache()
    # log_level = logging.getLevelName(cfg.LOG_LEVEL)
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     handlers=[logging.StreamHandler()]
    # )
    #world_size = 1

    if world_size == 0:
        main(-1, world_size, cfg)
    elif world_size == 1:
        main(0, world_size, cfg)
    else:
        if cfg.DISTRIBUTED.MODEL: # split model across 2 gpus
            assert world_size == 2, "Only 2 gpus supported for model parallelism"
            world_size = world_size//2
            mp.spawn(main, args=(world_size, cfg), nprocs=world_size)
        elif cfg.DISTRIBUTED.DATA: # split data across n gpus
            # cfg.BATCH_SIZE = cfg.BATCH_SIZE*world_size
            mp.spawn(main, args=(world_size, cfg), nprocs=world_size)

    