from aisformer import AISFormer
# from datasets.asd_dataset import AmodalSynthDriveDataset
# from datasets.KINS_dataset import KINS
import sys
sys.path.append('/Midgard/home/tibbe/3damodal/3DAmodal/datasets')
from dataloader import get_dataloader
from utils import get_obj_from_str

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
        self.gpu_id = gpu_id
        self.cfg = cfg
        self.in_shape = tuple((self.cfg.MODEL.PARAMS.INPUT_H, self.cfg.MODEL.PARAMS.INPUT_W))

        # if self.cfg.DEVICE == 'cuda':
        #     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # else:
        #     self.device = 'cpu'

        self.save_interval = self.cfg.SAVE_INTERVAL
        
        # init model and move it to device
        self.model = AISFormer(self.in_shape, self.cfg)
        self.model.train()
        self.model.backbone.eval()
        self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        # init dataloader
        if train:
            data_root = self.cfg.DATASET.TRAIN
            dataset = get_obj_from_str(self.cfg.DATASET.NAME)(data_root)
            # self.img_views = dataset.img_settings
        else:
            raise NotImplementedError

        self.dataloader = get_dataloader(dataset, batch_size=10, num_workers=5, partition="image", distributed=self.cfg.DISTRIBUTED)

        # self.resize = transforms.Resize((64, 64))
        self.roi_size = tuple((self.cfg.MODEL.PARAMS.ROI_OUT_SIZE*2, self.cfg.MODEL.PARAMS.ROI_OUT_SIZE*2))

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.LEARNING_RATE)

        self.epochs = self.cfg.EPOCHS

        

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
        _, y_min, x_min, y_max, x_max = roi
        cropped_mask = mask[x_min:x_max, y_min:y_max]
        resized_mask = cropped_mask.resize_(self.roi_size)
        # resized_mask = self.resize(cropped_mask)
        return resized_mask
    

    def compute_loss(self, pred_mask, gt_mask):
        bce_loss = torch.nn.BCELoss()
        return bce_loss(pred_mask, gt_mask)


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
            poly = [[poly[i].item(), poly[i+1].item()] for i in range(0, len(poly), 2)]
            poly = np.array(poly, np.int32)
            poly = poly.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [poly], color=(255, 255, 255), lineType=cv2.LINE_AA)
    
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
        return mask


    def get_masks_kins(self, anns):
        anns['amodal_full']['counts'] = anns['amodal_full']['counts'][0]
        # 1) fetch necessary data from data and anns
        h, w = anns['img_height'], anns['img_width']
        amodal_mask = self.decode_mask_rle(anns['amodal_full'])
        seg_masks = []
        inmodal_masks = []
        for key in anns.keys():
            if key == "amodal_full" or key == "img_height" or key == "img_width":
                continue
            segmentation = self.decode_mask_lst(anns[key]['segmentation'], h, w)
            anns[key]['inmodal_seg']['counts'] = anns[key]['inmodal_seg']['counts'][0]
            inmodal_seg = self.decode_mask_rle(anns[key]['inmodal_seg'])
            inmodal_masks.append(inmodal_seg)
            seg_masks.append(segmentation)
            save_image(inmodal_seg, "test_seg.png")
        seg_masks = torch.stack(seg_masks)
        inmodal_masks = torch.stack(inmodal_masks)
        # combined_seg = torch.einsum('nchw->chw', seg_masks)
        # combined_inmodal = torch.einsum('nchw->chw', inmodal_masks)

        return inmodal_masks, seg_masks
    
    def calculate_IOU(self, mask1, mask2):
        """
            Calculate IOU between two given masks.
        """
        intersection = (mask1.int() & mask2.int()).float().sum((0,1))
        union = (mask1.int() | mask2.int()).float().sum((0,1))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou

    def calculate_IOU_matrix(self, pred_masks, gt_masks, bbs):
        """
            Calculate IOU matrix between prediction masks and GT masks.
            Input:
                pred_masks: Tensor [K, 1, 64, 64]
                gt_masks: Tensor [K_gt, 2, 540, 960]
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
                if gt_masks[j, 1, 0, 0] == bbs[i, 0]: # check if gt mask is in same image as prediction
                    gt_mask_roi = self.crop_and_resize_mask(gt_masks[j], bbs[i])
                    iou_matrix[i, j] = self.calculate_IOU(pred_masks[i], gt_mask_roi)
                else:
                    iou_matrix[i, j] = 0.0

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
                gt_masks: Tensor [K_gt, 2, 540, 960]
                bbs: Tensor [K, 5]
        """
        loss = 0
        for i, j in assignment.items():
            if j == 'fp': # if false positive
                zero_mask = torch.zeros((pred_masks[i].shape), device=self.gpu_id)
                loss += self.compute_loss(pred_masks[i], zero_mask)
            else:
                gt_mask = self.crop_and_resize_mask(gt_masks[j, 1], bbs[i])
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



def main(rank: int, world_size: int, cfg: OmegaConf):
    ddp_setup(rank, world_size)
    trainer = Trainer(gpu_id=rank, cfg=cfg)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    cfg_path="/Midgard/home/tibbe/3damodal/3DAmodal/configs/config.yaml"
    cfg = OmegaConf.load(cfg_path)
    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size, cfg), nprocs=world_size)