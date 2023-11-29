from aisformer import AISFormer
from datasets.asd_dataset import AmodalSynthDriveDataset
from datasets.KINS_dataset import KINS
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


class Trainer():
    def __init__(self, cfg_path="/Midgard/home/tibbe/3damodal/3DAmodal/configs/config.yaml", try_cuda=True, train=True):
        self.cfg = OmegaConf.load(cfg_path)
        self.in_shape = tuple((self.cfg.MODEL.PARAMS.INPUT_H, self.cfg.MODEL.PARAMS.INPUT_W))

        if try_cuda:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'
        
        # init model and move it to device
        self.model = AISFormer(self.in_shape, self.cfg)
        self.model.train()
        self.model.backbone.eval()
        self.model.to(self.device)

        # init dataloader
        if train:
            data_root = self.cfg.DATASET.TRAIN
            dataset = get_obj_from_str(self.cfg.DATASET.NAME)(data_root)
            # self.img_views = dataset.img_settings
        else:
            raise NotImplementedError

        self.dataloader = DataLoader(dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True)

        # self.resize = transforms.Resize((64, 64))
        self.roi_size = tuple((self.cfg.MODEL.PARAMS.ROI_OUT_SIZE*2, self.cfg.MODEL.PARAMS.ROI_OUT_SIZE*2))

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.LEARNING_RATE)

        self.epochs = self.cfg.EPOCHS

        self.viewstr2int = {"front_full": 0,
                            "back_full": 1,
                            "left_full": 2,
                            "righ_full": 3,
                            "bev_full": 4}


    def crop_and_resize_mask(self, mask, roi):
        y_min, x_min, y_max, x_max = roi
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
    
    def get_gt_masks_asd(self, anns):
        gt_vis_masks = [ann['visible_mask'].squeeze(0) for ann in anns]
        gt_vis_masks = torch.stack(gt_vis_masks) # [4, 3, 540, 960]
        gt_amod_masks = []
        gt_invis_masks = []
        gt_occl_masks = []
        for i, ann in enumerate(anns):
            for key in ann.keys():
                if key == 'visible_mask':
                    continue
                if not ann[key]['occlusion_mask'] == []:
                    occl_mask = ann[key]['occlusion_mask'].squeeze()
                    im_id = i*torch.ones(occl_mask.shape)
                    occl_mask = torch.stack((occl_mask,im_id))
                    gt_occl_masks.append(occl_mask)
                    amod_mask = ann[key]['amodal_mask'].squeeze()
                    amod_mask = torch.stack((amod_mask,im_id))
                    gt_amod_masks.append(amod_mask)
                    invis_mask = amod_mask[0] - gt_vis_masks[i,0]
                    invis_mask[invis_mask<0] = 0
                    invis_mask = torch.stack((invis_mask,im_id))
                    gt_invis_masks.append(invis_mask)
        
        gt_amod_masks = torch.stack(gt_amod_masks) # [K_gt, 2, 540, 960]
        gt_occl_masks = torch.stack(gt_occl_masks) # [K_gt, 2, 540, 960]
        gt_invis_masks = torch.stack(gt_invis_masks) # [K_gt, 2, 540, 960]

        return gt_vis_masks, gt_amod_masks, gt_occl_masks, gt_invis_masks
    
    def calculate_IOU(self, mask1, mask2):
        intersection = (mask1 & mask2).float().sum((1,2))
        union = (mask1 | mask2).float().sum((1,2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou

    def calculate_IOU_matrix(self, pred_masks, gt_masks, bbs):
        num_pred = pred_masks.shape[0]
        num_gt = gt_masks.shape[0]
        iou_matrix = torch.zeros((num_pred, num_gt))
        for i in range(num_pred):
            for j in range(num_gt):
                gt_mask_roi = self.crop_and_resize_mask(gt_masks[j], bbs[i])
                iou_matrix[i, j] = self.calculate_IOU(pred_masks[i], gt_mask_roi)

        return iou_matrix
    
    def assign_iou(self, pred_masks, gt_masks, bbs, iou_threshold=0.5):
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

    def train(self):

        for epoch_idx in tqdm(range(self.epochs)):
            # for data, anns in self.dataloader:
            for img_batch, anns in self.dataloader:
                # 1) Extract masks from annotations
                gt_vis_masks, gt_amod_masks, gt_occl_masks, gt_invis_masks = self.get_gt_masks_asd(anns)
                # img_batch size: [B, 5, 3, 540, 960]
                img_batch = img_batch[:,:4] # we don't care about the BEV image here
                B, V, C, H, W = img_batch.shape
                img_batch = img_batch.reshape((B*V, C, H, W))
                
                # 2) pass images through model (output will be the four masks)
                # img = cv2.imread("3DAmodal/front_full_0000_rgb.jpg")
                # img2 = cv2.imread("3DAmodal/front_full_0063_rgb.jpg")
                # H, W, C = img.shape # for ASD: [1080, 1920, 3]
                # transform = transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.ConvertImageDtype(torch.float),
                #     transforms.RandomCrop((int(H/1.5), int(W/1.5))),
                #     transforms.Resize(self.in_shape)
                # ])
                # img = transform(img).unsqueeze(0)
                # img2 = transform(img2).unsqueeze(0)

                # imgs = torch.cat((img, img2), dim=0)
                # gt_amod_masks = torch.ones((1, 1, 540, 960))
                # gt_occl_masks = torch.ones((1, 1, 540, 960))
                # gt_vis_masks = torch.ones((4, 3, 540, 960))
                # gt_vis_masks = gt_vis_masks[:,0,:,:] # has three channels, but they're all the same
                # gt_invis_masks = gt_amod_masks - gt_vis_masks

                masks, rois = self.model(img_batch) # [K, 4, 64, 64], [K, 5]
                rois = rois.to(torch.int)
                K, num_masks, H_m, W_m = masks.shape
                m_o = masks[:, 0]
                m_v = masks[:, 1]
                m_a = masks[:, 2]
                m_i = masks[:, 3]



                # 3) calculate the loss
                # 3.1) assign predicted masks to gt masks using IOU matrix and Hungarian algorithm


                total_loss = 0
                # for i in range()
                for ins in range(K):
                    bbx = rois[ins, 1:]
                    img_id = rois[ins, 0]
                    # crop GT masks to bbx
                    gt_vis_mask = self.crop_and_resize_mask(gt_vis_masks[img_id], bbx)
                    loss_v = self.compute_loss(m_v[ins], gt_vis_mask)
                    loss_o = 1e6
                    loss_a = 1e6
                    loss_i = 1e6
                    for mask_id in range(gt_amod_masks.shape[0]):
                        gt_amod_mask = self.crop_and_resize_mask(gt_amod_masks[mask_id], bbx)
                        gt_occl_mask = self.crop_and_resize_mask(gt_occl_masks[mask_id], bbx)
                        gt_invis_mask = self.crop_and_resize_mask(gt_invis_masks[mask_id], bbx)
                        loss_ok = self.compute_loss(m_o[ins], gt_occl_mask)
                        if loss_ok < loss_o:
                            loss_o = loss_ok
                        loss_ak = self.compute_loss(m_a[ins], gt_amod_mask)
                        if loss_ak < loss_a:
                            loss_a = loss_ak
                        loss_ik = self.compute_loss(m_i[ins], gt_invis_mask)
                        if loss_ik < loss_i:
                            loss_i = loss_ik
                    
                    total_loss += loss_o + loss_a + loss_i + loss_v

                # 4) update the weights (for this we need to define an optimizer -> check paper)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                # 5) start training and save masks, feature tensors, and attention maps for visualisation (check paper what they used)

            print("Loss after epoch {}: {}".format(epoch_idx, total_loss))



def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()