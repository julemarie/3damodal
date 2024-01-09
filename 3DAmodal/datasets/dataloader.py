import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from itertools import combinations
import datasets.KINS_dataset
import datasets.asd_dataset


def bbox_corners_to_xyzwhltheta(bbox):
    combs = list(combinations(bbox[:,:3], 2))
    distances = np.array([[np.linalg.norm(p1[0]-p2[0]), np.linalg.norm(p1[1]-p2[1]), np.linalg.norm(p1[2]-p2[2])] for p1, p2 in combs])

    center = np.mean(bbox, axis=0)
    w = np.max(distances[:, 0])
    h = np.max(distances[:, 1])
    l = np.max(distances[:, 2])
    
    centered_points = bbox - center
    e_val, e_vec = np.linalg.eig(np.cov(centered_points, rowvar=False))
    sorted_indices = np.argsort(e_val)[::-1]
    e_vec = e_vec[:, sorted_indices]
    rot_axis = e_vec[:, 0].astype(np.float32)
    theta = np.arctan2(rot_axis[1], rot_axis[0])
    theta_deg = np.degrees(theta)


    formatted_bbox = np.concatenate([center, np.array([w, h, l, theta_deg])], dtype=np.float32)
    return formatted_bbox

def retrieve_gt_bboxes(data):
    settings = ["front_full_", "back_full_", "right_full_", "left_full_"]
    seen_ids = []
    gt_bboxes = []
    for setting in settings:
        for anno in data[setting][1:]:
            bbox = anno["bbox"]
            if anno["track_id"] not in seen_ids:
                seen_ids.append(anno["track_id"])
                bbox = bbox_corners_to_xyzwhltheta(bbox)
                gt_bboxes.append(bbox)
    return np.array(gt_bboxes)

def collate_fn_lidar(list_data):
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list = []
    for data_dict in list_data:
        inputs, labels = data_dict
        pts = inputs["lidar"]["points"]
        if pts.shape[1] == 3:
            col = np.zeros((pts.shape[0], 1)).astype(np.float32)
            pts = np.hstack((pts, col))
        gt_bboxes_3d = retrieve_gt_bboxes(labels)
        gt_labels = labels["lidar"]

        batched_pts_list.append(torch.from_numpy(pts).contiguous())
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))

    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list
    )

    return rt_data_dict

def get_gt_masks_asd(anns):
        gt_vis_masks = [ann[0] for ann in anns]
        gt_vis_masks = torch.stack(gt_vis_masks) # [4, 3, 540, 960]
        gt_amod_masks = []
        gt_invis_masks = []
        gt_occl_masks = []
        for i, ann in enumerate(anns):
            for j, m_dict in enumerate(ann[1:]):
                if not m_dict['occlusion_mask'] == []:
                    occl_mask = m_dict['occlusion_mask']
                    im_id = i*torch.ones(occl_mask.shape)
                    occl_mask = torch.stack((occl_mask,im_id))
                    gt_occl_masks.append(occl_mask)
                    amod_mask = m_dict['amodal_mask']
                    amod_mask = torch.stack((amod_mask,im_id))
                    gt_amod_masks.append(amod_mask)
                    invis_mask = amod_mask[0] - gt_vis_masks[i,0]
                    invis_mask[invis_mask<0] = 0
                    invis_mask = torch.stack((invis_mask,im_id))
                    gt_invis_masks.append(invis_mask)
        
        gt_amod_masks = torch.stack(gt_amod_masks) # [K_gt, 2, 540, 960]
        gt_occl_masks = torch.stack(gt_occl_masks) # [K_gt, 2, 540, 960]
        gt_invis_masks = torch.stack(gt_invis_masks) # [K_gt, 2, 540, 960]

        masks_dict = {"visible": gt_vis_masks,
                      "amodal": gt_amod_masks,
                      "occluder": gt_occl_masks,
                      "invisible": gt_invis_masks}

        return masks_dict


def collate_fn_mask(list_data):
    print("DATALOADER")
    mask_dict_list = []
    img_batch = torch.zeros((len(list_data), 5, 3, 540, 960))
    for i, (X, Y) in enumerate(list_data):
        img_batch[i] = X['imgs']

        masks = [Y[key] for key in list(Y.keys())[1:]] # list of lists

        masks_dict = get_gt_masks_asd(masks)
        mask_dict_list.append(masks_dict)

    # assert gpu_id < num_devices, "GPU ID cannot be larger than number of devices"

    # only return the image of the view coresponding to the gpu_id
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        num_devices = torch.cuda.device_count()
        img_batch = img_batch[:,gpu_id%num_devices]
        mask_dict_list = [mask_dict_list[gpu_id%num_devices]]
    

    return img_batch, mask_dict_list

def collate_fn_image_kins(list_data):
    anns_dict_list = []
    img_batch = torch.zeros_like(list_data[0][0]).unsqueeze(0).repeat((len(list_data), 1, 1, 1))
    # img_batch = torch.zeros((len(list_data), 3, 540, 960))
    for i, (X, Y) in enumerate(list_data):
        img_batch[i] = X
        anns_dict_list.append(Y)
    
    return img_batch, anns_dict_list


def get_dataloader(dataset, batch_size, partition="lidar", num_workers=0, shuffle=True, drop_last=False, distributed=False):
    global NUM_DEVICES, GPU_ID
    
    if partition=="lidar":
        collate = collate_fn_lidar
    elif partition=="image":
        if isinstance(dataset, datasets.KINS_dataset.KINS):
            collate = collate_fn_image_kins
        elif isinstance(dataset, datasets.asd_dataset.AmodalSynthDriveDataset):
            collate = collate_fn_mask
    else:
        raise ValueError
    if distributed:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last, 
            collate_fn=collate,
            sampler=DistributedSampler(dataset)
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last, 
            collate_fn=collate,
        )
    return dataloader


if __name__ == "__main__":
    kins_dataset = KINS()
    dl = get_dataloader(kins_dataset, batch_size=1, partition="image")

    for X, Y in dl:
        print(X.shape)
        print(Y)
        break