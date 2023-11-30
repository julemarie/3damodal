import numpy as np
import torch
from torch.utils.data import DataLoader
from itertools import combinations

# AmodalSynthDrive classes
CLASSES = {
        'unlabeled': 0,
        'ego vehicle': 1,
        'rectification border': 2,
        'out of roi': 3,
        'static': 4,
        'dynamic': 5,
        'ground': 6,
        'road': 7,
        'sidewalk': 8,
        'parking': 9,
        'rail track': 10,
        'building': 11,
        'wall': 12,
        'fence': 13,
        'guard rail': 14,
        'bridge': 15,
        'tunnel': 16,
        'polegroup': 17,
        'pole': 18,
        'traffic light': 19,
        'traffic sign': 20,
        'vegetation': 21,
        'terrain': 22,
        'sky': 23,
        'person': 24,
        'rider': 25,
        'car': 26,
        'truck': 27,
        'bus': 28,
        'caravan': 29,
        'trailer': 30,
        'train': 31,
        'motor': 32,
        'bike': 33,
        'license plate': -1,
        'road line': 34,
        'other': 35,
        'water': 36
        }

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
        for anno in data[setting]:
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


def collate_fn_mask(list_data):
    pass

def get_lidar_dataloader(dataset, batch_size, num_workers, partition="lidar", shuffle=True, drop_last=False):
    if partition=="lidar":
        collate = collate_fn_lidar
    elif partition=="image":
        collate = collate_fn_mask
    else:
        raise ValueError
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate,
    )
    return dataloader
