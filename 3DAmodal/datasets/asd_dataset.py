import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import json
import cv2 as cv
from pycocotools import mask as coco_mask
# from pycocotools import coco

FRAMES_PER_SETTING = 100

def custom_collate_fn(batch):
    # 'batch' is a list of dictionaries where each dictionary has keys corresponding to views (e.g., 'front_full_', 'lidar')

    # Separate the batch into lists of dictionaries for X and Y
    X_batch, Y_batch = zip(*batch)

    # Combine dictionaries in X_batch and Y_batch into a single dictionary for each
    X_collated = {key: torch.stack([sample[key] for sample in X_batch]) for key in X_batch[0]}
    Y_collated = {key: torch.stack([sample[key] for sample in Y_batch]) for key in Y_batch[0]}

    return X_collated, Y_collated

class AmodalSynthDriveDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.img_settings = ["front_full_",
                             "back_full_",
                             "left_full_",
                             "right_full_",
                             "bev_full_"]
        self.settings = os.listdir(os.path.join(data_root, "amodal_instance_seg"))

    def __len__(self):
        return len(self.settings) * FRAMES_PER_SETTING
    
    def __getitem__(self, index):
        setting_name, str_id = self.map_to_folder(index)

        # while setting_name not in (os.listdir(os.path.join(self.data_root, "images")) and
        #                         os.listdir(os.path.join(self.data_root, "bboxes")) and
        #                         os.listdir(os.path.join(self.data_root, "amodal_instance_seg")) and
        #                         os.listdir(os.path.join(self.data_root, "lidar"))):
        #     setting_name, str_id = self.map_to_folder(index)

        # assert setting_name in os.listdir(os.path.join(self.data_root, "images")), "Folder {} not in directory {}.".format(setting_name, os.path.join(self.data_root, "images"))
        # assert setting_name in os.listdir(os.path.join(self.data_root, "bboxes")), "Folder {} not in directory {}.".format(setting_name, os.path.join(self.data_root, "bboxes"))
        # assert setting_name in os.listdir(os.path.join(self.data_root, "amodal_instance_seg")), "Folder {} not in directory {}.".format(setting_name, os.path.join(self.data_root, "amodal_instance_seg"))
        # assert setting_name in os.listdir(os.path.join(self.data_root, "lidar")), "Folder {} not in directory {}.".format(setting_name, os.path.join(self.data_root, "lidar"))

        # imgs = self.get_imgs(setting_name, str_id)
        # bboxes = self.get_bboxes(setting_name, str_id)
        aminseg_anno = self.get_amodal_instance_annos(setting_name, str_id)
        # lidar = self.get_lidar(setting_name, str_id)

        # X = {
        #     "lidar": {
        #         "points": lidar["points"],
        #         "transform": np.array(lidar["transform"]),
        #         "horizontal_angle": lidar["horizontal_angle"]
        #     }
        # }
        # Y = {
        #     "lidar": lidar["labels"].astype(np.int32)
        # }

        # for i, view in enumerate(self.img_settings):
        #     X[view] = imgs[i]
        #     if view == "bev_full_":
        #         continue
        #     annos = [aminseg_anno[i][aminseg] for aminseg in aminseg_anno[i]]
        #     for anno in annos:
        #         anno["bbox"] = np.array(bboxes[anno["track_id"]])
        #     Y[view] = annos

        return torch.zeros((2,2)), aminseg_anno


    def map_to_folder(self, index):
        setting = index // FRAMES_PER_SETTING
        id = index % FRAMES_PER_SETTING

        setting_str = self.settings[setting]
        id_to_str = str(id).rjust(4, "0")
        return setting_str, id_to_str
    
    def get_imgs(self, setting_name, str_id):
        imgs = []
        for img_view in self.img_settings:
            img_path = os.path.join(self.data_root, "images", setting_name, img_view + str_id + "_rgb.jpg")
            img = cv.imread(img_path)
            imgs.append(img)

        return imgs
    
    def get_bboxes(self, setting_name, str_id):
        bbox_path = os.path.join(self.data_root, "bboxes", setting_name, "bboxess_" + str_id + ".pkl")
        with open(bbox_path, "rb") as bf:
            bboxes = pickle.load(bf)
        return bboxes
    
    def get_amodal_instance_annos(self, setting_name, str_id):
        masks_dict_list = []
        for img_view in self.img_settings:
            if img_view == "bev_full_":
                continue
            amodal_path = os.path.join(self.data_root, "amodal_instance_seg", setting_name, img_view + str_id + "_aminseg.json")
            with open(amodal_path,"r") as af:
                anno = json.load(af)
            img_annos = {"visible_mask": []}
            for key in anno: # iterating over the annotations of the instances
                img_annos[anno[key]["track_id"]] = {
                    "category_id": anno[key]["category_id"],
                    "occlusion_mask": [],
                    "amodal_mask": []
                }
                if anno[key]["occluded"]:
                    # anno[key]["occlusion_mask"]["counts"] = anno[key]["occlusion_mask"]["counts"]
                    # anno[key]["amodal_mask"]["counts"] = anno[key]["amodal_mask"]["counts"]
                    # print(anno[key]["amodal_mask"])
                    occl_mask = coco_mask.decode(anno[key]["occlusion_mask"])
                    amod_mask = coco_mask.decode(anno[key]["amodal_mask"])

                    img_annos[anno[key]["track_id"]]["occlusion_mask"] = torch.tensor(occl_mask, dtype=torch.float).unsqueeze(0)
                    img_annos[anno[key]["track_id"]]["amodal_mask"] = torch.tensor(amod_mask, dtype=torch.float).unsqueeze(0)

            # amodal_path = ".".join([amodal_path.split(".")[0], "png"])
            # mask = cv.imread(amodal_path)
            # img_annos["visible_mask"] = mask
            masks_dict_list.append(img_annos)
            
        assert len(masks_dict_list) == len(self.img_settings) - 1, "The dictionary does not have the right size!"

            # ann_mask = {"coco_annotation": anno,
            #      "visible_mask": mask}
            # masks_dict_list.append(ann_mask)
            
        return masks_dict_list
    
    def get_lidar(self, setting_name, str_id):
        lidar_path = os.path.join(self.data_root, "lidar", setting_name, "full_" + str_id + "_Lidar.pkl")
        with open(lidar_path, "rb") as lf:
            lidar = pickle.load(lf)
        return lidar
    


if __name__ == "__main__":
    ds = AmodalSynthDriveDataset("/Midgard/Data/tibbe/datasets/AmodalSynthDrive/train")
    views = ds.img_settings
    dl = DataLoader(ds, batch_size=1)
    for amodl_anns in dl:
        print(type(amodl_anns))
        print(amodl_anns.keys())


        
    #     masks_dict = {}
    #     img_masks = {}
    #     for i, ann in  enumerate(amodl_anns): # iterating over the annotations of the loaded images
    #         ann_dict = ann["coco_annotation"]
    #         vis_mask = ann["visible_mask"]
    #         bin_occl_masks = []
    #         bin_amod_masks = []
    #         img_masks["visible_mask"] = vis_mask
    #         for key in ann_dict: # iterating over the annotations of the instances
    #             inst_mask = {ann_dict[key]["track_id"]: {
    #                 "occlusion_mask": None,
    #                 "amodal_mask": None
    #             }}
    #             if ann_dict[key]["occluded"]:
    #                 ann_dict[key]["occlusion_mask"]["counts"] = ann_dict[key]["occlusion_mask"]["counts"][0]
    #                 ann_dict[key]["amodal_mask"]["counts"] = ann_dict[key]["amodal_mask"]["counts"][0]
    #                 print(ann_dict[key]["amodal_mask"])
    #                 occl_mask = coco_mask.decode(ann_dict[key]["occlusion_mask"])
    #                 amod_mask = coco_mask.decode(ann_dict[key]["amodal_mask"])

    #                 bin_occl_mask = torch.tensor(occl_mask, dtype=torch.float).unsqueeze(0)
    #                 bin_amod_mask = torch.tensor(amod_mask, dtype=torch.float).unsqueeze(0)

    #                 inst_mask[ann_dict[key]["track_id"]]["occlusion_mask"] = torch.tensor(occl_mask, dtype=torch.float).unsqueeze(0)
    #                 inst_mask[ann_dict[key]["track_id"]]["amodal_mask"] = torch.tensor(amod_mask, dtype=torch.float).unsqueeze(0)

    #                 img_masks["other_masks"] = inst_mask

    #         masks_dict[views[i]] = img_masks
    #     break


    # [{
    #     "visible_mask": "MASK",
    #     "other_masks": {
    #         "track_id": {
    #             "category_id": "ID",
    #             "occlusion_mask": "MASK",
    #             "amodal_mask": "MASK"},
    #         "track_id": {
    #             "category_id": "ID",
    #             "occlusion_mask": "MASK",
    #             "amodal_mask": "MASK"},
    #         "track_id": {
    #             "category_id": "ID",
    #             "occlusion_mask": "MASK",
    #             "amodal_mask": "MASK"}
    #       }
    #    },
    # ]
