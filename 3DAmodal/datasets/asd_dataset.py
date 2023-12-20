import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from dataloader import get_dataloader
import os
import pickle
import json
import cv2 as cv
from pycocotools import mask as coco_mask

FRAMES_PER_SETTING = 100

# AmodalSynthDrive classes
NUM_ALL_CLASSES = 38
ALL_CLASSES = {
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

NUM_CLASSES = 9
INSTANCE_CLASSES = {
    'person':24 ,
    'rider':25 ,
    'car':26 ,
    'truck':27 ,
    'bus':28 ,
    'caravan':29 ,
    'trailer':30 ,
    'motor':32 ,
    'bike':33
}

CLASS_MAPPING = {
    24: 0,
    25: 1,
    26: 2,
    27: 3,
    28: 4,
    29: 5,
    30: 6,
    32: 7,
    33: 8
}


class AmodalSynthDriveDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.img_settings = ["front_full_",
                             "back_full_",
                             "left_full_",
                             "right_full_",
                             "bev_full_"]
        self.settings = os.listdir(os.path.join(data_root, "amodal_instance_seg"))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((540, 960))
            ])

    def __len__(self):
        return len(self.settings) * FRAMES_PER_SETTING
    
    def __getitem__(self, index):
        setting_name, str_id = self.map_to_folder(index)

        while setting_name not in (os.listdir(os.path.join(self.data_root, "images")) and
                                os.listdir(os.path.join(self.data_root, "bboxes")) and
                                os.listdir(os.path.join(self.data_root, "amodal_instance_seg")) and
                                os.listdir(os.path.join(self.data_root, "lidar"))):
            setting_name, str_id = self.map_to_folder(index)

        assert setting_name in os.listdir(os.path.join(self.data_root, "images")), "Folder {} not in directory {}.".format(setting_name, os.path.join(self.data_root, "images"))
        assert setting_name in os.listdir(os.path.join(self.data_root, "bboxes")), "Folder {} not in directory {}.".format(setting_name, os.path.join(self.data_root, "bboxes"))
        assert setting_name in os.listdir(os.path.join(self.data_root, "amodal_instance_seg")), "Folder {} not in directory {}.".format(setting_name, os.path.join(self.data_root, "amodal_instance_seg"))
        assert setting_name in os.listdir(os.path.join(self.data_root, "lidar")), "Folder {} not in directory {}.".format(setting_name, os.path.join(self.data_root, "lidar"))

        imgs = self.get_imgs(setting_name, str_id)
        bboxes = self.get_bboxes(setting_name, str_id)
        aminseg_anno = self.get_amodal_instance_annos(setting_name, str_id)
        lidar = self.get_lidar(setting_name, str_id)

        X = {
            "lidar": {
                "points": lidar["points"],
                "transform": np.array(lidar["transform"]),
                "horizontal_angle": lidar["horizontal_angle"]
            },
            "imgs": imgs
        }
        Y = {
            "lidar": np.array([CLASS_MAPPING[l]  if l in CLASS_MAPPING.keys() else -1 for l in lidar["labels"]], dtype=np.float32)
        }

        for i, view in enumerate(self.img_settings):
            if view == "bev_full_":
                continue
            annos = [aminseg_anno[i][aminseg] for aminseg in aminseg_anno[i]]
            for anno in annos[1:]:
                anno["bbox"] = np.array(bboxes[anno["track_id"]])
            Y[view] = annos

        return X, Y


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
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = self.transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs)

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
                    "category_id": CLASS_MAPPING[anno[key]["category_id"]] if anno[key]["category_id"] in CLASS_MAPPING.keys() else -1,
                    "track_id": anno[key]["track_id"],
                    "occlusion_mask": [],
                    "amodal_mask": []
                }
                if anno[key]["occluded"]:
                    occl_mask = coco_mask.decode(anno[key]["occlusion_mask"])
                    amod_mask = coco_mask.decode(anno[key]["amodal_mask"])
                    
                    img_annos[anno[key]["track_id"]]["occlusion_mask"] = self.transform(occl_mask)
                    img_annos[anno[key]["track_id"]]["amodal_mask"] = self.transform(amod_mask)

            amodal_path = ".".join([amodal_path.split(".")[0], "png"])
            mask = cv.imread(amodal_path)
            mask = self.transform(mask)
            img_annos["visible_mask"] = mask
            masks_dict_list.append(img_annos)
            
        assert len(masks_dict_list) == len(self.img_settings) - 1, "The dictionary does not have the right size!"

        return masks_dict_list
    
    def get_lidar(self, setting_name, str_id):
        lidar_path = os.path.join(self.data_root, "lidar", setting_name, "full_" + str_id + "_Lidar.pkl")
        with open(lidar_path, "rb") as lf:
            lidar = pickle.load(lf)
        return lidar
    


# if __name__ == "__main__":
#     ds = AmodalSynthDriveDataset("/Midgard/Data/tibbe/datasets/AmodalSynthDrive/train")
#     # ds = AmodalSynthDriveDataset("/home/jule-magnus/dd2414/Data/AmodalSynthDrive/train")
#     views = ds.img_settings
#     dl = get_dataloader(ds, batch_size=3, partition="image")

#     for X, Y in dl:
#         print(X.shape)
#         print(len(Y))
        

