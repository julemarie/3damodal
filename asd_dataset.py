import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import json
import cv2 as cv

FRAMES_PER_SETTING = 100

class AmodalSynthDriveDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.img_settings = ["front_full_",
                             "back_full_",
                             "left_full_",
                             "right_full_",
                             "bev_full_"]
        self.settings = os.listdir(os.path.join(data_root, "lidar"))

    def __len__(self):
        return len(self.settings) * FRAMES_PER_SETTING
    
    def __getitem__(self, index):
        setting_name, str_id = self.map_to_folder(index)

        imgs = self.get_imgs(setting_name, str_id)
        bboxes = self.get_bboxes(setting_name, str_id)
        aminseg_anno = self.get_amodal_instance_annos(setting_name, str_id)
        lidar = self.get_lidar(setting_name, str_id)

        X = {
            "lidar": {
                "points": lidar["points"],
                "transform": np.array(lidar["transform"]),
                "horizontal_angle": lidar["horizontal_angle"]
            }
        }
        Y = {
            "lidar": lidar["labels"].astype(np.int32)
        }

        for i, view in enumerate(self.img_settings):
            X[view] = imgs[i]
            if view == "bev_full_":
                continue
            annos = [aminseg_anno[i][aminseg] for aminseg in aminseg_anno[i]]
            for anno in annos:
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
            imgs.append(img)

        return imgs
    
    def get_bboxes(self, setting_name, str_id):
        bbox_path = os.path.join(self.data_root, "bboxes", setting_name, "bboxess_" + str_id + ".pkl")
        with open(bbox_path, "rb") as bf:
            bboxes = pickle.load(bf)
        return bboxes
    
    def get_amodal_instance_annos(self, setting_name, str_id):
        annos = []
        for img_view in self.img_settings:
            if img_view == "bev_full_":
                continue
            amodal_path = os.path.join(self.data_root, "amodal_instance_seg", setting_name, img_view + str_id + "_aminseg.json")
            with open(amodal_path) as af:
                anno = json.load(af)
            annos.append(anno)
        return annos
    
    def get_lidar(self, setting_name, str_id):
        lidar_path = os.path.join(self.data_root, "lidar", setting_name, "full_" + str_id + "_Lidar.pkl")
        with open(lidar_path, "rb") as lf:
            lidar = pickle.load(lf)
        return lidar
    


if __name__ == "__main__":
    ds = AmodalSynthDriveDataset("/home/jule-magnus/dd2414/Data/AmodalSynthDrive/train")
    dl = DataLoader(ds)
    for dp, l in dl:
        print(dp.keys(), l.keys())
        break

