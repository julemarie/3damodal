import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import time
import json
import cv2
from PIL import Image

class KINS(Dataset):
    def __init__(self, root_dir="/Midgard/Data/tibbe/datasets/KINS/", mode="train", transform=None):
        assert (mode == "train" or mode == "test"), "mode should be either 'train' or 'test'."

        self.root_dir = root_dir
        self.imgs_dir = os.path.join(self.root_dir, mode + "_imgs")
        self.imgs_lst = os.listdir(self.imgs_dir)

        anns_dir = os.path.join(self.root_dir, "annotations")
        with open(os.path.join(anns_dir, "update2020_"+mode + ".json"), "r") as af:
            self.anns_dict = json.load(af)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),

                transforms.Resize((256, 832)) # oriiginal size is (375, 1242)
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.imgs_lst)

    def get_annotations(self, image_id):
        anns = self.anns_dict['annotations']
        idxs = [i for i, d in enumerate(anns) if d['image_id'] == image_id]
        anns_lst = [anns[i] for i in idxs]
        anns_dict = {}
        for d in anns_lst:
            anns_dict[d['id']] = d
        return anns_dict

    def get_image(self, idx):
        file = self.imgs_lst[idx]
        img = Image.open(os.path.join(self.imgs_dir, file)).convert('RGB')
        img = self.transform(img)
        index = [i for i, d in enumerate(self.anns_dict['images']) if d['file_name'] == file]
        if len(index) == 0:
            print("{} not in annotations dictionary!".format(file))
            return self.get_image(idx+1)
        img_dict = self.anns_dict['images'][index[0]]
        return img, img_dict

    def __getitem__(self, idx):
        img, img_dict = self.get_image(idx)

        anns_dict = self.get_annotations(img_dict['id'])
        # anns_dict['amodal_full'] = img_dict['amodal_full']
        anns_dict['img_height'] = img_dict['height']
        anns_dict['img_width'] = img_dict['width']
        
        return img, anns_dict



def testing():
    kins_dataset = KINS()
    print(isinstance(kins_dataset, KINS))
    dl = DataLoader(kins_dataset, batch_size=1)
    start = time.time()
    for img, anns in dl:
        print(anns.keys())
        print("TIME",time.time()-start)
        start = time.time()
        break


if __name__ == "__main__":
    testing()