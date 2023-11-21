from aisformer import AISFormer
from datasets.asd_dataset import AmodalSynthDriveDataset
from utils import get_obj_from_str

from omegaconf import OmegaConf
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, cfg_path="/Midgard/home/tibbe/3damodal/3DAmodal/configs/config.yaml", try_cuda=True, train=True):
        self.cfg = OmegaConf.load(cfg_path)
        in_shape = tuple((self.cfg.MODEL.PARAMS.INPUT_H, self.cfg.MODEL.PARAMS.INPUT_W))

        if try_cuda:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'
        
        # init model and move it to device
        self.model = AISFormer(in_shape, self.cfg)
        self.model.train()
        self.model.to(self.device)

        # init dataloader
        if train:
            data_root = self.cfg.DATASET.TRAIN
            dataset = get_obj_from_str(self.cfg.DATASET.NAME)(data_root)
            self.img_views = dataset.img_settings
        else:
            raise NotImplementedError

        self.dataloader = DataLoader(dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True)


    def train(self):

        for data, anns in tqdm(self.dataloader):
            # TODO: 
            # 1) fetch necessary data from data and anns
            imgs = []
            for view in self.img_views:
                imgs.append(data[view])
            
            imgs = torch.tensor(imgs)
            print(imgs.shape)
            # 2) pass images through model (output will be the four masks)
            # 3) calculate the loss
            # 4) update the weights (for this we need to define an optimizer -> check paper)
            # 5) start training and save masks, feature tensors, and attention maps for visualisation (check paper what they used)
    



def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()