from aisformer import AISFormer
from ..asd_dataset import AmodalSynthDriveDataset

from omegaconf import OmegaConf
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader


def train():
    cfg = OmegaConf.load("configs/config.yaml")
    in_shape = tuple((cfg.MODEL.PARAMS.INPUT_H, cfg.MODEL.PARAMS.INPUT_W))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AISFormer(in_shape, cfg)

    model.train()
    model.to(device)

    data_root = "dir"
    dataset = AmodalSynthDriveDataset(data_root)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for data, anns in tqdm(dataloader):
        # TODO: 
        # 1) fetch necessary data from data and anns
        # 2) pass images through model (output will be the four masks)
        # 3) calculate the loss
        # 4) update the weights (for this we need to define an optimizer -> check paper)
        # 5) start training and save masks, feature tensors, and attention maps for visualisation (check paper what they used)
        raise NotImplementedError
    
