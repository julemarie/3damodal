from open3d.ml.torch.models import PointPillars
import open3d.ml as ml3d
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from asd_dataset import AmodalSynthDriveDataset
from open3d.ml.torch.pipelines import ObjectDetection
#from PointPillars.model import PointPillars

class LiDARDetection(nn.Module):

    def __init__(self, cfg_file):
        super().__init__()
        self.cfg = ml3d.utils.Config.load_from_file(cfg_file)
        self.backbone = PointPillars(**self.cfg.model)

    def forward(self, x):
        return self.backbone.call(x, training=True)


def train(data_loader, 
          num_epochs, 
          lr=0.001, 
          momentum=0.9, 
          out_path="./pointpillars_asd.pth"):
    
    net = LiDARDetection()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9
                                )
    acc_loss = 0.0
    for ep in num_epochs:
        print(f"Begin epoch {ep+1}")
        loss = epoch(
                data_loader=data_loader,
                optimizer=optimizer,
                criterion=criterion)
        print(f"[Epoch {ep+1}] loss: {loss}")
        acc_loss += loss

    print("Finished training. Saving model...")    
    torch.save(net.state_dict(), out_path)
    print("Model saved.")
        

def epoch(data_loader, net, optimizer, criterion):
    running_loss = 0.0
    for i, data in enumerate(data_loader,0):
        inputs, labels = data

        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss

def test(data_loader, model_path):
    net = LiDARDetection()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            outputs = net(inputs)


if __name__ == "__main__":
    train_data_root = "/home/jule-magnus/dd2414/Data/AmodalSynthDrive/train"
    val_data_root = "/home/jule-magnus/dd2414/Data/AmodalSynthDrive/val"
    train_ds = AmodalSynthDriveDataset(train_data_root)
    val_ds = AmodalSynthDriveDataset(val_data_root)
    train_dl = DataLoader(train_ds)
    test_dl = DataLoader(val_ds) 

    #model_path = "pointpillars_kitti_epoch_160.pth"
    index = 0
    for input, label in train_dl:
        print(label["front_full_"])
        break

