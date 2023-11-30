import torch
from datasets.asd_dataset import AmodalSynthDriveDataset
import numpy as np
from datasets.dataloader import get_dataloader
from pointpillars.model import PointPillars

class PointPillars_FT(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pointpillars = PointPillars()
        if torch.cuda.is_available():
            self.pointpillars = self.pointpillars.cuda()
        self.pointpillars.load_state_dict(torch.load("/home/jule-magnus/dd2414/3damodal/3DAmodal/model_checkpoints/pretrained_pointpillars_epoch_160.pth"))
        #adapt the final layer
        self.final_layer = torch.nn.Linear(num_classes, num_classes)

    def forward(self, batched_pts,
                mode="test",
                batched_gt_bboxes=None,
                batched_gt_labels=None):
        x = self.pointpillars(batched_pts=batched_pts, 
                              mode="train",
                              batched_gt_bboxes=gt_bboxes,
                              batched_gt_labels=batched_gt_labels)
        x = self.final_layer(x)
        return x
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 38
    model = PointPillars_FT(num_classes)
    model = model.to(device)

    batch_size=1

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = AmodalSynthDriveDataset("/home/jule-magnus/dd2414/Data/AmodalSynthDrive/train")
    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        partition="lidar",
        shuffle=True
    )

    num_epochs = 10
    for epoch in range(num_epochs):
        for data in train_loader:
            pts = data["batched_pts"]                
            gt_bboxes = data["batched_gt_bboxes"]
            labels = data["batched_labels"]
            for i in range(batch_size):
                pts[i].to(device)
                gt_bboxes[i].to(device)
                labels[i].to(device)


            optimizer.zero_grad()
            outputs = model(pts, "train", gt_bboxes, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    
    torch.save(model.state_dict("pretrained/pointpillars_asd.pth"))