from torch.utils.data import Dataset
from .KINS_dataset import KINS
from .kitti import Kitti

from tqdm import tqdm


class AmodalLiDAR(Dataset):
    def __init__(self, 
                 root_dir='/home/jule-magnus/dd2414/Data/', 
                 mode='train', 
                 transform=None,
                 pts_prefix='velodyne_reduced'):
        self.mode = mode
        self.amodal_dataset = KINS(root_dir=root_dir + "KINS",
                                   mode=mode,
                                   transform=transform)
        self.lidar_dataset = Kitti(data_root=root_dir + "kitti",
                                   split=mode,
                                   pts_prefix=pts_prefix,
                                   shuffle=False)
        
        assert (len(self.amodal_dataset) == len(self.lidar_dataset)), f"length of datasets is not compatible, amodal: len={len(self.amodal_dataset)}, lidar: len={len(self.lidar_dataset)}"

        self.get_data_mapping()

    def get_data_mapping(self):
        self.amodal_to_lidar_mapping = {}
        with open(f"/home/jule-magnus/dd2414/3damodal/3DAmodal/datasets/amodal_to_lidar_mapping_{self.mode}.txt", 'r') as txtfile:
            data = txtfile.readlines()
        for line in data:
            amodal, lidar = line.split(",")
            self.amodal_to_lidar_mapping[int(amodal)] = int(lidar)
        
        
    def __getitem__(self, index):
        if index not in self.amodal_to_lidar_mapping.keys():
            index += 1
        amodal = self.amodal_dataset[index]
        lidar = self.lidar_dataset[self.amodal_to_lidar_mapping[index]]
        return {"amodal": amodal, 
                "lidar": lidar}
    
    def __len__(self):
        return len(self.amodal_dataset)
    

def create_amodal_to_lidar_mapping_file(mode="train", data_root="/home/jule-magnus/dd2414/Data"):
    kins = KINS(root_dir=data_root + "KINS",
                mode=mode)
    kitti = Kitti(data_root=data_root + "kitti",
                  split=mode,
                  pts_prefix="velodyne_reduced",
                  shuffle=False)
    data_length = len(kins)
    
    amodal_to_lidar_mapping = {}
    for i in tqdm(range(data_length)):
        d_key = list(kins[i][1].keys())[0]
        if isinstance(d_key, int):
            amodal_img_id = kins[i][1][d_key]["image_id"]
            if amodal_img_id not in amodal_to_lidar_mapping.keys():
                amodal_to_lidar_mapping[amodal_img_id] = {"amodal": i, "lidar": -1}
            else:
                amodal_to_lidar_mapping[amodal_img_id]["amodal"] = i
        lidar_img_id = kitti[i]["image_info"]["image_idx"]
        if lidar_img_id not in amodal_to_lidar_mapping.keys():
            amodal_to_lidar_mapping[lidar_img_id] = {"amodal": -1, "lidar": i}
        else:
            amodal_to_lidar_mapping[lidar_img_id]["lidar"] = i
        
    print(len(amodal_to_lidar_mapping.keys()))

    with open(f"amodal_to_lidar_mapping_{mode}.txt", "w") as txtfile:
        for k, val in amodal_to_lidar_mapping.items():
            txtfile.write(str(val["amodal"])+ "," + str(val["lidar"]) + "\n")



if __name__ == "__main__":
    all_data = AmodalLiDAR()
    print(all_data[150]['amodal'][1][94552]["image_id"])
    print(all_data[150]['lidar']["image_info"]['image_idx'])

