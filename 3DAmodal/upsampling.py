import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


from pointpillars.utils import read_points, read_calib, bbox3d2corners
from pointpillars.utils.process import group_rectangle_vertexs, group_plane_equation
from pointpillars.model import PointPillars

"""
From PointPillars original repo
"""
CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 

def points_in_bboxes(points, plane_equation_params):
    '''
    points: shape=(N, 3)
    plane_equation_params: shape=(n, 6, 4)
    return: shape=(N, n), bool
    '''
    N, n = len(points), len(plane_equation_params)
    m = plane_equation_params.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            bbox_plane_equation_params = plane_equation_params[j]
            for k in range(m):
                a, b, c, d = bbox_plane_equation_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks
"""
end original PointPillars code
"""

def setup_input(ckpt, pc_path, calib_path, img_path):
    model = PointPillars(nclasses=len(CLASSES)).cuda()
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    pc = read_points(pc_path)
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc).cuda()

    calib_info = read_calib(calib_path)
    
    img = cv2.imread(img_path, 1)
    
    return model, pc, pc_torch, calib_info, img

def map_pc_to_img_space(pc, calib_info):
    tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float128)
    r0_rect = calib_info['R0_rect'].astype(np.float128)
    P2 = calib_info['P2'].astype(np.float128)

    lidar_points_projected = np.dot(tr_velo_to_cam, pc.T).T
    lidar_points_projected = np.dot(r0_rect, lidar_points_projected.T).T
    lidar_points_projected = np.dot(P2, lidar_points_projected.T).T
    lidar_points_projected /= lidar_points_projected[:, 2][:, np.newaxis]

    return lidar_points_projected


def test_depthmask(ckpt, pc_path, calib_path, img_path):
    model, pc, pc_torch, calib, img = setup_input(ckpt, pc_path, calib_path, img_path)
    thresh = 0.7
        
    # retrieve bboxes, use those with confidence > thresh
    result3d = model(batched_pts=[pc_torch], 
                              mode='test')[0]
    scores = result3d['scores']
    valid_idx = torch.where(torch.tensor(scores) > thresh)

    # find points that lie within bboxes
    bboxes_corners = bbox3d2corners(result3d['lidar_bboxes']) # (n, 8, 3)
    bbox_group_rectangle_vertexs = group_rectangle_vertexs(bboxes_corners) # (n, 6, 4, 3)
    group_plane_equation_params = group_plane_equation(bbox_group_rectangle_vertexs)
    masks = points_in_bboxes(pc[:, :3], group_plane_equation_params) # masks (N, n), bool; N=#points, n=#bboxes


    # initialize depth mask the size of the image
    depth_mask = np.zeros((img.shape[0], img.shape[1]))

    # iterate over valid bboxes
    for idx in valid_idx[0]:
        points_in_box = np.array([pc[i] for i in range(len(pc)) if masks[i][idx] == True])
        points_in_box_img = map_pc_to_img_space(points_in_box, calib)

        x_img = np.floor(points_in_box_img[:, 0]).astype(np.int32)
        y_img = np.floor(points_in_box_img[:, 1]).astype(np.int32)

        z = points_in_box[:, 2] # z coords from pc space to keep depth info
        z_normed = (z - pcd_limit_range[2]) / (pcd_limit_range[5] - pcd_limit_range[2])

        # map x, y coordinates to 
        x_min, x_max = np.floor(np.min(x_img)).astype(np.int32), np.floor(np.max(x_img)).astype(np.int32)
        y_min, y_max = np.floor(np.min(y_img)).astype(np.int32), np.floor(np.max(y_img)).astype(np.int32)

        # upsample point cloud
        X = np.linspace(x_min, x_max, num=x_max-x_min)
        Y = np.linspace(y_min, y_max, num=y_max-y_min)
        X, Y = np.meshgrid(X, Y)
        interp = LinearNDInterpolator(list(zip(x_img, y_img)), z_normed)
        Z = interp(X, Y)
        Z = np.nan_to_num(Z, nan=0.0)

        depth_mask[y_min:y_max, x_min:x_max] = Z

    return depth_mask
                        
        
if __name__ == '__main__':
    ckpt = "model_checkpoints/pretrained_pointpillars_epoch_160.pth"
    pc_path = "/home/jule-magnus/dd2414/Data/kitti/testing/velodyne_reduced/000000.bin"
    calib_path = "/home/jule-magnus/dd2414/Data/kitti/testing/calib/000000.txt"
    img_path = "/home/jule-magnus/dd2414/Data/kitti/testing/image_2/000000.png"

    depth_mask =create_depthmask(ckpt, pc_path, calib_path, img_path)
    plt.imshow(depth_mask)
    plt.show()
    

