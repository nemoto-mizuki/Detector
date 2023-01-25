from sympy import Point
import torch
import numpy as np
import spconv.pytorch as spconv 
from torch import nn
from torch.utils.data import Dataset
import open3d as o3d
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
import os 
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Net(nn.Module):
    def __init__(self, i_channels):
        super().__init__()
        
        self.conv1 = spconv.SparseConv4d(i_channels, 16, (1, 3, 3, 3), stride=1, padding=0)
    
    def forward(self, input, coords):
        input_sp_tensor = spconv.SparseConvTensor(
            features=input,
            indices=coords,
            spatial_shape=[1000, 1000, 40],
            batch_size=1
        )
        y  = self.conv1(input_sp_tensor)

        return y

from cumm import tensorview as tv
points_0 = np.load('/media/WD6THDD/Detector/data/waymo/waymo_processed_data_v0_5_0/segment-10203656353524179475_7625_000_7645_000_with_camera_labels/0000.npy')
points_1 = np.load('/media/WD6THDD/Detector/data/waymo/waymo_processed_data_v0_5_0/segment-10203656353524179475_7625_000_7645_000_with_camera_labels/0004.npy')
points_2 = np.load('/media/WD6THDD/Detector/data/waymo/waymo_processed_data_v0_5_0/segment-10203656353524179475_7625_000_7645_000_with_camera_labels/0008.npy')
points_3 = np.load('/media/WD6THDD/Detector/data/waymo/waymo_processed_data_v0_5_0/segment-10203656353524179475_7625_000_7645_000_with_camera_labels/0012.npy')
len0 = len(points_0)
len1 = len(points_1)
len2 = len(points_2)
len3 = len(points_3)
t0 = np.zeros((len0, 1), np.float)
t1 = np.ones((len1, 1), np.float)
t2 = np.ones((len2, 1), np.float)+1
t3 = np.ones((len3, 1), np.float)+2
points_0 = torch.from_numpy(np.concatenate((points_0, t0), axis=1))
points_1 = torch.from_numpy(np.concatenate((points_1, t1), axis=1))
points_2 = torch.from_numpy(np.concatenate((points_2, t2), axis=1))
points_3 = torch.from_numpy(np.concatenate((points_3, t3), axis=1))
points = [points_0, points_1, points_2, points_3]
# points = np.concatenate((points_0, points_1, points_2, points_3))
# pc_th = torch.from_numpy(points)
grid_size = [0.1, 0.1, 0.1]
pointcloud_range = np.array([-80, -80, -6, 80, 80, 6])
shape = (pointcloud_range[3:] - pointcloud_range[:3]) / grid_size
shape = np.round(shape).astype(np.int32).tolist()
# index = np.ones(shape, dtype=np.int32)

gen = PointToVoxel(
    vsize_xyz=grid_size,
    coors_range_xyz=pointcloud_range,
    num_point_features=7,
    max_num_voxels=40000,
    max_num_points_per_voxel=5
)
# pc_th = torch.from_numpy(points)

voxels_th, indices_th, num_p_in_vx_th = [],[],[]

for i in range(len(points)):
    voxels, indices, num_p_in_vx = gen(points[i])
    voxels_th.append([voxels])
    indices_th.append([indices])
    num_p_in_vx_th.append([num_p_in_vx])


sparse_shape = np.asarray(shape[::-1] + [1, 0, 0])
points_mean = voxels_th[:, :, :].sum(dim=1, keepdim=False)
normalizer = torch.clamp_min(num_p_in_vx_th.view(-1, 1), min=1.0).type_as(voxels_th)
points_mean = points_mean / normalizer
feature = points_mean.contiguous()

net = Net()