from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import open3d

# # == == == == == debug == == == == == == == == =
# # --------------------------------
#
#
#
# # Create coordinate grids along each dimension
# x = torch.arange(3)-1
# y = torch.arange(3)-1
# z = torch.arange(3)-1
#
# # Use torch.meshgrid to generate coordinate grids
# grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
#
# # Concatenate the coordinate grids to form a tensor
# cube = torch.stack((grid_x, grid_y, grid_z), dim=-1)
# coordinates = cube.reshape(-1, 3)
#
# cube = cube.to(dtype=torch.float32)
#
# sampling_grids = torch.tensor([2.0, 1.0, 1.0])
# cube_sample = cube[:,2,1,1]
#
# sampling_grids_norm = (2 * sampling_grids / (3-1)) - 1
#
# swap_matrix = torch.tensor([[0.0, 0.0, 1.0],
#                            [0.0, 1.0, 0.0],
#                            [1.0, 0.0, 0.0]])
#
# sampling_grids_norm = torch.matmul(sampling_grids_norm, swap_matrix)
# sampling_grids_norm = sampling_grids_norm.view(1,1,1,1,3)
#
# sampling_value_l_ = F.grid_sample(cube.unsqueeze(0), sampling_grids_norm,
#                                   mode='bilinear', padding_mode='zeros', align_corners=True)
#
# print("Sample point:", sampling_grids)
# print("Sample result:", sampling_value_l_)
# result_tensor = torch.cat([coordinates, sampling_grids.unsqueeze(0)], dim=0)
#
# pointshow = result_tensor.cpu().numpy()
# point_cloud = open3d.geometry.PointCloud()
# point_cloud.points = open3d.utility.Vector3dVector(pointshow)
#
# # Set colors for each point
# red_color = np.array([1, 0, 0])
# colors = np.tile(red_color, (len(pointshow), 1))
# point_cloud.colors = open3d.utility.Vector3dVector(colors)  # Correct assignment
#
#
# # Set point size
# point_cloud.paint_uniform_color(red_color)  # Use the red_color for uniform coloring
# point_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#
# # Set visualization settings with point size
# visualizer = open3d.visualization.Visualizer()
# visualizer.create_window()
# visualizer.add_geometry(point_cloud)
# render_option = visualizer.get_render_option()
# render_option.point_size = 50  # Set the point size here
#
# visualizer.run()
# visualizer.destroy_window()
#
# # --------------------------------
# # == == == == == debug == == == == == == == == =


# == == == == == debug 2 == == == == == == == == =
# --------------------------------

import torch
import torch.nn.functional as F

# 创建一个全零的体素特征图，大小为 (1, 16, 4, 5, 8)
voxel_features = torch.zeros((1, 16, 4, 5, 8), dtype=torch.float)
crt_normalizer = torch.tensor([4,5,8], dtype=torch.float)

# 设置非空体素 [2, 3, 4] 的特征值
voxel_features[0, :, 2, 3, 4] = torch.arange(16, dtype=torch.float)

# 设置非空体素 [1, 2, 7] 的特征值
voxel_features[0, :, 1, 2, 7] = torch.arange(16, 32, dtype=torch.float)

aa = voxel_features[:,:,2, 3, 4]

grid = torch.tensor([[[[[2, 3, 4]]]] * 26], dtype=torch.float)
sampling_grids_2 = 2*grid/ (crt_normalizer - 1) - 1

swap_matrix = torch.tensor([[0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0]])
sampling_grids_2_ = torch.matmul(sampling_grids_2, swap_matrix)

# 使用 grid_sample 进行单点三维体素采样
output = F.grid_sample(voxel_features, sampling_grids_2_, mode='bilinear',padding_mode='zeros', align_corners=True)


print("输出特征图（单点采样，16 个通道）：")
print(output)






