import argparse
import glob
import time
from pathlib import Path


import open3d as o3d
from open3d import geometry as o3d_geometry
from visual_utils import open3d_vis_utils as V
OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

import numpy as np
import torch
import os
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.kitti.kitti_dataset_demo import KittiDataset_demo
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from img2avi import img2avi


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float64).reshape(-1, 4) #float64 for real data, float32 for dataset
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'raw_points': points,
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


import numpy as np

import numpy as np

import numpy as np


def points_in_boxes(points, boxes):
    # points: (N, 3) array of points
    # boxes: (1, M, 7) array of boxes with [x, y, z, dx, dy, dz, yaw]
    # 首先处理boxes的维度，确保它是 (M, 7)
    boxes = boxes.squeeze(0)  # 移除第一维，使boxes成为 (M, 7)

    centers = boxes[:, :3]
    lengths = boxes[:, 3:6]
    yaws = boxes[:, 6]

    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)

    # 扩展 points 维度
    points_expanded = np.expand_dims(points, 1)  # (N, 1, 3)

    # 计算 points 在 boxes 坐标系中的位置
    dx = points_expanded[:, :, 0] - centers[:, 0]
    dy = points_expanded[:, :, 1] - centers[:, 1]
    dz = points_expanded[:, :, 2] - centers[:, 2]

    p_x = dx * cos_yaw - dy * sin_yaw
    p_y = dx * sin_yaw + dy * cos_yaw
    p_z = dz

    # 检查点是否在框内
    mask_x = np.abs(p_x) <= lengths[:, 0] / 2
    mask_y = np.abs(p_y) <= lengths[:, 1] / 2
    mask_z = np.abs(p_z) <= lengths[:, 2] / 2

    mask = mask_x & mask_y & mask_z
    return np.any(mask, axis=1)  # 检查每个点是否在任何框内


# 示例使用方式
# points = np.random.rand(100, 3) * 100  # 随机生成点
# boxes = np.array([[[50, 50, 50, 20, 20, 20, 0]]])  # 单个框，无旋转
# mask = points_in_boxes(points, boxes)
# foreground_points = points[mask]


# # 简单测试
# points = np.array([[1, 2, 3], [4, 5, 6], [1.5, 2.5, 3.5]])
# boxes = np.array([[[2, 2, 3, 2, 2, 2, 0]]])  # 单个框
# mask = points_in_boxes(points, boxes)
# print("Filtered points:", points[mask])


# 示例使用方式
# points = np.random.rand(100, 3) * 100  # 随机生成点
# boxes = np.array([[[50, 50, 50, 20, 20, 20, 0.5]]])  # 单个框，无旋转
# mask = points_in_boxes(points, boxes)
# foreground_points = points[mask]


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pv_rcnn.yaml',
    #                     help='specify the config for demo')
    # parser.add_argument('--data_path', type=str, default='../data/kitti',
    #                     help='specify the point cloud data file or directory')

    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_pp_multihead.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../data/nuscenes',
                        help='specify the point cloud data file or directory')

    # '../output/kitti_models/VPfusionRCNN_kitti/default/ckpt/softmax_55_4096.pth'
    parser.add_argument('--ckpt', type=str,
                        default='../output/pp_multihead_nds5823_updated.pth',
                        help='specify the pretrained model')

    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():

    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    if cfg['DATA_CONFIG'].DATASET == 'KittiDataset':
        demo_dataset = KittiDataset_demo(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), logger=logger
        )
    else:
        demo_dataset = NuScenesDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), logger=logger
        )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    file_dir = args.data_path + '/capture/'
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='results', width=800, height=600, visible=True)

    # 定义四种颜色
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])

    for idx in range(len(demo_dataset) - 3):  # 确保有足够的帧可以显示
        vis.clear_geometries()
        for i in range(4):  # 处理连续四帧
            data_dict = demo_dataset[idx + i]
            data_dict = demo_dataset.collate_batch([data_dict])
            points = np.array(data_dict['points'][:, 1:4])

            # 过滤背景点
            foreground_mask = points_in_boxes(points, np.array(data_dict['gt_boxes']))
            foreground_points = points[foreground_mask]

            # 创建点云对象
            pcd = o3d_geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(foreground_points)


            # 设置当前帧的颜色
            pcd.colors = o3d.utility.Vector3dVector(np.tile(colors[i], (len(pcd.points), 1)))

            vis.get_render_option().point_size = 1.5
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
        # 改变视角
        ctr = vis.get_view_control()
        pc_path = args.data_path + '/viewpoint.json'
        param = o3d.io.read_pinhole_camera_parameters(pc_path)
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(file_dir + f"{idx}.jpg", do_render=True)

    vis.destroy_window()
    img2avi(file_dir)
    logger.info('Demo done.')



if __name__ == '__main__':
    main()
