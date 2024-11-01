import open3d as o3d
import numpy as np

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=800, height=600)
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=800, height=600)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    pc_path = '/home/xrd/PycharmProjects/VPfusion/data/nuscenes/v1.0-trainval/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915265947124.pcd.bin'
    points = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 5)  # 5 for nuscenes
    # pc_path = '/media/xrd/本地磁盘/data/points/000001.bin'
    points = np.fromfile(str(pc_path), dtype=np.float64).reshape(-1, 4)  # 4 for kitti
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 传入自己当前的pcd文件
    save_view_point(pcd, "/media/xrd/本地磁盘/data/points/viewpoint.json")  # 保存好得json文件位置
    load_view_point(pcd, "/media/xrd/本地磁盘/data/points/viewpoint.json")  # 加载修改时较后的pcd文件
