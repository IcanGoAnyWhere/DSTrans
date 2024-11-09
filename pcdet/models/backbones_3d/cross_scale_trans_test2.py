import torch
import time


def find_neighboring_voxels(crt_indice, features, M, manhattan_dist):
    start_time = time.time()

    # 获取体素数量和特征通道数
    N, C = features.shape

    # 计算曼哈顿距离并转换为浮点类型
    coord_diff = crt_indice[:, None, :] - crt_indice[None, :, :]  # (N, N, 3)
    manhattan_distances = coord_diff.abs().sum(dim=-1).float()  # (N, N)
    print(f" size: {manhattan_distances.element_size() * manhattan_distances.nelement()} bytes")

    # 找到在曼哈顿距离范围内的体素
    within_range = manhattan_distances <= manhattan_dist
    manhattan_distances[~within_range] = float('inf')  # 将超出范围的距离设为 inf

    # 对每个体素选择最小距离的 M 个体素
    distances, indices = torch.topk(-manhattan_distances, k=M, largest=True)
    indices = indices.masked_fill(distances == -float('inf'), -1)  # 将无效索引填充为 -1

    # 初始化存储张量并确保数据类型匹配
    neighboring_coords = torch.zeros((N, M, 3), dtype=crt_indice.dtype, device=crt_indice.device)
    neighboring_features = torch.zeros((N, M, C), dtype=features.dtype, device=features.device)

    # 只填充有效的相邻体素
    valid_mask = indices >= 0  # 有效体素掩码
    neighboring_coords[valid_mask] = crt_indice[indices[valid_mask]]
    neighboring_features[valid_mask] = features[indices[valid_mask]]

    # 结束计时
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    return neighboring_coords, neighboring_features


import torch
import time
from collections import defaultdict


def spatial_hash(voxel_coords, grid_size):
    """将每个体素坐标映射到空间哈希网格中"""
    hashed_voxels = defaultdict(list)
    for idx, coord in enumerate(voxel_coords):
        # 计算哈希桶的坐标（这里假设使用整数除法来确定桶大小）
        hash_key = tuple((coord // grid_size).int().tolist())
        hashed_voxels[hash_key].append(idx)
    return hashed_voxels


import torch
import time


def find_neighboring_voxels_optimized(crt_indice, features, M, manhattan_dist, batch_size=1024):
    start_time = time.time()

    # 获取体素数量和特征通道数
    N, C = features.shape
    device = crt_indice.device

    # 初始化存储张量
    neighboring_coords = torch.zeros((N, M, 3), dtype=crt_indice.dtype, device=device)
    neighboring_features = torch.zeros((N, M, C), dtype=features.dtype, device=device)

    # 分块处理
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        crt_block = crt_indice[i:end_i]  # 当前块的索引
        coord_diff = crt_block[:, None, :] - crt_indice[None, :, :]  # (batch_size, N, 3)

        # 计算曼哈顿距离
        manhattan_distances = coord_diff.abs().sum(dim=-1)  # (batch_size, N)

        # 筛选出在曼哈顿距离范围内的体素
        within_range = manhattan_distances <= manhattan_dist
        manhattan_distances[~within_range] = float('inf')  # 超出范围的距离设为 inf

        # 对每个体素选择最小距离的 M 个体素
        distances, indices = torch.topk(-manhattan_distances, k=M, largest=True)
        indices = indices.masked_fill(distances == -float('inf'), -1)  # 将无效索引填充为 -1

        # 只填充有效的相邻体素
        valid_mask = indices >= 0
        neighboring_coords[i:end_i][valid_mask] = crt_indice[indices[valid_mask]]
        neighboring_features[i:end_i][valid_mask] = features[indices[valid_mask]]

    # 结束计时
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    return neighboring_coords, neighboring_features


# 示例代码
if __name__ == "__main__":
    N = 16000  # 体素数量
    C = 64       # 特征通道数
    M = 5       # 最大相邻体素数量
    D = 10        # 曼哈顿距离

    # 随机生成数据
    crt_indice = torch.randint(0, 100, (N, 3)).cuda().float()
    features = torch.randn(N, C).cuda()


    # 调用函数
    neighboring_coords, neighboring_features = find_neighboring_voxels_optimized(crt_indice, features, M, D,
                                                                                 batch_size=1024)
    # neighboring_coords, neighboring_features = find_neighboring_voxels(crt_indice, features, M, D)
    print(neighboring_coords.shape)





