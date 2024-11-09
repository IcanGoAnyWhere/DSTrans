from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...utils.spconv_utils import replace_feature, spconv
from .spconv_backbone import VoxelBackBone8x
import math
import time


class cross_scale_trans(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.n_heads = model_cfg.N_HEADS
        self.n_points = model_cfg.N_POINTS
        self.d_chl = model_cfg.D_backbone
        self.d_model = model_cfg.D_embed
        self.dist = model_cfg.Dist

        # Backbone and attention modules
        self.spconv = VoxelBackBone8x(model_cfg, input_channels, grid_size)
        self.input_proj = nn.ModuleList([
            nn.Linear(channels, self.d_model) for channels in self.d_chl
        ])
        self.fusion_module = nn.ModuleList([
            EnhancedFusion(self.d_model, self.d_chl[i]) for i in range(4)
        ])
        self.attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads)
        self.dynamic_pe = DynamicLearnablePositionalEncoding(self.d_model)
        self.ffn = FFN(self.d_model, model_cfg.D_ffn, model_cfg.Dropout)

        self.num_point_features = self.spconv.num_point_features
        self.backbone_channels = self.spconv.backbone_channels

    def apply_attention(self, x_conv, input_proj, fusion_module, level):
        features = x_conv.features
        fea_shape = x_conv.spatial_shape
        crt_normalizer = torch.tensor(fea_shape, device=features.device)
        crt_indice = x_conv.indices
        crt_indice = crt_indice[:, 1:4].float()
        positional_encodings = self.dynamic_pe(crt_indice / (crt_normalizer - 1))
        src = input_proj(features) + positional_encodings
        neighboring_coords, neighboring_features = find_neighboring_voxels_optimized(
            crt_indice, src, self.n_points[level], self.dist[level]
        )
        query, key_value = neighboring_features[:, 0:1].view(1, -1, self.d_model), \
                           neighboring_features.view(self.n_points[0], -1, self.d_model)
        attn_output, _ = self.attention(query, key_value, key_value)
        fused_features = fusion_module(features, self.ffn(attn_output.transpose(0, 1)).squeeze(1))
        return x_conv.replace_feature(fused_features)

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        # get input tensor of voxel feature
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.spconv.sparse_shape,
            batch_size=batch_size
        )
        input_tensor = self.spconv.conv_input(input_sp_tensor)

        x_conv1 = self.spconv.conv1(input_tensor)
        x_conv1 = self.apply_attention(x_conv1, self.input_proj[0], self.fusion_module[0], level=0)

        x_conv2 = self.spconv.conv2(x_conv1)
        x_conv2 = self.apply_attention(x_conv2, self.input_proj[1], self.fusion_module[1], level=1)

        x_conv3 = self.spconv.conv3(x_conv2)
        x_conv3 = self.apply_attention(x_conv3, self.input_proj[2], self.fusion_module[2], level=2)

        x_conv4 = self.spconv.conv4(x_conv3)
        x_conv4 = self.apply_attention(x_conv4, self.input_proj[3], self.fusion_module[3], level=3)

        # Apply final output layer
        out = self.spconv.conv_out(x_conv4)

        # Keep batch_dict.update as is
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        return self.norm(tgt + self.dropout2(self.linear2(self.dropout1(F.relu(self.linear1(tgt))))))


class EnhancedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhancedFusion, self).__init__()
        self.Linear1 = nn.Linear(in_channels, out_channels)
        self.Linear2 = nn.Linear(out_channels * 2, out_channels)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, original_features, transformed_features):
        # Concatenate the original features from PV-RCNN and the transformed features
        transformed_features = self.Linear1(transformed_features)
        fused_features = torch.cat([original_features, transformed_features], dim=1)

        # Process the concatenated features through convolutional layers
        fused_features = self.Linear2(fused_features)
        fused_features = self.norm1(fused_features)
        fused_features = self.relu(fused_features)

        return fused_features


class DynamicLearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(DynamicLearnablePositionalEncoding, self).__init__()
        self.linear1 = nn.Linear(3, d_model // 2)
        self.linear2 = nn.Linear(d_model // 2, d_model)
        self.activation = nn.ReLU()

    def forward(self, voxel_coords):
        x = self.linear1(voxel_coords)
        x = self.activation(x)
        positional_encodings = self.linear2(x)
        return positional_encodings


def find_neighboring_voxels(crt_indice, features, M, manhattan_dist):

    start_time = time.time()

    # 获取体素数量和特征通道数
    N, C = features.shape

    # 计算曼哈顿距离并转换为浮点类型
    coord_diff = crt_indice[:, None, :] - crt_indice[None, :, :]  # (N, N, 3)
    manhattan_distances = coord_diff.abs().sum(dim=-1).float()  # (N, N)

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
    # print(f"Execution time: {end_time - start_time:.4f} seconds")

    return neighboring_coords, neighboring_features

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
    # print(f"Execution time: {end_time - start_time:.4f} seconds")

    return neighboring_coords, neighboring_features