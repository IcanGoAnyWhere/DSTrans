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
    """
    Use muti-level transformer encoder to get local attention with same QKV paras.
    """

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.n_heads = model_cfg.N_HEADS
        self.n_points = model_cfg.N_POINTS
        self.d_chl = model_cfg.D_backbone
        self.n_levels = 4
        self.dist = model_cfg.Dist
        self.d_model = model_cfg.D_embed

        self.input_proj = nn.ModuleList()
        self.fusion_module = nn.ModuleList()

        self.spconv = VoxelBackBone8x(model_cfg, input_channels, grid_size)
        self.num_point_features = self.spconv.num_point_features
        self.backbone_channels = self.spconv.backbone_channels

        # Use MultiheadAttention directly for attention computation
        self.attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, dropout=0.1)
        self.dynamic_pe = DynamicLearnablePositionalEncoding(self.d_model)

        # \\ffn
        self.ffn_linear1 = nn.Linear(self.d_model, model_cfg.D_ffn)
        self.ffn_dropout1 = nn.Dropout(model_cfg.Dropout)
        self.ffn_linear2 = nn.Linear(model_cfg.D_ffn, self.d_model)
        self.ffn_dropout2 = nn.Dropout(model_cfg.Dropout)
        self.ffn_norm1 = nn.LayerNorm(self.d_model)

        for i in range(self.n_levels):
            self.input_proj.append(nn.Sequential(
                nn.Linear(self.d_chl[i], self.d_model)
            ))
            self.fusion_module.append(EnhancedFusion(self.d_model, self.d_chl[i]))


        # \\spconv output
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=0,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )


    def forward_ffn(self, tgt):
        tgt2 = self.ffn_linear2(self.ffn_dropout1(F.relu(self.ffn_linear1(tgt))))
        tgt = tgt + self.ffn_dropout2(tgt2)
        tgt = self.ffn_norm1(tgt)
        return tgt

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

        # apply sparse conv layer and local attention on 1st level
        x_conv1 = self.spconv.conv1(input_tensor)
        # \\get input vector with same dim
        features = x_conv1.features
        src = self.input_proj[0](features)
        #  \\position encode
        fea_shape = x_conv1.spatial_shape
        crt_normalizer = torch.tensor(fea_shape, device=src.device)
        crt_indice = x_conv1.indices
        crt_indice = crt_indice[:, 1:4].float()
        voxel_coods = crt_indice / (crt_normalizer - 1)
        positional_encodings = self.dynamic_pe(voxel_coods)
        src += positional_encodings
        neighboring_coords, neighboring_features = find_neighboring_voxels(crt_indice, src, self.n_points[0],
                                                                           self.dist[0])
        query_input = neighboring_features[:, 0:1, :].view(1, -1, self.d_model)
        key_input = value_input = neighboring_features.view(self.n_points[0], -1, self.d_model)
        attn_output, _ = self.attention(query_input, key_input, value_input)
        attn_output = attn_output.transpose(0, 1)
        tgt = self.forward_ffn(attn_output).squeeze(1)
        fused_features = self.fusion_module[0](features, tgt)
        x_conv1 = x_conv1.replace_feature(fused_features)

        # apply sparse conv layer and local attention on 2nd level
        x_conv2 = self.spconv.conv2(x_conv1)
        # \\get input vector with same dim
        features = x_conv2.features
        src = self.input_proj[1](features)
        #  \\position encode
        fea_shape = x_conv2.spatial_shape
        crt_normalizer = torch.tensor(fea_shape, device=src.device)
        crt_indice = x_conv2.indices
        crt_indice = crt_indice[:, 1:4].float()
        voxel_coods = crt_indice / (crt_normalizer - 1)
        positional_encodings = self.dynamic_pe(voxel_coods)
        src += positional_encodings
        neighboring_coords, neighboring_features = find_neighboring_voxels(crt_indice, src, self.n_points[1],
                                                                           self.dist[1])
        query_input = neighboring_features[:, 0:1, :].view(1, -1, self.d_model)
        key_input = value_input = neighboring_features.view(self.n_points[1], -1, self.d_model)
        attn_output, _ = self.attention(query_input, key_input, value_input)
        attn_output = attn_output.transpose(0, 1)
        tgt = self.forward_ffn(attn_output).squeeze(1)
        fused_features = self.fusion_module[1](features, tgt)
        x_conv2 = x_conv2.replace_feature(fused_features)

        # apply sparse conv layer and local attention on 3rd level
        x_conv3 = self.spconv.conv3(x_conv2)
        # \\get input vector with same dim
        features = x_conv3.features
        src = self.input_proj[2](features)
        #  \\position encode
        fea_shape = x_conv3.spatial_shape
        crt_normalizer = torch.tensor(fea_shape, device=src.device)
        crt_indice = x_conv3.indices
        crt_indice = crt_indice[:, 1:4].float()
        voxel_coods = crt_indice / (crt_normalizer - 1)
        positional_encodings = self.dynamic_pe(voxel_coods)
        src += positional_encodings
        neighboring_coords, neighboring_features = find_neighboring_voxels(crt_indice, src, self.n_points[2],
                                                                           self.dist[2])
        query_input = neighboring_features[:, 0:1, :].view(1, -1, self.d_model)
        key_input = value_input = neighboring_features.view(self.n_points[1], -1, self.d_model)
        attn_output, _ = self.attention(query_input, key_input, value_input)
        attn_output = attn_output.transpose(0, 1)
        tgt = self.forward_ffn(attn_output).squeeze(1)
        fused_features = self.fusion_module[2](features, tgt)
        x_conv3 = x_conv3.replace_feature(fused_features)

        # apply sparse conv layer and local attention on 4th level
        x_conv4 = self.spconv.conv4(x_conv3)
        # \\get input vector with same dim
        features = x_conv4.features
        src = self.input_proj[3](features)
        #  \\position encode
        fea_shape = x_conv4.spatial_shape
        crt_normalizer = torch.tensor(fea_shape, device=src.device)
        crt_indice = x_conv4.indices
        crt_indice = crt_indice[:, 1:4].float()
        voxel_coods = crt_indice / (crt_normalizer - 1)
        positional_encodings = self.dynamic_pe(voxel_coods)
        src += positional_encodings
        neighboring_coords, neighboring_features = find_neighboring_voxels(crt_indice, src, self.n_points[3],
                                                                           self.dist[3])
        query_input = neighboring_features[:, 0:1, :].view(1, -1, self.d_model)
        key_input = value_input = neighboring_features.view(self.n_points[2], -1, self.d_model)
        attn_output, _ = self.attention(query_input, key_input, value_input)
        attn_output = attn_output.transpose(0, 1)
        tgt = self.forward_ffn(attn_output).squeeze(1)
        fused_features = self.fusion_module[3](features, tgt)
        x_conv4 = x_conv4.replace_feature(fused_features)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

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

        # ==========debug=================
        # --------------------------------

        # import open3d
        # samplingpoints = sampling_locations[1,:,:,:]
        # samplingpoints = samplingpoints.view(-1, 3).detach().cpu().numpy()
        # pointshow = crt_indice.view(-1, 3).cpu().numpy()
        # merged_array = np.concatenate((samplingpoints, pointshow), axis=0)
        #
        # point_cloud = open3d.geometry.PointCloud()
        # point_cloud.points = open3d.utility.Vector3dVector(pointshow)
        # open3d.visualization.draw_geometries([point_cloud])

        # --------------------------------
        # ==========debug=================



        return batch_dict


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