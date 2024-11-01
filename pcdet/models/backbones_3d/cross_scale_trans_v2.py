from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...utils.spconv_utils import replace_feature, spconv
from torch.nn.init import xavier_uniform_, constant_
import math
import time
tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class cross_scale_trans(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.n_heads = model_cfg.N_HEADS
        self.n_points = model_cfg.N_POINTS
        self.d_chl = model_cfg.D_backbone
        self.n_levels = 4
        self.dist = model_cfg.Dist

        self.d_model = model_cfg.D_embed


        self.input_proj = nn.ModuleList()
        self.sampling_offsets = nn.ModuleList()
        self.attention_weights = nn.ModuleList()
        self.query_proj = nn.ModuleList()
        self.value_proj = nn.ModuleList()
        self.key_proj = nn.ModuleList()
        self.output_proj = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.linear1 = nn.ModuleList()
        self.dropout3 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.dropout4 = nn.ModuleList()
        self.norm3 = nn.ModuleList()
        self.dynamic_pe = DynamicLearnablePositionalEncoding(self.d_model)
        self.fusion_module = nn.ModuleList()

        # \\spconv output
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=0,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        for i in range(self.n_levels):
            self.input_proj.append(nn.Sequential(
                nn.Linear(self.d_chl[i], self.d_model)
            ))

            self.query_proj.append(nn.Linear(self.d_model, self.d_model))
            self.value_proj.append(nn.Linear(self.d_model, self.d_model))
            self.key_proj.append(nn.Linear(self.d_model, self.d_model))

            self.output_proj.append(nn.Linear(self.d_model, self.d_model))
            self.dropout.append(nn.Dropout(model_cfg.Dropout))
            self.norm.append(nn.LayerNorm(self.d_model))

            # \\ffn
            self.linear1.append(nn.Linear(self.d_model, model_cfg.D_ffn))
            self.dropout3.append(nn.Dropout(model_cfg.Dropout))
            self.linear2.append(nn.Linear(model_cfg.D_ffn, self.d_model))
            self.dropout4.append(nn.Dropout(model_cfg.Dropout))
            self.norm3.append(nn.LayerNorm(self.d_model))

            self.fusion_module.append(EnhancedFusion(self.d_model + self.d_chl[i], self.d_chl[i]))
        self._reset_parameters()


    def _reset_parameters(self):

        for i in range(self.n_levels):
                xavier_uniform_(self.value_proj[i].weight.data)
                constant_(self.value_proj[i].bias.data, 0.)
                xavier_uniform_(self.output_proj[i].weight.data)
                constant_(self.output_proj[i].bias.data, 0.)

    def forward_ffn(self, tgt, num):
        tgt2 = self.linear2[num](self.dropout3[num](F.relu(self.linear1[num](tgt))))
        tgt = tgt + self.dropout4[num](tgt2)
        tgt = self.norm3[num](tgt)
        return tgt

    def forward(self, batch_dict):

        ms_features = batch_dict['multi_scale_3d_features']
        for num, lvl in enumerate(ms_features):
            # \\get input vector with same dim
            features = ms_features[lvl].features
            src = self.input_proj[num](features)
            src_shape = src.shape[0]
            #  \\position encode
            fea_shape = ms_features[lvl].spatial_shape
            crt_normalizer = torch.tensor(fea_shape, device=src.device)
            crt_indice = ms_features[lvl].indices
            crt_indice = crt_indice[:, 1:4].float()
            voxel_coods = crt_indice / (crt_normalizer - 1)
            positional_encodings = self.dynamic_pe(voxel_coods)
            src += positional_encodings

            neighboring_coords, neighboring_features = find_neighboring_voxels(crt_indice, src, self.n_points[num],
                                                                               self.dist[num])

            # \\get query and value
            query = self.query_proj[num](neighboring_features[:, 0:1, :])
            key = self.key_proj[num](neighboring_features)
            value = self.value_proj[num](neighboring_features)

            query = query.view(-1, self.n_heads, 1, self.d_model // self.n_heads)
            key = key.view(-1, self.n_heads, self.n_points[num], self.d_model // self.n_heads)
            value = value.view(-1, self.n_heads, self.n_points[num], self.d_model // self.n_heads)

            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)  # [N, H, 1, M]
            attention_weights = F.softmax(attention_scores, dim=-1)  # [N, H, 1, M]

            head_output = torch.matmul(attention_weights, value)  # [N, H, 1, head_dim]
            head_output = head_output.squeeze(2)  # [N, H, head_dim]
            concat_output = head_output.view(-1, self.d_model)

            # query_ = query.view(-1, self.n_heads*self.d_chl[num])
            out = self.output_proj[num](concat_output)
            tgt = src + self.dropout[num](out)
            tgt = self.norm[num](tgt)
            tgt = self.forward_ffn(tgt, num)

            fused_features = self.fusion_module[num](features, tgt)


            ms_features[lvl] = ms_features[lvl].replace_feature(fused_features)

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
        out_bev = self.conv_out(ms_features[lvl])
        batch_dict.update({
            'encoded_spconv_tensor': out_bev,
            'encoded_spconv_tensor_stride': 8
        })

        return batch_dict


class EnhancedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhancedFusion, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, original_features, transformed_features):
        # Concatenate the original features from PV-RCNN and the transformed features
        fused_features = torch.cat([original_features, transformed_features], dim=1)

        # Process the concatenated features through convolutional layers
        fused_features = self.conv1(fused_features)
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