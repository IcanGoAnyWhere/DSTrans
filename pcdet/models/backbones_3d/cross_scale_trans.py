from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...utils.spconv_utils import replace_feature, spconv
from torch.nn.init import xavier_uniform_, constant_
import math
import time


class cross_scale_trans(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.d_model = model_cfg.D_QUERY
        self.n_heads = model_cfg.N_HEADS
        self.n_points = model_cfg.N_POINTS
        self.d_chl = model_cfg.D_backbone
        self.n_levels = 4

        self.input_proj = nn.ModuleList()
        self.sampling_offsets = nn.ModuleList()
        self.attention_weights = nn.ModuleList()
        self.query_proj = nn.ModuleList()
        self.value_proj = nn.ModuleList()
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

        for i in range(self.n_levels):
            self.input_proj.append(nn.Sequential(
                nn.Linear(self.d_chl[i], self.d_model),
                nn.GroupNorm(self.d_chl[i], self.d_model)
            ))

            self.sampling_offsets.append(nn.Linear(self.d_chl[i], self.n_heads * self.n_points[i] * 3))
            self.attention_weights.append(nn.Linear(self.d_chl[i], self.n_heads * self.n_points[i]))
            self.query_proj.append(nn.Linear(self.d_model, self.d_chl[i]))
            self.value_proj.append(nn.Linear(self.d_model, self.d_chl[i]))
            self.output_proj.append(nn.Linear(self.d_chl[i], self.d_chl[i]))
            self.dropout.append(nn.Dropout(model_cfg.Dropout))
            self.norm.append(nn.LayerNorm(self.d_chl[i]))

            # \\ffn
            self.linear1.append(nn.Linear(self.d_chl[i], model_cfg.D_ffn))
            self.dropout3.append(nn.Dropout(model_cfg.Dropout))
            self.linear2.append(nn.Linear(model_cfg.D_ffn, self.d_chl[i]))
            self.dropout4.append(nn.Dropout(model_cfg.Dropout))
            self.norm3.append(nn.LayerNorm(self.d_chl[i]))

            self.fusion_module.append(EnhancedFusion(self.d_chl[i] * 2, self.d_chl[i]))

        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.n_levels):
            constant_(self.sampling_offsets[i].weight.data, 0.)
            constant_(self.attention_weights[i].weight.data, 0.)
            constant_(self.attention_weights[i].bias.data, 0.)
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
            #  \\position encode
            crt_indice = ms_features[lvl].indices
            crt_indice = crt_indice[:, 1:4].float()
            fea_shape = ms_features[lvl].spatial_shape
            crt_normalizer = torch.tensor(fea_shape, device=src.device)
            voxel_coods = crt_indice / (crt_normalizer - 1)
            positional_encodings = self.dynamic_pe(voxel_coods)
            src += positional_encodings

            src_shape = src.shape[0]
            # \\get query and value
            query = self.query_proj[num](src)
            value = self.value_proj[num](src)

            sampling_offsets = torch.sigmoid(self.sampling_offsets[num](query)) \
                .view(-1, self.n_heads, self.n_points[num], 3) / 2.0

            attention_weights = torch.softmax(self.attention_weights[num](query) \
                .view(-1, self.n_heads, self.n_points[num]), -1)


            # \\crt_normalizer is uesd to norm cood to [0,1]

            value_epd_shape = torch.cat((torch.tensor(value.shape[1],
                                                      device=value.device).unsqueeze(0), crt_normalizer), dim=0)


            # \\expand value to complete cube with zeros, for meeting grid_sample input

            crt_value_epd = torch.full(list(value_epd_shape), 1e-6, device=crt_indice.device)
            crt_value_epd[:, crt_indice[:, 0].long(), crt_indice[:, 1].long(),\
                crt_indice[:, 2].long()] = value.transpose(0, 1)
            crt_value_epd_l = crt_value_epd.unsqueeze(0)
            # fea_test = ms_features[lvl].dense()
            # aa = crt_value_epd_l[:,:,25,407,156]
            # bb = fea_test[:,:,25,407,156]


            # \\norm paras to [0,1]
            # crt_offset = sampling_offsets / (crt_normalizer-1)

            sampling_locations = sampling_offsets + voxel_coods[:, None, None, :]
            swap_matrix = torch.tensor([[0.0, 0.0, 1.0],
                                        [0.0, 1.0, 0.0],
                                        [1.0, 0.0, 0.0]], device=value.device)
            sampling_locations_swap = torch.matmul(sampling_locations, swap_matrix)
            sampling_locations = sampling_locations_swap.unsqueeze(0)

            # \\norm grid to [-1,1]
            sampling_grids = 2 * sampling_locations - 1
            sampling_value_l_ = F.grid_sample(crt_value_epd_l, sampling_grids,
                                              mode='bilinear', padding_mode='zeros', align_corners=True)

            out = (sampling_value_l_ * attention_weights[None, None, :, :, :]).sum(-1).transpose(1, 2).sum(-1)
            out = out.reshape(src_shape, self.d_chl[num]).contiguous()

            out = self.output_proj[num](out)
            tgt = query + self.dropout[num](out)
            tgt = self.norm[num](tgt)
            tgt = self.forward_ffn(tgt, num)

            fused_features = self.fusion_module[num](ms_features[lvl].features, tgt)

            # tgt = ms_features[lvl].features + tgt

            ms_features[lvl] = ms_features[lvl].replace_feature(fused_features)

            # ==========debug=================
            # --------------------------------

            # import open3d
            # samplingpoints = sampling_offsets[1, 1, :, :].view(-1, 3)
            # pointshow = crt_indice.cpu().numpy()
            # point_cloud = open3d.geometry.PointCloud()
            # point_cloud.points = open3d.utility.Vector3dVector(pointshow)
            # open3d.visualization.draw_geometries([point_cloud])

            # --------------------------------
            # ==========debug=================

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