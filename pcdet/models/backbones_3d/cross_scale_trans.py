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

        # self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

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
            # \\get query and value
            query = self.query_proj[num](src)
            value = self.value_proj[num](src)

            sampling_offsets = self.sampling_offsets[num](query) \
                .view(-1, self.n_heads, self.n_points[num], 3)
            # test_2 =self.attention_weights[num](query)
            attention_weights = torch.softmax(self.attention_weights[num](query) \
                .view(-1, self.n_heads, self.n_points[num]), -1)

            # attention_weights = self.attention_weights[num](query) \
            #     .view(-1, self.n_heads, self.n_points[num])

            # test_1 = attention_weights[1,1,:]

            # \\crt_normalizer is uesd to norm cood to [0,1]
            fea_shape = ms_features[lvl].spatial_shape
            crt_normalizer = torch.tensor(fea_shape, device=value.device)
            value_epd_shape = torch.cat((torch.tensor(value.shape[1],
                                                      device=value.device).unsqueeze(0), crt_normalizer), dim=0)
            crt_indice = ms_features[lvl].indices

            # \\expand value to complete cube with zeros, for meeting grid_sample input
            crt_value_epd = torch.zeros(list(value_epd_shape), device=crt_indice.device)
            crt_indice = crt_indice[:, 1:4]
            crt_value_epd[:, crt_indice[:, 0].long(), crt_indice[:, 1].long(),\
                crt_indice[:, 2].long()] = value.transpose(0, 1)
            crt_value_epd_l = crt_value_epd.unsqueeze(0)


            # \\norm paras to [0,1]
            crt_offset = sampling_offsets / (crt_normalizer-1)
            voxel_coods = crt_indice / (crt_normalizer-1)
            sampling_locations = crt_offset + voxel_coods[:, None, None, :]
            swap_matrix = torch.tensor([[0.0, 0.0, 1.0],
                                        [0.0, 1.0, 0.0],
                                        [1.0, 0.0, 0.0]],device=value.device)
            sampling_locations_swap = torch.matmul(sampling_locations, swap_matrix)
            sampling_locations = sampling_locations_swap.unsqueeze(0)

            # \\norm grid to [-1,1]
            sampling_grids = 2 * sampling_locations - 1
            sampling_value_l_ = F.grid_sample(crt_value_epd_l, sampling_grids,
                                              mode='nearest', padding_mode='zeros', align_corners=True)

            out = (sampling_value_l_ * attention_weights[None, None, :, :, :]).sum(-1).transpose(1, 2).sum(-1)
            out = out.reshape(src_shape, self.d_chl[num]).contiguous()

            out = self.output_proj[num](out)
            tgt = query + self.dropout[num](out)
            tgt = self.norm[num](tgt)
            tgt = self.forward_ffn(tgt, num)


            # out = ms_features[lvl].features + tgt

            ms_features[lvl] = ms_features[lvl].replace_feature(tgt)

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
