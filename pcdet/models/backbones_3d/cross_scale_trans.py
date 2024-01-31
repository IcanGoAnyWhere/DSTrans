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

        for i in range(self.n_levels):
            self.input_proj.append(nn.Sequential(
                nn.Linear(self.d_chl[i], self.d_model),
                nn.GroupNorm(self.d_chl[i], self.d_model)
            ))

            self.sampling_offsets.append(nn.Linear(self.d_chl[i], self.n_heads * self.n_points[i] * 3))
            self.attention_weights.append(nn.Linear(self.d_chl[i], self.n_heads * self.n_points[i]))
            self.query_proj.append(nn.Linear(self.d_model, self.d_chl[i]))
            self.value_proj.append(nn.Linear(self.d_model, self.d_chl[i]))

        # self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # grid_init = torch.stack([thetas.cos(), thetas.sin(), thetas.sin()], -1)
        # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 3).repeat(1, np.sum(self.n_points), 1)
        # for i in range(np.sum(self.n_points)):
        #     grid_init[:, i, :] *= i + 1
        # with torch.no_grad():
        # self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))


    def forward(self, batch_dict):

        ms_features = batch_dict['multi_scale_3d_features']
        for num, lvl in enumerate(ms_features):
            # \\get input vector with same dim
            features = ms_features[lvl].features
            src = self.input_proj[num](features)

            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            #
            # print(prof)

            src_shape = src.shape[0]
            # \\get query and value
            query = self.query_proj[num](src)
            value = self.value_proj[num](src)


            sampling_offsets = self.sampling_offsets[num](query)\
                .view(-1, self.n_heads, self.n_points[num], 3)
            attention_weights = torch.softmax(self.attention_weights[num](query), -1)\
                .view(-1, self.n_heads, self.n_points[num])

            # \\crt_normalizer is uesd to norm cood to [0,1]
            fea_shape = ms_features[lvl].spatial_shape
            crt_normalizer = torch.tensor(fea_shape, device=value.device)
            value_epd_shape = torch.cat((torch.tensor(value.shape[1],
                                                      device=value.device).unsqueeze(0), crt_normalizer), dim=0)
            crt_indice = ms_features[lvl].indices

            # \\expand value to complete cube with zeros, for meeting grid_sample input
            crt_value_epd = torch.zeros(list(value_epd_shape), device=crt_indice.device)
            crt_indice = crt_indice[:, 1:4]
            crt_value_epd[:, crt_indice[:,0].long(), crt_indice[:,1].long(), crt_indice[:,2].long()] = value.transpose(0,1)
            crt_value_epd_l = crt_value_epd.unsqueeze(0)

            # \\norm paras to [0,1]
            crt_offset = sampling_offsets / crt_normalizer
            voxel_coods = crt_indice /crt_normalizer
            sampling_locations = crt_offset + voxel_coods[:,None,None,:]
            sampling_locations = sampling_locations.unsqueeze(0)
            # \\norm grid to [-1,1]
            sampling_grids = 2 * sampling_locations - 1
            sampling_value_l_ = F.grid_sample(crt_value_epd_l, sampling_grids,
                                              mode='bilinear', padding_mode='zeros', align_corners=False)

            out = (sampling_value_l_ * attention_weights[None,None,:,:,:]).sum(-1).transpose(1, 2).sum(-1)

            out = out.reshape(src_shape, self.d_chl[num]).contiguous()
            out = ms_features[lvl].features + out


            ms_features[lvl] = ms_features[lvl].replace_feature(out)

        #==========debug=================
        #--------------------------------

        # import open3d
        # pointshow = crt_levl.indices[:, 1:4].cpu().numpy()
        # point_cloud = open3d.geometry.PointCloud()
        # point_cloud.points = open3d.utility.Vector3dVector(pointshow)
        # open3d.visualization.draw_geometries([point_cloud])

        # --------------------------------
        # ==========debug=================


        return batch_dict
