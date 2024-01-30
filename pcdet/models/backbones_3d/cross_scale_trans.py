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
        self.input_proj = []
        self.sampling_offsets = []
        self.attention_weights = []
        self.query_proj = []
        self.value_proj = []

        for i in range(self.n_levels):
            self.input_proj.append(nn.Sequential(
                nn.Linear(self.d_chl[i], self.d_model),
                nn.GroupNorm(self.d_chl[i], self.d_model)).to("cuda:0"))
            self.sampling_offsets.append(
                nn.Linear(self.d_chl[i], self.n_heads * self.n_points[i] * 3).to("cuda:0")
            )
            self.attention_weights.append(
                nn.Linear(self.d_chl[i], self.n_heads * self.n_points[i]).to("cuda:0")
            )
            self.query_proj.append(
                nn.Linear(self.d_model, self.d_chl[i]).to("cuda:0")
            )
            self.value_proj.append(
                nn.Linear(self.d_model, self.d_chl[i]).to("cuda:0")
            )


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
        # get input vector with same dim
        ms_features = batch_dict['multi_scale_3d_features']

        # expand value to complete cube with zeros, for meeting grid_sample input
        indice_list = []
        spatial_shapes_list = []
        expd_fea_list = []
        start_time = time.time()
        for num, lvl in enumerate(ms_features):

            # get input vector with same dim
            src = self.input_proj[num](ms_features[lvl].features)
            src_shape = src.shape[0]
            # get query and value
            query = self.query_proj[num](src)
            value = self.value_proj[num](src)
            sampling_offsets = self.sampling_offsets[num](query)\
                .view(-1, self.n_heads, self.n_points[num], 3)
            attention_weights = torch.softmax(self.attention_weights[num](query), -1)\
                .view(-1, self.n_heads, self.n_points[num])

            # crt_normalizer is uesd to norm cood to [0,1]
            fea_shape = ms_features[lvl].spatial_shape
            crt_normalizer = torch.tensor(fea_shape, device=value.device)
            value_epd_shape = torch.cat((torch.tensor(value.shape[1], device=value.device).unsqueeze(0), crt_normalizer), dim=-0)
            crt_indice = ms_features[lvl].indices
            crt_value_epd = torch.zeros(list(value_epd_shape), device=crt_indice.device)
            crt_indice = crt_indice[:, 1:4]
            crt_cood = list(crt_indice.long())
            for j, idx in enumerate(crt_cood):
                crt_value_epd[:, idx[0], idx[1], idx[2]] = value[j, :]
            crt_value_epd_l = crt_value_epd.unsqueeze(0)

            # norm paras to [0,1]
            crt_offset = sampling_offsets / crt_normalizer
            voxel_coods = crt_indice /crt_normalizer
            sampling_locations = crt_offset + voxel_coods[:,None,None,:]
            sampling_locations = sampling_locations.unsqueeze(0)
            # norm grid to [-1,1]
            sampling_grids = 2 * sampling_locations - 1


            sampling_value_l_ = F.grid_sample(crt_value_epd_l, sampling_grids,
                                              mode='nearest', padding_mode='zeros', align_corners=False)


            out = (sampling_value_l_ * attention_weights[None,None,:,:,:]).sum(-1).transpose(1, 2).sum(-1)

            out = out.reshape(src_shape, self.d_chl[num]).contiguous()



            ms_features[lvl] = ms_features[lvl].replace_feature(out)

        end_time = time.time()
        run_time_1 = end_time - start_time
        print(run_time_1)

        # import open3d
        # crt_levl = ms_features[lvl].feature
        # pointshow = crt_levl.indices[:, 1:4].cpu().numpy()
        # point_cloud = open3d.geometry.PointCloud()
        # point_cloud.points = open3d.utility.Vector3dVector(pointshow)
        # open3d.visualization.draw_geometries([point_cloud])

        return batch_dict
