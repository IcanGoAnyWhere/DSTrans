from functools import partial

import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv


class cross_scale_trans(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.d_model = 64
        # self.n_levels = n_levels
        # self.n_heads = n_heads
        # self.n_points = n_points

        self.query_proj_1 = nn.Linear(16, self.d_model)
        self.query_proj_2 = nn.Linear(32, self.d_model)
        self.query_proj_3 = nn.Linear(32, self.d_model)
        self.query_proj_4 = nn.Linear(64, self.d_model)

        self.query_proj_1 = nn.Linear(16, self.d_model)
        self.query_proj_2 = nn.Linear(32, self.d_model)
        self.query_proj_3 = nn.Linear(32, self.d_model)
        self.query_proj_4 = nn.Linear(64, self.d_model)


    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, batch_dict):


        ms_features = batch_dict['multi_scale_3d_features']
        query_1 = self.query_proj_1(ms_features['x_conv1'].features)
        query_2 = self.query_proj_2(ms_features['x_conv2'].features)
        query_3 = self.query_proj_3(ms_features['x_conv3'].features)
        query_4 = self.query_proj_4(ms_features['x_conv4'].features)

        ms_query = [query_1,query_2,query_3,query_4]

        for levl_1 in ms_features:
            for levl_2 in ms_features:
                if levl_1 is not levl_2:
                    crt_levl = ms_features[levl_1]

                    # import open3d
                    # pointshow = crt_levl.indices[:, 1:4].cpu().numpy()
                    # point_cloud = open3d.geometry.PointCloud()
                    # point_cloud.points = open3d.utility.Vector3dVector(pointshow)
                    # open3d.visualization.draw_geometries([point_cloud])



                    pass



        return batch_dict
