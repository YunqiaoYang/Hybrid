# fusion method by disconet
# no kd loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

    # fusion method by disconet
# no kd loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

class PointwiseFusion(nn.Module):
    def __init__(self, args):
        super(PointwiseFusion, self).__init__()

        self.conv1_1 = nn.Conv2d(args['channel_size'] * 2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
    
    def generate_weight(self, x):
        # x: (N, 2C, H, W)
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = self.conv1_3(x_1)
        return x_1
    
    def forward(self, *kargs):
        if len(kargs) == 2:
            return self.forward_wowarp(kargs[0], kargs[1])
        else:
            return self.forward_wwarp(kargs[0], kargs[1], kargs[2])


    def forward_wowarp(self, curr_feature, hist_feature):
        _, C, _, _ = curr_feature.shape
        # x: (N, 2C, H, W)
        hist_curr_feature = torch.cat((hist_feature, curr_feature), dim=1)
        # (N, 1, H, W)
        temporal_weight = self.generate_weight(hist_curr_feature) 
        # (N, 1, H, W)
        temporal_weight = F.sigmoid(temporal_weight)
        # (N, C, H, W)
        temporal_weight = temporal_weight.expand(-1, C, -1, -1)
        # (N, C, H, W)
        fused_feature = temporal_weight * curr_feature + (1 - temporal_weight) * hist_feature
        return fused_feature

    def forward_wwarp(self, x, record_len, pairwise_t_matrix):
        ########## FUSION START ##########
        # we concat ego's feature with other agent
        # first transform feature to ego's coordinate
        split_x = regroup(x, record_len)

        B = pairwise_t_matrix.shape[0]
        _, C, H, W = x.shape

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        out = []

        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # update each node i
            i = 0 # ego
            # (N, C, H, W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(split_x[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))

            # (N, C, H, W)
            ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)
            # (N, 2C, H, W)
            neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
            # (N, 1, H, W)
            agent_weight = self.generate_weight(neighbor_feature_cat)
            # (N, 1, H, W)
            agent_weight = F.softmax(agent_weight, dim=0)

            agent_weight = agent_weight.expand(-1, C, -1, -1)
            # (N, C, H, W)
            feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
            out.append(feature_fused)

        return torch.stack(out)
