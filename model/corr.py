import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import time


class CorrBlock(nn.Module):
    def __init__(self, num_levels=3, base_scale=0.25, resolution=3, truncate_k=128, knn=32):
        super(CorrBlock, self).__init__()
        self.truncate_k = truncate_k
        self.num_levels = num_levels
        self.resolution = resolution  # local resolution
        self.num_base = 7
        self.base_scale = base_scale  # search (base_sclae * resolution)^3 cube
        self.offset_conv = nn.Sequential(
            nn.Conv1d(self.truncate_k * 4, 512, 1),
            nn.GroupNorm(8, 512),
            nn.PReLU(),
            nn.Conv1d(512, 128, 1),
            nn.GroupNorm(8, 128),
            nn.PReLU(),
            nn.Conv1d(128, self.num_base * 3, 1),
            nn.Sigmoid()
        )

        self.voxel_conv = nn.Sequential(
            nn.Conv2d(4, 32, 1),
            nn.GroupNorm(8, 32),
            nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.voxel_out = nn.Sequential(
            nn.Conv1d(self.num_base * self.num_levels, 128, 1),
            nn.GroupNorm(8, 128),
            nn.PReLU(),
            nn.Conv1d(128, 64, 1)
        )

        self.knn = knn
        self.knn_conv = nn.Sequential(
            nn.Conv2d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.PReLU()
        )

        self.knn_out = nn.Conv1d(64, 64, 1)

    def init_module(self, fmap1, fmap2, xyz2):
        b, n_p, _ = xyz2.size()
        xyz2 = xyz2.view(b, 1, n_p, 3).expand(b, n_p, n_p, 3)

        corr = self.calculate_corr(fmap1, fmap2)

        corr_topk = torch.topk(corr.clone(), k=self.truncate_k, dim=2, sorted=True)
        self.truncated_corr = corr_topk.values
        indx = corr_topk.indices.reshape(b, n_p, self.truncate_k, 1).expand(b, n_p, self.truncate_k, 3)
        self.ones_matrix = torch.ones_like(self.truncated_corr)
        # num = self.resolution ** 3
        # b, n1, n2 = self.truncated_corr.shape
        # self.ones_matrix = torch.ones_like(self.truncated_corr).unsqueeze(dim=0).expand((num, b, n1, n2)).view(num*b, n1, n2)
        
        self.truncate_xyz2 = torch.gather(xyz2, dim=2, index=indx)  # b, n_p1, k, 3

    def __call__(self, coords):
        return self.get_voxel_feature(coords) + self.get_knn_feature(coords)

    def get_voxel_feature(self, coords):
        b, n_p, _ = coords.size()
        corr_feature = []
        base = self.get_base7()
        b, n1, n2, _ = self.truncate_xyz2.shape
        num = self.num_base

        offset_input = torch.cat([self.truncate_xyz2 - coords.unsqueeze(dim=-2), 
                                  self.truncated_corr.unsqueeze(dim=-1)], dim=-1)
        offset = self.offset_conv(offset_input.view(b, n1, -1).transpose(1, 2).contiguous())
        offset = offset.view(b, n1, 3, num).contiguous()

        from torch_scatter import scatter_add
        for i in range(self.num_levels):
            with torch.no_grad():
                r = self.base_scale * (2 ** i)
                offset_r = (offset - 0.5) * r

                # start = time.time()
                # dis_voxel = torch.round((self.truncate_xyz2 - coords.unsqueeze(dim=-2)) / r)            # B, 8192, 512, 3
                # valid_scatter = (torch.abs(dis_voxel) <= np.floor(self.resolution / 2)).all(dim=-1)     # B, 8192, 512
                # dis_voxel = dis_voxel - (-1)
                # cube_idx = dis_voxel[:, :, :, 0] * (self.resolution ** 2) +\
                #     dis_voxel[:, :, :, 1] * self.resolution + dis_voxel[:, :, :, 2]                     # B, 8192, 512
                # cube_idx_scatter = cube_idx.type(torch.int64) * valid_scatter                           # B, 8192, 512
                # valid_scatter = valid_scatter.detach()
                # cube_idx_scatter = cube_idx_scatter.detach()
                # time1 = time.time() - start
                # start = time.time()

                # valid_scatter = []
                # dis_voxel = torch.round((self.truncate_xyz2 - coords.unsqueeze(dim=-2)) / r)
                # for k in range(num):
                #     valid_scatter.append((dis_voxel == base[k].to(dis_voxel.device)).all(dim=-1))   # B, 8192, 512
                # valid_scatter = torch.stack(valid_scatter).view(num*b, n1, n2).type(torch.int64).detach()

                # corr = []
                # with torch.no_grad():
                #     dis_voxel = torch.round((self.truncate_xyz2 - coords.unsqueeze(dim=-2)) / r)
                # for k in range(num):
                #     with torch.no_grad():
                #         valid_scatter = (dis_voxel == base[k].to(dis_voxel.device)).all(dim=-1).type(torch.int64).detach()
                #     corr_add = scatter_add(self.truncated_corr, valid_scatter)
                #     corr_cnt = torch.clamp(scatter_add(self.ones_matrix, valid_scatter), 1, n_p)
                #     if corr_add.shape[-1] == 2 and corr_cnt.shape[-1] == 2:
                #         corr_avg = corr_add[:, :, 1] / corr_cnt[:, :, 1]
                #     else:
                #         corr_avg = torch.zeros([b, n_p], device=coords.device)
                #     corr.append(corr_avg.unsqueeze(dim=-1))
                # corr = torch.cat(corr, dim=-1)
                # corr_feature.append(corr.transpose(1, 2).contiguous())

                # idx_scatter = torch.zeros((b, n1, n2), device=coords.device)
                # cube_idx_scatter = torch.zeros((b, n1, n2), device=coords.device)
                # start = time.time()
                # for k in range(num):
                #     dis_voxel = torch.round((self.truncate_xyz2 - (coords + offset[:, :, :, k]).unsqueeze(dim=-2)) / r)
                #     valid_scatter = (dis_voxel == base[k].to(dis_voxel.device)).all(dim=-1)   # B, 8192, 512
                #     idx_scatter[valid_scatter] += (2 ** k)
                # time1 = time.time() - start
                # start = time.time()

                # idx_unique = torch.unique(idx_scatter)
                # root_list = []
                # for idx in idx_unique:
                #     root = self.get_root(idx)
                #     if np.sum(root) == 0:
                #         cube_idx_scatter[idx_scatter == idx] = 0
                #     elif np.sum(root) == 1:
                #         cube_idx_scatter[idx_scatter == idx] = int(np.where(root==1)[0])
                #     else:
                #         cube_idx_scatter[idx_scatter == idx] = len(root_list) + num
                #         root_list.append(root)
                # cube_idx_scatter = cube_idx_scatter.type(torch.int64).detach()
                # time2 = time.time() - start
                # start = time.time()

                idx_scatter = torch.zeros((b, n1, n2), device=coords.device)
                idx_scatter_bi = torch.zeros((b, n1, n2, num), device=coords.device)
                cube_idx_scatter = torch.zeros((b, n1, n2), device=coords.device)
                start = time.time()

                for k in range(num):
                    dis_voxel = torch.round((self.truncate_xyz2 - (coords + offset_r[:, :, :, k]).unsqueeze(dim=-2)) / r)
                    valid_scatter = (dis_voxel == base[k].to(dis_voxel.device)).all(dim=-1)   # B, 8192, 512
                    idx_scatter[valid_scatter] += (2 ** k)
                    idx_scatter_bi[valid_scatter] += torch.zeros_like(idx_scatter_bi[valid_scatter]).index_fill_(1, torch.tensor(k).cuda(), 1)

                idx_unique = torch.unique(idx_scatter)
                root_list = []
                for idx in idx_unique:
                    idx_select = (idx_scatter == idx)
                    root = idx_scatter_bi[idx_select][0]
                    # root = self.get_root(idx)
                    if torch.sum(root) == 0:
                        cube_idx_scatter[idx_select] = 0
                    elif torch.sum(root) == 1:
                        cube_idx_scatter[idx_select] = int(torch.where(root==1)[0])
                    else:
                        cube_idx_scatter[idx_select] = len(root_list) + num
                        root_list.append(root)
                cube_idx_scatter = cube_idx_scatter.type(torch.int64).detach()

            # corr_add = scatter_add(self.truncated_corr * valid_scatter, cube_idx_scatter)               # B, 8192, 27
            # corr_cnt = torch.clamp(scatter_add(self.ones_matrix * valid_scatter, cube_idx_scatter), 1, n_p)
            # time2 = time.time() - start
            # corr = corr_add / corr_cnt
            # if corr.shape[-1] != self.resolution ** 3:
            #     repair = torch.zeros([b, n_p, self.resolution ** 3 - corr.shape[-1]], device=coords.device)
            #     corr = torch.cat([corr, repair], dim=-1)

            # corr_add = scatter_add(self.truncated_corr.unsqueeze(dim=0).expand((num, b, n1, n2)).view(num*b, n1, n2), 
            #                        valid_scatter)
            # corr_add = corr_add[:, :, 1].view(num, b, n1).permute(1, 2, 0).contiguous()
            # corr_cnt = torch.clamp(scatter_add(self.ones_matrix, valid_scatter), 1, n_p)[:, :, 1].view(num, b, n1).permute(1, 2, 0).contiguous()
            # corr = corr_add / corr_cnt

            corr_add = scatter_add(self.truncated_corr, cube_idx_scatter)
            corr_cnt = scatter_add(self.ones_matrix, cube_idx_scatter)
            for cnt, root in enumerate(root_list):
                for bit in torch.where(root == 1)[0]:
                    corr_add[:, :, bit] = corr_add[:, :, bit] + corr_add[:, :, cnt + num]
                    corr_cnt[:, :, bit] = corr_cnt[:, :, bit] + corr_cnt[:, :, cnt + num]
            voxel_corr = corr_add[:, :, :num] / torch.clamp(corr_cnt[:, :, :num], 1, n_p)
            if voxel_corr.shape[-1] != self.num_base:
                repair = torch.zeros([b, n_p, self.num_base - voxel_corr.shape[-1]], device=coords.device)
                voxel_corr = torch.cat([voxel_corr, repair], dim=-1)
            voxel_xyz = (base.cuda() * r).unsqueeze(dim=0).unsqueeze(dim=1).expand(b, n1, num, 3) + offset_r.transpose(2, 3).contiguous()
            voxel_feature = self.voxel_conv(torch.cat([voxel_corr.unsqueeze(dim=1), voxel_xyz.permute(0, 3, 1, 2).contiguous()], dim=1))
            corr = voxel_feature.squeeze(dim=1)
            corr_feature.append(corr.transpose(1, 2).contiguous())

        return self.voxel_out(torch.cat(corr_feature, dim=1))

    def get_knn_feature(self, coords):
        b, n_p, _ = coords.size()

        dist = self.truncate_xyz2 - coords.view(b, n_p, 1, 3)
        dist = torch.sum(dist ** 2, dim=-1)     # b, 8192, 512

        neighbors = torch.topk(-dist, k=self.knn, dim=2).indices

        b, n_p, _ = coords.size()
        knn_corr = torch.gather(self.truncated_corr.view(b * n_p, self.truncate_k), dim=1,
                                index=neighbors.reshape(b * n_p, self.knn)).reshape(b, 1, n_p, self.knn)

        neighbors = neighbors.view(b, n_p, self.knn, 1).expand(b, n_p, self.knn, 3)
        knn_xyz = torch.gather(self.truncate_xyz2, dim=2, index=neighbors).permute(0, 3, 1, 2).contiguous()
        knn_xyz = knn_xyz - coords.transpose(1, 2).reshape(b, 3, n_p, 1)

        knn_feature = self.knn_conv(torch.cat([knn_corr, knn_xyz], dim=1))
        knn_feature = torch.max(knn_feature, dim=3)[0]
        return self.knn_out(knn_feature)

    @staticmethod
    def calculate_corr(fmap1, fmap2):
        batch, dim, num_points = fmap1.shape
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr / torch.sqrt(torch.tensor(dim).float())
        return corr

    def get_base(self):
        num = self.resolution ** 3
        base = torch.zeros((num, 3))
        for i in range(num):
            quotient = i
            for j in range(3):
                base[i][2-j] = quotient % self.resolution - 1
                quotient = quotient // self.resolution
        return base

    def get_base7(self):
        base = torch.tensor([[0, 0, 0],
                             [-1, 0, 0],
                             [1, 0, 0],
                             [0, -1, 0],
                             [0, 1, 0],
                             [0, 0, -1],
                             [0, 0, 1]])
        return base

    def get_root(self, idx):
        # num = self.resolution ** 3
        num = 7
        quotient = idx
        root = np.zeros((num,))
        for i in range(num):
            if quotient == 0:
                break
            root[i] = quotient % 2
            quotient = quotient // 2
        return root
