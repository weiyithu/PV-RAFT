import torch
import torch.nn as nn

import numpy as np


class CorrBlock(nn.Module):
    def __init__(self, num_levels=3, base_scale=0.25, resolution=3, truncate_k=128, knn=32):
        super(CorrBlock, self).__init__()
        self.truncate_k = truncate_k
        self.num_levels = num_levels
        self.resolution = resolution  # local resolution
        self.base_scale = base_scale  # search (base_sclae * resolution)^3 cube
        self.out_conv = nn.Sequential(
            nn.Conv1d((self.resolution ** 3) * (self.num_levels + 1), 128, 1),
            nn.GroupNorm(8, 128),
            nn.PReLU(),
            nn.Conv1d(128, 64, 1)
        )
        self.voxel_conv = nn.Sequential(
            nn.Conv2d(4, 32, 1),
            nn.GroupNorm(8, 32),
            nn.PReLU(),
            nn.Conv2d(32, 1, 1)
        )
        
        self.knn = knn
        self.knn_conv = nn.Sequential(
            nn.Conv2d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.PReLU()
        )

        self.knn_out = nn.Conv1d(64, 64, 1)

    def init_module(self, fmap1, fmap2, xyz2, mode='voxel'):
        b, n_p, _ = xyz2.size()
        xyz2 = xyz2.view(b, 1, n_p, 3).expand(b, n_p, n_p, 3)

        corr = self.calculate_corr(fmap1, fmap2)

        corr_topk = torch.topk(corr.clone(), k=self.truncate_k, dim=2, sorted=True)
        if mode == 'voxel':
            self.truncated_corr = corr_topk.values
        elif mode == 'point':
            self.truncated_corr_point = corr_topk.values
        indx = corr_topk.indices.reshape(b, n_p, self.truncate_k, 1).expand(b, n_p, self.truncate_k, 3)
        if mode == 'voxel':
            self.ones_matrix = torch.ones_like(self.truncated_corr)
            self.truncate_xyz2 = torch.gather(xyz2, dim=2, index=indx)  # b, n_p1, k, 3
        elif mode == 'point':
            self.ones_matrix_point = torch.ones_like(self.truncated_corr_point)
            self.truncate_xyz2_point = torch.gather(xyz2, dim=2, index=indx)
            
    def __call__(self, coords):
        return self.get_voxel_feature(coords) + self.get_knn_feature(coords)

    def get_voxel_feature(self, coords):
        b, n_p, _ = coords.size()
        b, n1, n2, _ = self.truncate_xyz2.shape
        corr_feature = []
        base = self.get_base().to(coords.device)
        num = self.resolution ** 3
        from torch_scatter import scatter_add, scatter_max
        for i in range(self.num_levels):
            with torch.no_grad():
                r = self.base_scale * (2 ** i)
                dis_voxel = torch.round((self.truncate_xyz2 - coords.unsqueeze(dim=-2)) / r)
                valid_scatter = (torch.abs(dis_voxel) <= np.floor(self.resolution / 2)).all(dim=-1)
                dis_voxel = dis_voxel - (-1)
                cube_idx = dis_voxel[:, :, :, 0] * (self.resolution ** 2) +\
                    dis_voxel[:, :, :, 1] * self.resolution + dis_voxel[:, :, :, 2]
                cube_idx_scatter = cube_idx.type(torch.int64) * valid_scatter

                valid_scatter = valid_scatter.detach()
                cube_idx_scatter = cube_idx_scatter.detach()

            corr_add = scatter_add(self.truncated_corr * valid_scatter, cube_idx_scatter)
            corr_cnt = torch.clamp(scatter_add(self.ones_matrix * valid_scatter, cube_idx_scatter), 1, n_p)
            corr = corr_add / corr_cnt
            if corr.shape[-1] != self.resolution ** 3:
                repair = torch.zeros([b, n_p, self.resolution ** 3 - corr.shape[-1]], device=coords.device)
                corr = torch.cat([corr, repair], dim=-1)

            corr_feature.append(corr.transpose(1, 2).contiguous())

        with torch.no_grad():
            r = self.base_scale * (2 ** 3)
            R = 2 * r
            dis_voxel = torch.round((self.truncate_xyz2 - coords.unsqueeze(dim=-2)) / R)
            valid_center_candidate = (torch.abs(dis_voxel) <= 1.5).all(dim=-1)
            dis_voxel_intpos = dis_voxel - (-1)
            center_candidates_idx = dis_voxel_intpos[:, :, :, 0] * (self.resolution ** 2) +\
                dis_voxel_intpos[:, :, :, 1] * self.resolution + dis_voxel_intpos[:, :, :, 2]
            center_candidates_idx = (center_candidates_idx.type(torch.int64) + 1) * valid_center_candidate

            assert (center_candidates_idx < 0).nonzero().shape[0] == 0 and (center_candidates_idx >= 512).nonzero().shape[0] == 0, 'Out of scatter max bound!'
            center_corr, center_idx = scatter_max(self.truncated_corr * valid_center_candidate, center_candidates_idx)
            center_idx = center_idx[:, :, 1:]
            mask_oob = ((center_idx >= n2) | (center_idx < 0) | (center_corr[:, :, 1:] <= 0))
            center_idx[mask_oob] = 0
            center_coord = torch.gather(self.truncate_xyz2, 2, center_idx.unsqueeze(dim=-1).expand(-1, -1, -1, 3))

            idx_scatter = torch.ones((b, n1, n2), device=coords.device) * num
            center_coords_move = []
            for k in range(num):
                center_coord_k = center_coord[:, :, k:k+1, :]
                voxel_k_center = (coords + base[k].unsqueeze(dim=0).unsqueeze(dim=0) * R).unsqueeze(dim=2)
                if center_coord_k.shape[2] == 0:
                    center_coords_move.append(voxel_k_center)
                    continue
                center_coord_k_move = torch.clamp(center_coord_k - voxel_k_center, -(R-r) / 2, (R-r) / 2) + voxel_k_center
                center_coords_move.append(center_coord_k_move)
                dis_center = (self.truncate_xyz2 - center_coord_k_move) / r
                replace = (torch.abs(dis_center) <= 0.5).all(dim=-1) & (~mask_oob[:, :, k:k+1])
                idx_scatter[replace] = k

            idx_scatter = idx_scatter.type(torch.int64).detach()
            voxel_xyz = torch.cat(center_coords_move, dim=2)

        assert (idx_scatter < 0).nonzero().shape[0] == 0 and (idx_scatter >= 512).nonzero().shape[0] == 0, 'Out of scatter add bound!'
        corr_add = scatter_add(self.truncated_corr, idx_scatter)
        corr_cnt = scatter_add(self.ones_matrix, idx_scatter)
        voxel_corr = corr_add[:, :, :num] / torch.clamp(corr_cnt[:, :, :num], 1, n_p)
        voxel_feature = self.voxel_conv(torch.cat([voxel_corr.unsqueeze(dim=1), voxel_xyz.permute(0, 3, 1, 2).contiguous()], dim=1))
        corr = voxel_feature.squeeze(dim=1)
        corr_feature.append(corr.transpose(1, 2).contiguous())
        
        return self.out_conv(torch.cat(corr_feature, dim=1))

    def get_knn_feature(self, coords):
        b, n_p, _ = coords.size()

        dist = self.truncate_xyz2_point - coords.view(b, n_p, 1, 3)
        dist = torch.sum(dist ** 2, dim=-1)     # b, 8192, 512

        neighbors = torch.topk(-dist, k=self.knn, dim=2).indices

        b, n_p, _ = coords.size()
        knn_corr = torch.gather(self.truncated_corr_point.view(b * n_p, self.truncate_k), dim=1,
                                index=neighbors.reshape(b * n_p, self.knn)).reshape(b, 1, n_p, self.knn)

        neighbors = neighbors.view(b, n_p, self.knn, 1).expand(b, n_p, self.knn, 3)
        knn_xyz = torch.gather(self.truncate_xyz2_point, dim=2, index=neighbors).permute(0, 3, 1, 2).contiguous()
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