import torch
import torch.nn as nn
import numpy as np

from MinkowskiEngine import SparseTensor
from model.extractor import FlotEncoder, FlotTiny
from model.minkowski.res16unet import Res16UNet34C
from model.corr import CorrBlock
from model.update import UpdateBlock


class DPV_RAFT(nn.Module):
    def __init__(self, args):
        super(DPV_RAFT, self).__init__()
        self.hidden_dim = 64
        self.context_dim = 64
        self.gamma = 0.8
        self.feature_extractor = Res16UNet34C(in_channels=3, out_channels=512, config=args)
        self.feature_mlp = FlotTiny()
        self.context_extractor = FlotEncoder()
        self.corr_block = CorrBlock(num_levels=args.corr_levels, base_scale=args.base_scales,
                                    resolution=3, truncate_k=args.truncate_k)
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)

    def forward(self, p, gt=None, num_iters=12, drop_thresh=0.8):
        # feature extraction
        [xyz1, xyz2] = p['sequence']
        sinput_pc1 = SparseTensor(p['sparse'][0][:, 1:].type(torch.float32), p['sparse'][0].type(torch.int), device=xyz1.device)
        sinput_pc2 = SparseTensor(p['sparse'][1][:, 1:].type(torch.float32), p['sparse'][1].type(torch.int), device=xyz2.device)
        fmap1_sparse = self.feature_extractor(sinput_pc1)
        fmap2_sparse = self.feature_extractor(sinput_pc2)

        fmap1 = []
        fmap2 = []
        for b in sinput_pc1.C[:, 0].unique():
            fmap1.append(fmap1_sparse.F[fmap1_sparse.C[:, 0] == b][p['idx_inverse'][0][b], :].transpose(0, 1).contiguous().unsqueeze(dim=0))
            fmap2.append(fmap2_sparse.F[fmap2_sparse.C[:, 0] == b][p['idx_inverse'][1][b], :].transpose(0, 1).contiguous().unsqueeze(dim=0))
        fmap1 = torch.cat(fmap1, dim=0)
        fmap2 = torch.cat(fmap2, dim=0)

        fmap1_point = fmap1 + self.feature_mlp(xyz1)
        fmap2_point = fmap2 + self.feature_mlp(xyz2)

        # correlation matrix
        self.corr_block.init_module(fmap1, fmap2, xyz2, mode='voxel')
        self.corr_block.init_module(fmap1_point, fmap2_point, xyz2, mode='point')

        fct1, graph_context = self.context_extractor(p['sequence'][0])

        net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords1, coords2 = xyz1.clone(), xyz1.clone()
        loss_flow_sum = 0
        loss_drop_sum = 0
        loss_flow = None
        mask = torch.ones((coords2.shape[0], coords2.shape[1]), device=coords2.device, dtype=torch.bool)

        if gt is not None:
            gt_mask = gt[0][..., 0]
            gt_flow = gt[1]
            for itr in range(num_iters):
                coords2 = coords2.detach()
                corr = self.corr_block(coords=coords2)
                flow = coords2 - coords1

                net, delta_flow, drop_conf = self.update_block(net, inp, corr, flow, graph_context)
                if loss_flow is None:
                    est_flow = coords2 + delta_flow - coords1
                    loss_flow_all = self.cal_loss_flow(est_flow, gt_flow, gt_mask)
                    drop_conf_gt = self.get_drop_conf_gt(loss_flow_all)
                loss_drop = torch.abs(drop_conf - drop_conf_gt.detach()).mean()         # loss_drop of dropped points are also included!
                loss_drop_sum += loss_drop

                mask = mask.clone()
                mask[(drop_conf_gt > drop_thresh)] = False
                if len(mask.nonzero()) == 0:
                    break
                coords2[mask] = coords2[mask] + delta_flow[mask]
                est_flow = coords2 - coords1
                loss_flow_all = self.cal_loss_flow(est_flow, gt_flow, gt_mask)
                drop_conf_gt = self.get_drop_conf_gt(loss_flow_all)
                loss_flow = loss_flow_all[mask.squeeze(dim=0)]
                loss_flow_sum += torch.mean(loss_flow) * (self.gamma ** (num_iters - itr - 1))
            flow_pred_final = coords2 - coords1
            return flow_pred_final, loss_flow_sum, loss_drop_sum
        else:
            for itr in range(num_iters):
                coords2 = coords2.detach()
                corr = self.corr_block(coords=coords2)
                flow = coords2 - coords1
                net, delta_flow, drop_conf = self.update_block(net, inp, corr, flow, graph_context)

                mask[(drop_conf > drop_thresh)] = False
                if len(mask.nonzero()) == 0:
                    break
                coords2[mask] = coords2[mask] + delta_flow[mask]
            flow_pred_final = coords2 - coords1
            return flow_pred_final

    def cal_loss_flow(self, est_flow, gt_flow, gt_mask):
        error = est_flow - gt_flow
        error = torch.abs(error[gt_mask > 0])

        return error

    def get_drop_conf_gt(self, loss_flow):
        error = torch.norm(loss_flow, dim=-1, p=2).unsqueeze(dim=0)
        error_max = 0.1
        error = (error / error_max).clamp(0, 1)
        drop_conf_gt = 1 - error

        return drop_conf_gt

    def compute_epe_mask(self, est_flow, sf_gt, mask):
        mask = mask.cpu().numpy()[..., 0]
        sf_gt = sf_gt.cpu().numpy()[mask > 0]
        sf_pred = est_flow.cpu().numpy()[mask > 0]

        #
        l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
        EPE3D = l2_norm.mean()

        #
        sf_norm = np.linalg.norm(sf_gt, axis=-1)
        relative_err = l2_norm / (sf_norm + 1e-4)
        acc3d_strict = (
            (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
        )
        acc3d_relax = (
            (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
        )
        outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()

        return EPE3D, acc3d_strict, acc3d_relax, outlier

