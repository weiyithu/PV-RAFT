import torch
import torch.nn as nn
import numpy as np

from MinkowskiEngine import SparseTensor
from model.extractor import FlotEncoder, FlotTiny
from model.minkowski.res16unet import Res16UNet34C
from model.corr import CorrBlock
from model.update import UpdateBlock
from model.refine import FlotRefine


class RSF_refine(nn.Module):
    def __init__(self, args):
        super(RSF_refine, self).__init__()
        self.hidden_dim = 64
        self.context_dim = 64
        self.feature_extractor = Res16UNet34C(in_channels=3, out_channels=512, config=args)
        self.feature_mlp = FlotTiny()
        self.context_extractor = FlotEncoder()
        self.corr_block = CorrBlock(num_levels=args.corr_levels, base_scale=args.base_scales,
                                    resolution=3, truncate_k=args.truncate_k)
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)
        self.refine_block = FlotRefine()

    def forward(self, p, num_iters=12, drop_thresh=1.0):
        with torch.no_grad():
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
            mask = torch.ones((coords2.shape[0], coords2.shape[1]), device=coords2.device, dtype=torch.bool)

            for itr in range(num_iters):
                coords2 = coords2.detach()
                corr = self.corr_block(coords=coords2)
                flow = coords2 - coords1
                net, delta_flow, drop_conf = self.update_block(net, inp, corr, flow, graph_context)
                mask[(drop_conf > drop_thresh)] = False
                if len(mask.nonzero()) == 0:
                    break
                coords2[mask] = coords2[mask] + delta_flow[mask]
        refined_flow = self.refine_block(coords2 - coords1, graph_context)

        return refined_flow
