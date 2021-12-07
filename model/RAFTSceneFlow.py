import torch
import torch.nn as nn

from MinkowskiEngine import SparseTensor
from model.extractor import FlotEncoder
from model.minkowski.res16unet import Res16UNet18
from model.corr import CorrBlock
from model.update import UpdateBlock
from model.refine import FlotRefine


class RSF(nn.Module):
    def __init__(self, args):
        super(RSF, self).__init__()
        self.hidden_dim = 64
        self.context_dim = 64
        self.feature_extractor = Res16UNet18(in_channels=3, out_channels=512, config=args)
        self.context_extractor = FlotEncoder()
        self.corr_block = CorrBlock(num_levels=args.corr_levels, base_scale=args.base_scales,
                                    resolution=3, truncate_k=args.truncate_k)
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)
        # self.refine_block = FlotRefine()

    def forward(self, p, num_iters=12):
        # feature extraction
        [xyz1, xyz2] = p['sequence']
        sinput_pc1 = SparseTensor(p['sparse'][0].type(torch.float32), torch.cat([torch.zeros((p['sparse'][0].shape[0], 1), device=xyz1.device), p['sparse'][0]], dim=1), device=xyz1.device)
        sinput_pc2 = SparseTensor(p['sparse'][1].type(torch.float32), torch.cat([torch.zeros((p['sparse'][1].shape[0], 1), device=xyz1.device), p['sparse'][1]], dim=1), device=xyz2.device)
        fmap1_sparse = self.feature_extractor(sinput_pc1)
        fmap2_sparse = self.feature_extractor(sinput_pc2)

        fmap1 = fmap1_sparse.F[p['idx_inverse'][0], :].transpose(0, 1).contiguous().unsqueeze(dim=0)
        fmap2 = fmap2_sparse.F[p['idx_inverse'][1], :].transpose(0, 1).contiguous().unsqueeze(dim=0)

        # correlation matrix
        self.corr_block.init_module(fmap1, fmap2, xyz2)

        fct1, graph_context = self.context_extractor(p['sequence'][0])

        net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords1, coords2 = xyz1, xyz1
        flow_predictions = []

        for itr in range(num_iters):
            coords2 = coords2.detach()
            corr = self.corr_block(coords=coords2)
            flow = coords2 - coords1
            net, delta_flow = self.update_block(net, inp, corr, flow, graph_context)
            coords2 = coords2 + delta_flow
            flow_predictions.append(coords2 - coords1)
        # refined_flow = self.refine_block(coords2 - coords1, graph)
        # flow_predictions.append(refined_flow)

        return flow_predictions

