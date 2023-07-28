import torch.nn as nn
import torch.nn.functional as F

from model.flot.gconv import SetConv
from model.flot.graph import Graph


class FlotEncoder(nn.Module):
    def __init__(self, num_neighbors=32):
        super(FlotEncoder, self).__init__()
        n = 32
        self.num_neighbors = num_neighbors

        self.feat_conv1 = SetConv(3, n)
        self.feat_conv2 = SetConv(n, 2 * n)
        self.feat_conv3 = SetConv(2 * n, 4 * n)

    def forward(self, pc):
        graph = Graph.construct_graph(pc, self.num_neighbors)
        x = self.feat_conv1(pc, graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)
        x = x.transpose(1, 2).contiguous()

        return x, graph



class FlotTiny(nn.Module):
    def __init__(self, num_neighbors=32):
        super(FlotTiny, self).__init__()
        n = 32
        self.num_neighbors = num_neighbors

        self.feat_conv1 = SetConv(3, n)
        self.feat_conv2 = SetConv(n, n * 2)
        self.feat_conv3 = SetConv(n * 2, n * 4)
        self.mlp_conv = MLPConv(input_dim=n * 4, hidden_dim=[256], output_dim=512)

    def forward(self, pc):
        graph = Graph.construct_graph(pc, self.num_neighbors)
        x = self.feat_conv1(pc, graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)
        x = self.mlp_conv(x.transpose(1, 2).contiguous())

        return x


class MLPConv(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.num_layers = len(hidden_dim) + 1
        h = hidden_dim
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, 1) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.norms = nn.ModuleList(
            norm_layer(k) for k in (h + [output_dim])
        )

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = F.relu(norm(layer(x))) if i < self.num_layers - 1 else layer(x)
        return x