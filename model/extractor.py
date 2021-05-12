import torch.nn as nn

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