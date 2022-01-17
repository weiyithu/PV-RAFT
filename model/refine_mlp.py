import torch.nn as nn
import torch.nn.functional as F


class FlotRefine(nn.Module):
    def __init__(self):
        super(FlotRefine, self).__init__()
        n = 32

        self.ref_conv1 = MLPConv(3, n, n, 3)
        self.ref_conv2 = MLPConv(n, 2 * n, 2 * n, 3)
        self.ref_conv3 = MLPConv(2 * n, 4 * n, 4 * n, 3)
        self.fc = nn.Conv1d(4 * n, 3, 1)

    def forward(self, flow):
        x = self.ref_conv1(flow.transpose(1, 2).contiguous())
        x = self.ref_conv2(x)
        x = self.ref_conv3(x)
        x = self.fc(x)

        return flow + x.transpose(1, 2).contiguous()


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