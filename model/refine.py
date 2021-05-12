import torch.nn as nn

from model.flot.gconv import SetConv


class FlotRefine(nn.Module):
    def __init__(self):
        super(FlotRefine, self).__init__()
        n = 32

        self.ref_conv1 = SetConv(3, n)
        self.ref_conv2 = SetConv(n, 2 * n)
        self.ref_conv3 = SetConv(2 * n, 4 * n)
        self.fc = nn.Linear(4 * n, 3)

    def forward(self, flow, graph):
        x = self.ref_conv1(flow, graph)
        x = self.ref_conv2(x, graph)
        x = self.ref_conv3(x, graph)
        x = self.fc(x)

        return flow + x
