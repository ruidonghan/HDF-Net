import torch
from torch import nn
from torch.nn import functional as F


class SRMConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(SRMConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.SRMWeights = nn.Parameter(
            self._get_srm_list(), requires_grad=False)

    def _get_srm_list(self):
        # srm kernel 1
        srm1 = [[0,  0, 0,  0, 0],
                [0, -1, 2, -1, 0],
                [0,  2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0,  0, 0,  0, 0]]
        srm1 = torch.tensor(srm1, dtype=torch.float32) / 4.

        # srm kernel 2
        srm2 = [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
        srm2 = torch.tensor(srm2, dtype=torch.float32) / 12.

        # srm kernel 3
        srm3 = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
        srm3 = torch.tensor(srm3, dtype=torch.float32) / 2.

        return torch.stack([torch.stack([srm1, srm1, srm1], dim=0), torch.stack([srm2, srm2, srm2], dim=0), torch.stack([srm3, srm3, srm3], dim=0)], dim=0)

    def forward(self, X):
        # X1 =
        return F.conv2d(X, self.SRMWeights, stride=self.stride, padding=self.padding)