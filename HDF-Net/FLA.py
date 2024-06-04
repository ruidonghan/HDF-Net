import torch
import torch.nn as nn
import torch.nn.functional as F

class FLA(nn.Module):
    def __init__(self, plane):
        super(FLA, self).__init__()
        self.conv1 = nn.Linear(plane, plane)
        self.conv2 = nn.Linear(plane, plane)
        self.conv = nn.Sequential(nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(plane),
                                  nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        feat_h = x.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)
        feat_w = x.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)
        encode_h = self.conv1(F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())
        encode_w = self.conv2(F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())

        energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))
        energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))
        full_relation_h = self.softmax(energy_h)  # [b*w, c, c]
        full_relation_w = self.softmax(energy_w)

        full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3, 1)
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)
        out = self.gamma * (full_aug_h + full_aug_w) + x
        out = self.conv(out)
        return out