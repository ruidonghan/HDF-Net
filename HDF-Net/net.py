import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import SRM
from ASPP import ASPP
from FLA import FLA
from ViT import ViT


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
  def __init__(self, kernel_size=7):
    super(SpatialAttention, self).__init__()
    assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
    padding = 3 if kernel_size == 7 else 1
    self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
    self.sigmoid = nn.Sigmoid()
  def forward(self, x):
    avg_out = torch.mean(x, dim=1, keepdim=True)
    max_out, _ = torch.max(x, dim=1, keepdim=True)
    x = torch.cat([avg_out, max_out], dim=1)
    x = self.conv1(x)
    return self.sigmoid(x)

# 深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # 逐通道卷积：groups=in_channels=out_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        # 逐点卷积：普通1x1卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class MultiscaleModel(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(MultiscaleModel, self).__init__()

        self.branch1=nn.Sequential(
            SeparableConv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels,out_channels,kernel_size=1,stride=2,padding=0,bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        self.branch2=nn.Sequential(
            SeparableConv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        self.branch3=nn.Sequential(
            SeparableConv2d(in_channels,out_channels,kernel_size=7,stride=1,padding=3,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels,out_channels,kernel_size=5,stride=2,padding=2,bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3(x)
        outputs=out3+out2
        outputs=outputs+out1
        return outputs

class Block(nn.Module):

    def __init__(self, in_planes, out_planes, expansion,stride):
        super(Block, self).__init__()
        self.stride=stride
        planes=expansion*in_planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,stride=1,padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        if self.stride !=1 :
           self.ca=ChannelAttention(out_planes)
           self.sa=SpatialAttention()

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out=self.ca(out)*out if self.stride !=1 else out
        out=self.sa(out)*out if self.stride !=1 else out

        out += self.shortcut(x) if self.stride !=1 else out
        out = F.relu6(out)

        return out

class BB_Net(nn.Module):
    def __init__(self, block, num_blocks):
        super(BB_Net, self).__init__()
        self.in_planes = 64

        self.multi_scale = MultiscaleModel(3, 64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], expansion=1, stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], expansion=6, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], expansion=6, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], expansion=6, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, expansion,stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,expansion, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.multi_scale(x)
        out1 = self.relu(out1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        return out1,out2,out3,out4,out5


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

# Feature Fusion Block
class FFB(nn.Module):
    def __init__(self,channel):
        super(FFB, self).__init__()
        self.conv_1=conv3x3(channel,channel)
        self.bn_1=nn.BatchNorm2d(channel)
        self.conv_2=conv3x3(channel,channel)
        self.bn_2=nn.BatchNorm2d(channel)

    def forward(self,x_1,x_2):
        out=x_1*x_2
        out=F.relu(self.bn_1(self.conv_1(out)),inplace=True)
        out=F.relu(self.bn_2(self.conv_2(out)),inplace=True)
        return out

# Spatial Attention Block
class SAB(nn.Module):
    def __init__(self,in_chan,out_chan):
        super(SAB, self).__init__()
        self.conv_atten=conv3x3(2,1)
        self.conv=conv3x3(in_chan,out_chan)
        self.bn=nn.BatchNorm2d(out_chan)

    def forward(self,x):
        avg_out=torch.mean(x,dim=1,keepdim=True)
        max_out,_=torch.max(x,dim=1,keepdim=True)
        atten=torch.cat([avg_out,max_out],dim=1)
        atten=torch.sigmoid(self.conv_atten(atten))
        out=torch.mul(x,atten)
        out=F.relu(self.bn(self.conv(out)),inplace=True)
        return out

# Cross Fusion Block
class CFB(nn.Module):
    def __init__(self,channel):
        super(CFB, self).__init__()
        self.down=nn.Sequential(
            conv3x3(channel,channel,stride=2),
            nn.BatchNorm2d(channel)
            )
        self.conv_1=conv3x3(channel,channel)
        self.bn_1=nn.BatchNorm2d(channel)
        self.conv_2=conv3x3(channel,channel)
        self.bn_2=nn.BatchNorm2d(channel)
        self.mul=FFB(channel)

    def forward(self,x_high,x_low):
        left_1=x_low
        left_2=F.relu(self.down(x_low),inplace=True)
        right_1=F.interpolate(x_high,size=x_low.size()[2:],mode='bilinear',align_corners=True)
        right_2=x_high
        left=F.relu(self.bn_1(self.conv_1(left_1*right_1)),inplace=True)
        right=F.relu(self.bn_2(self.conv_2(left_2*right_2)),inplace=True)
        right=F.interpolate(right,size=x_low.size()[2:],mode='bilinear',align_corners=True)
        out=self.mul(left,right)
        return out

#Edge Refine Block
class ERB(nn.Module):
    def __init__(self,channel):
        super(ERB, self).__init__()
        self.conv_atten=conv1x1(channel,channel)
        self.conv_1=conv3x3(channel,channel)
        self.bn_1=nn.BatchNorm2d(channel)
        self.conv_2=conv3x3(channel,channel)
        self.bn_2=nn.BatchNorm2d(channel)

    def forward(self,x_1,x_edge):
        x=x_1+x_edge
        atten=F.avg_pool2d(x,x.size()[2:])
        atten=torch.sigmoid(self.conv_atten(atten))
        out=torch.mul(x,atten)+x
        out=F.relu(self.bn_1(self.conv_1(out)),inplace=True)
        out=F.relu(self.bn_2(self.conv_2(out)),inplace=True)
        return out

class HDF_Net(nn.Module):
    def __init__(self):
        super(HDF_Net, self).__init__()
        block=Block
        self.bkbone=BB_Net(block,[1,3,3,1])

        self.path1_1=nn.Sequential(
            conv1x1(512,64),
            nn.BatchNorm2d(64)
        )
        self.path1_2=nn.Sequential(
            conv1x1(256,64),
            nn.BatchNorm2d(64)
        )

        self.path2=SAB(128,64)

        self.path3=nn.Sequential(
            conv1x1(64,64),
            nn.BatchNorm2d(64)
        )

        self.srm=SRM.SRMConv2D(3,3)
        self.v1=ViT(patch_size=32,depth=2)
        self.v2=ViT(patch_size=16,depth=4)
        self.v3=ViT(patch_size=8,depth=6)

        self.aspp=ASPP(512,[6,12,18])      # ADD 2022.12.18
        self.fuse1_1=FFB(64)               # ADD 2022.12.18
        self.fuse1_2=FFB(64)
        self.fuse12=CFB(64)
        self.fuse13=FFB(64)
        self.fuse23=ERB(64)

        self.fla=FLA(64)

        self.head_1 = conv3x3(64, 1, bias=True)
        self.head_2 = conv3x3(64, 1, bias=True)
        self.head_3 = conv3x3(64, 1, bias=True)

    def forward(self,x,shape=None):
        shape=x.size()[2:] if shape is None else shape
        l1,l2,l3,l4,l5=self.bkbone(x)
        xx= self.srm(x)

        path1_1_1=self.aspp(l5)                             # ADD 2022.12.18
        path1_1=F.relu(self.path1_1(l5),inplace=True)
        path1_1=self.fuse1_1(path1_1_1,path1_1)             # ADD 2022.12.18
        path1_1=F.interpolate(path1_1,size=l4.size()[2:],mode='bilinear',align_corners=True)
        path1_2=F.relu(self.path1_2(l4),inplace=True)
        path1=self.fuse1_2(path1_1,path1_2)
        v1=self.v1(xx)
        a, b, c = v1.size()    # a: BatchSize b: ImageH*ImageW c: channels
        v1 = v1.reshape([a, c, int(b**(0.5)), -1])
        v1=F.interpolate(v1,size=l4.size()[2:],mode='bilinear',align_corners=True)
        path1=path1+v1
        path111=self.fla(path1)
        path1=path1+path111
        # path1=F.interpolate(path1,size=l3.size()[2:],mode='bilinear',align_corners=True)
        path2=self.path2(l3)
        path12=self.fuse12(path1,path2)
        v2=self.v2(xx)
        a, b, c = v2.size()
        v2 = v2.reshape([a, c, int(b ** (0.5)), -1])
        v2=F.interpolate(v2,size=l3.size()[2:],mode='bilinear',align_corners=True)
        path12=path12+v2
        path121=self.fla(path12)
        path12=path12+path121
        path12=F.interpolate(path12,size=l2.size()[2:],mode='bilinear',align_corners=True)
        path3_1=F.relu(self.path3(l2),inplace=True)
        path3_2=F.interpolate(path1_1,size=l2.size()[2:],mode='bilinear',align_corners=True)
        path3=self.fuse13(path3_1,path3_2)

        path13=self.fuse23(path12,path3)
        v3=self.v3(xx)
        a, b, c = v3.size()
        v3 = v3.reshape([a, c, int(b ** (0.5)), -1])
        v3=F.interpolate(v3,size=l2.size()[2:],mode='bilinear',align_corners=True)
        path13=path13+v3
        path131=self.fla(path13)
        path13=path13+path131
        # path13=F.interpolate(path13,size=l1.size()[2:],mode='bilinear',align_corners=True)
        logits_1=F.interpolate(self.head_1(path13),size=shape,mode='bilinear',align_corners=True)
        logits_2=F.interpolate(self.head_2(path12), size=shape, mode='bilinear', align_corners=True)
        logits_3=F.interpolate(self.head_3(path1),size=shape,mode='bilinear',align_corners=True)

        return logits_1,logits_2,logits_3
