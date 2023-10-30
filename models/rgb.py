import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

        self.create_weight = True
        self.weight = None

    def forward(self, img, depth):
        x_compress = self.compress(img)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        if self.create_weight:
            # print(img.shape)
            self.weight = nn.Parameter(torch.rand((1, 1, 32, 32))).to('cuda')
            self.create_weight = False
        # print(depth)
        scale = torch.sum(torch.cat((self.weight * scale, (1 - self.weight) * torch.unsqueeze(depth, dim=1)), dim=1), dim=1, keepdim=True)
        # print(scale)
        return img*scale