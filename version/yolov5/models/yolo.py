import math

import torch
import torch.nn as nn

from version.check import check
from version.yolov5.models.common import Conv, DWConv

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_anchor_order(m):

    a = m.anchors.prod(-1).mean(-1).view(
        -1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def fuse_conv_and_bn(conv, bn):

    fusedconv = (nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device))

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = (torch.zeros(conv.weight.size(0), device=conv.weight.device)
              if conv.bias is None else conv.bias)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(
        torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self,
                 nc=80,
                 anchors=(),
                 ch=(),
                 inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors',
                             torch.tensor(anchors).float().view(
                                 self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1)
                               for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (x[i].view(bs, self.na, self.no, ny,
                              nx).permute(0, 1, 3, 4, 2).contiguous())

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = x[i].sigmoid()
            xy, wh, conf = y.split(
                (2, 2, self.nc + 1),
                4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
            xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2)**2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, -1, self.no))

        return torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx,
                                                                 device=d,
                                                                 dtype=t)
        # yv, xv = torch.meshgrid(y, x, indexing='ij') # torch>=1.10.0
        yv, xv = torch.meshgrid(y, x)
        grid = (torch.stack((xv, yv), 2).expand(shape) - 0.5
                )  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = ((self.anchors[i] * self.stride[i]).view(
            (1, self.na, 1, 1, 2)).expand(shape))
        return grid, anchor_grid


class Model(nn.Module):
    # YOLOv5 model
    def __init__(self,
                 cfg='yolov5s.yaml',
                 ch=3,
                 nc=None,
                 anchors=None):  # model, input channels, number of classes
        super().__init__()

        self.yaml = cfg  # model dict
        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            print(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = None, None
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if check(type(m), 'Detect'):
            m.__class__ = Detect
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([
                s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))
            ])  # forward
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride

    def forward(self, x):

        return self._forward_once(x)  # single-scale inference, train

    def _forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (y[m.f] if isinstance(m.f, int) else
                     [x if j == -1 else y[j]
                      for j in m.f])  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if check(type(m), 'Conv') and hasattr(m, 'bn'):
                m.__class__ = Conv
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward

            elif check(type(m), 'DWConv') and hasattr(m, 'bn'):
                m.__class__ = DWConv
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward

        return self
