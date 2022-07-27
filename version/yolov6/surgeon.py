import torch
import torch.nn as nn

from version.check import End2End, check
from version.yolov6.layers.common import RepVGGBlock
from version.yolov6.models.yolo import Detect


class Conv(nn.Module):
    """Normal Conv with SiLU activation"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SiLU(nn.Module):
    """Activation of SiLU"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


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

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (torch.zeros(conv.weight.size(0), device=conv.weight.device)
              if conv.bias is None else conv.bias)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(
        torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model2deploy(weights, device=None, end2end=False, config=None, **kwargs):
    device = torch.device('cpu') if device is None else device
    ckpt = torch.load(weights, map_location=device)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].to(device).float()
    for m in model.modules():
        if check(type(m), 'Conv'):
            if hasattr(m, 'bn'):
                m.__class__ = Conv
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif check(type(m), 'RepVGGBlock'):
            m.__class__ = RepVGGBlock
            m.switch_to_deploy()
        elif check(type(m), 'Detect'):
            m.__class__ = Detect
            m.inplace = False
    if end2end:
        assert config is not None, 'There is no end2end config'
        model = End2End(model, **config, device=device)
    return model
