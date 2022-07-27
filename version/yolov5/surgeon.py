import torch
import torch.nn as nn

from version.check import End2End, check
from version.yolov5.models.experimental import Ensemble, SiLU
from version.yolov5.models.yolo import Detect, Model


def model2deploy(weights, device=None, end2end=False, config=None, **kwargs):
    device = torch.device('cpu') if device is None else device
    inplace = True
    model = Ensemble()
    ckpt = torch.load(weights, map_location='cpu')  # load
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
    model.append(ckpt.fuse().eval())  # fused or un-fused model in eval mode

    # Compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6):
            m.inplace = inplace  # torch 1.7.0 compatibility

        elif t is nn.SiLU:
            m.inplace = inplace
            m.__class__ = SiLU

        elif check(t, 'Detect'):
            m.inplace = inplace
            m.__class__ = Detect
            if not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)

        elif check(t, 'Model'):
            m.inplace = inplace
            m.__class__ = Model

        elif check(t, 'Conv'):
            m._non_persistent_buffers_set = set()

        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    model = model[-1] if len(model) == 1 else model
    if end2end:
        assert config is not None, 'There is no end2end config'
        model = End2End(model, **config, device=device)

    return model
