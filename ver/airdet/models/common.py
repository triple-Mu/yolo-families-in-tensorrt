# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from functools import partial
import math

import copy


def get_norm(name, out_channels, inplace=True):
    module = None
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    elif name == 'gn':
        module = nn.GroupNorm(num_channels=out_channels, num_groups=32)
    return module


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

def get_act_layer(name: Union[Type[nn.Module], str] = 'relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if not isinstance(name, str):
        # callable, module, etc
        return name
    if not (is_no_jit() or is_exportable() or is_scriptable()):
        if name in _ACT_LAYER_ME:
            return _ACT_LAYER_ME[name]
    if not (is_no_jit() or is_exportable()):
        if name in _ACT_LAYER_JIT:
            return _ACT_LAYER_JIT[name]
    return _ACT_LAYER_DEFAULT[name]

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])

    enclosed_rb, enclosed_lt = None, None
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious



class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self, in_channels, out_channels, ksize, stride=1, groups=1, bias=False, act="silu", norm='bn'
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            # x = self.norm(x)
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Scale(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class GFocalHead_Tiny(nn.Module):
    """Ref to Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,  # 4
                 feat_channels=256,
                 reg_max=12,
                 reg_topk=4,
                 reg_channels=64,
                 strides=[8, 16, 32],
                 add_mean=True,
                 norm='gn',
                 act='relu',
                 start_kernel_size=3,
                 conv_groups=1,
                 conv_type='BaseConv',
                 simOTA_cls_weight=1.0,
                 simOTA_iou_weight=3.0,
                 octbase=8,
                 **kwargs):
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.strides = strides
        self.feat_channels = feat_channels if isinstance(feat_channels, list) \
            else [feat_channels] * len(self.strides)

        self.cls_out_channels = num_classes + 1  # add 1 for keep consistance with former models
        # and will be deprecated in future.
        self.stacked_convs = stacked_convs
        self.conv_groups = conv_groups
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        self.start_kernel_size = start_kernel_size
        self.decode_in_inference = True  # will be set as False, when trying to convert onnx models

        self.norm = norm
        self.act = act
        self.conv_module = DWConv if conv_type == 'DWConv' else BaseConv

        if add_mean:
            self.total_dim += 1

        super(GFocalHead_Tiny, self).__init__()

        self._init_layers()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        self.relu = nn.ReLU(inplace=True)
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else self.start_kernel_size
            cls_convs.append(
                self.conv_module(
                    chn,
                    feat_channels,
                    kernel_size,
                    stride=1,
                    groups=self.conv_groups,
                    norm=self.norm,
                    act=self.act))
            reg_convs.append(
                self.conv_module(
                    chn,
                    feat_channels,
                    kernel_size,
                    stride=1,
                    groups=self.conv_groups,
                    norm=self.norm,
                    act=self.act))

        conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]
        reg_conf = nn.Sequential(*conf_vector)

        return cls_convs, reg_convs, reg_conf

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_confs = nn.ModuleList()

        for i in range(len(self.strides)):
            cls_convs, reg_convs, reg_conf = self._build_not_shared_convs(
                self.in_channels[i],
                self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.reg_confs.append(reg_conf)

        self.gfl_cls = nn.ModuleList(
            [nn.Conv2d(
                self.feat_channels[i],
                self.cls_out_channels,
                3,
                padding=1) for i in range(len(self.strides))])

        self.gfl_reg = nn.ModuleList(
            [nn.Conv2d(
                self.feat_channels[i],
                4 * (self.reg_max + 1),
                3,
                padding=1) for i in range(len(self.strides))])

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for cls_conv in self.cls_convs:
            for m in cls_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conv in self.reg_convs:
            for m in reg_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conf in self.reg_confs:
            for m in reg_conf:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

    def forward(self, xin, labels=None, imgs=None, conf_thre=0.05, nms_thre=0.7):

        # prepare labels during training
        b, c, h, w = xin[0].shape
        if labels is not None:
            gt_bbox_list = []
            gt_cls_list = []
            for label in labels:
                gt_bbox_list.append(label.bbox)
                gt_cls_list.append((label.get_field("labels") - 1).long())  # labels starts from 1

        # prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(
                xin[i].shape[0],
                xin[i].shape[-2:],
                stride,
                dtype=torch.float32,
                device=xin[0].device)
            for i, stride in enumerate(self.strides)]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)

        # forward for bboxes and classification prediction
        cls_scores, bbox_preds = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.reg_confs,
            self.scales,
        )
        flatten_cls_scores = torch.cat(cls_scores, dim=1)
        flatten_bbox_preds = torch.cat(bbox_preds, dim=1)

        output = self.get_bboxes(
            flatten_cls_scores,
            flatten_bbox_preds,
            mlvl_priors)
        return output

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, reg_conf, scale):
        """Forward feature of a single scale level.

        """
        cls_feat = x
        reg_feat = x

        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(gfl_reg(reg_feat)).float()
        N, C, H, W = bbox_pred.size()
        prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
        prob_topk, _ = prob.topk(self.reg_topk, dim=2)

        if self.add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                             dim=2)
        else:
            stat = prob_topk

        quality_score = reg_conf(stat.reshape(N, 4 * self.total_dim, H, W))
        cls_score = gfl_cls(cls_feat).sigmoid() * quality_score

        flatten_cls_score = cls_score.flatten(start_dim=2).transpose(1, 2)
        flatten_bbox_pred = bbox_pred.flatten(start_dim=2).transpose(1, 2)
        return flatten_cls_score, flatten_bbox_pred

    def get_single_level_center_priors(self,
                                       batch_size,
                                       featmap_size,
                                       stride,
                                       dtype,
                                       device):

        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype, device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype, device=device)) * stride

        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)

        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)

        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def get_bboxes(self, cls_preds, reg_preds, mlvl_center_priors, img_meta=None):

        device = cls_preds.device
        batch_size = cls_preds.shape[0]
        dis_preds = self.integral(reg_preds) * mlvl_center_priors[..., 2, None]
        bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)

        obj = torch.ones_like(cls_preds[..., 0:1])
        res = torch.cat([bboxes, obj, cls_preds[..., 0:self.num_classes]], dim=-1)

        return res


class YOLOXHead(nn.Module):
    def __init__(
            self,
            num_classes,
            decode_in_inference,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): wheather apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = decode_in_inference  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def init_weights(self, ):
        self.initialize_biases(1e-2)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

            outputs.append(output)

        self.hw = [x.shape[-2:] for x in outputs]
        # [batch, n_anchors_all, 85]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)
        return outputs

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class PAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=[2, 3, 4],
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
    def init_weights(self):
        pass

    def forward(self, out_features):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

class GiraffeNeck(nn.Module):
    def __init__(self, min_level, max_level, num_levels, norm_layer, norm_kwargs, act_type, fpn_config, fpn_name, fpn_channels, out_fpn_channels, weight_method, depth_multiplier, width_multiplier, with_backslash, with_slash, with_skip_connect, skip_connect_type, separable_conv, feature_info, merge_type, pad_type, downsample_type, upsample_type, apply_resample_bn, conv_after_downsample, redundant_bias, conv_bn_relu_pattern, alternate_init):
        super(GiraffeNeck, self).__init__()

        self.num_levels = num_levels
        self.min_level = min_level
        self.in_features = [0, 1, 2, 3, 4, 5, 6][self.min_level-1:self.min_level-1+num_levels]
        self.alternate_init = alternate_init
        norm_layer = norm_layer or nn.BatchNorm2d
        if norm_kwargs:
            norm_layer = partial(norm_layer, **norm_kwargs)
        act_layer = get_act_layer(act_type) or _ACT_LAYER
        fpn_config = fpn_config or get_graph_config(
            fpn_name, min_level=min_level, max_level=max_level, weight_method=weight_method, depth_multiplier=depth_multiplier, with_backslash=with_backslash, with_slash=with_slash, with_skip_connect=with_skip_connect, skip_connect_type=skip_connect_type)

        # width scale
        for i in range(len(fpn_channels)):
            fpn_channels[i] = int(fpn_channels[i] * width_multiplier)

        self.resample = nn.ModuleDict()
        for level in range(num_levels):
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                reduction = feature_info[level]['reduction']
            else:
                # Adds a coarser level by downsampling the last feature map
                reduction_ratio = 2
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=feature_info[level-1]['num_chs'],
                    pad_type=pad_type,
                    downsample=downsample_type,
                    upsample=upsample_type,
                    norm_layer=norm_layer,
                    reduction_ratio=reduction_ratio,
                    apply_bn=apply_resample_bn,
                    conv_after_downsample=conv_after_downsample,
                    redundant_bias=redundant_bias,
                )
                in_chs = feature_info[level-1]['num_chs']
                reduction = int(reduction * reduction_ratio)
                feature_info.append(dict(num_chs=in_chs, reduction=reduction))

        self.cell = SequentialList()
        logging.debug('building giraffeNeck')
        giraffe_layer = GiraffeLayer(
            feature_info=feature_info,
            fpn_config=fpn_config,
            inner_fpn_channels=fpn_channels,
            outer_fpn_channels=out_fpn_channels,
            num_levels=num_levels,
            pad_type=pad_type,
            downsample=downsample_type,
            upsample=upsample_type,
            norm_layer=norm_layer,
            act_layer=act_layer,
            separable_conv=separable_conv,
            apply_resample_bn=apply_resample_bn,
            conv_after_downsample=conv_after_downsample,
            conv_bn_relu_pattern=conv_bn_relu_pattern,
            redundant_bias=redundant_bias,
            merge_type=merge_type
        )
        self.cell.add_module('giraffeNeck', giraffe_layer)
        feature_info = giraffe_layer.feature_info

    def init_weights(self, pretrained=False):
        for n, m in self.named_modules():
            if 'backbone' not in n:
                if self.alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)


    def forward(self, x: List[torch.Tensor]):
        if type(x) is tuple:
            x = list(x)
        x = [x[f] for f in self.in_features]
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x

def giraffeneck_config(min_level, max_level, weight_method=None, depth_multiplier=5, with_backslash=False, with_slash=False, with_skip_connect=False, skip_connect_type='dense'):
    """Graph config with log2n merge and panet"""
    if skip_connect_type == 'dense':
        nodes, connections = get_dense_graph(depth_multiplier)
    elif skip_connect_type == 'log2n':
        nodes, connections = get_log2n_graph(depth_multiplier)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(connections)

    drop_node = []
    nodes, input_nodes, output_nodes = get_graph_info(graph)

    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    node_ids_per_layer = {}

    pnodes = {}
    def update_drop_node(new_id, input_offsets):
        if new_id not in drop_node:
            new_id = new_id
        else:
            while new_id in drop_node:
                if new_id in pnodes:
                    for n in pnodes[new_id]['inputs_offsets']:
                        if n not in input_offsets and n not in drop_node:
                            input_offsets.append(n)
                new_id = new_id - 1
        if new_id not in input_offsets:
            input_offsets.append(new_id)

    # top-down layer
    for i in range(max_level, min_level-1, -1):
        node_ids_per_layer[i] = []
        for id, node in enumerate(nodes):
            input_offsets = []
            if id in input_nodes:
                input_offsets.append(node_ids[i][0])
            else:
                if with_skip_connect:
                    for input_id in node.inputs:
                        new_id = nodeid_trans(input_id, i-min_level, num_levels)
                        update_drop_node(new_id, input_offsets)


            # add top2down
            new_id = nodeid_trans(id, i-min_level, num_levels)

            # add backslash node
            def cal_backslash_node(id):
                ind = id // num_levels
                mod = id % num_levels
                if ind % 2  == 0:   # even
                    if mod == (num_levels-1):
                        last = -1
                    else:
                        last = (ind - 1) * num_levels + (num_levels - 1 - mod - 1)
                else:    # odd
                    if mod == 0:
                        last = -1
                    else:
                        last = (ind - 1) * num_levels + (num_levels - 1 - mod + 1)

                return last

            # add slash node
            def cal_slash_node(id):
                ind = id // num_levels
                mod = id % num_levels
                if ind % 2  == 1:   # odd
                    if mod == (num_levels - 1):
                        last = -1
                    else:
                        last = (ind - 1) * num_levels + (num_levels - 1 - mod - 1)
                else:    # even
                    if mod == 0:
                        last = -1
                    else:
                        last = (ind - 1) * num_levels + (num_levels - 1 - mod + 1)

                return last

            # add last node
            last = new_id - 1
            update_drop_node(last, input_offsets)

            if with_backslash:
                backslash = cal_backslash_node(new_id)
                if backslash != -1 and backslash not in input_offsets:
                    input_offsets.append(backslash)

            if with_slash:
                slash = cal_slash_node(new_id)
                if slash != -1 and slash not in input_offsets:
                    input_offsets.append(slash)

            if new_id in drop_node:
                input_offsets = []

            pnodes[new_id] = {
                'reduction': 1 << i,
                'inputs_offsets': input_offsets,
                'weight_method': weight_method,
                'is_out': 0,
            }

        input_offsets = []
        for out_id in output_nodes:
            new_id = nodeid_trans(out_id, i-min_level, num_levels)
            input_offsets.append(new_id)

        pnodes[node_ids[i][0] + num_levels * (len(nodes) + 1)] = {
                'reduction': 1 << i,
                'inputs_offsets': input_offsets,
                'weight_method': weight_method,
                'is_out': 1,
            }

    pnodes = dict(sorted(pnodes.items(), key=lambda x:x[0]))
    return pnodes


def get_graph_config(fpn_name, min_level=3, max_level=7, weight_method='concat', depth_multiplier=5, with_backslash=False, with_slash=False, with_skip_connect=False, skip_connect_type='dense'):
    name_to_config = {
        'giraffeneck': giraffeneck_config(min_level=min_level, max_level=max_level, weight_method=weight_method, depth_multiplier=depth_multiplier, with_backslash=with_backslash, with_slash=with_slash, with_skip_connect=with_skip_connect, skip_connect_type=skip_connect_type),

    }
    return name_to_config[fpn_name]

def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "GFocalV2":
        return GFocalHead_Tiny(**head_cfg)
    elif name == "YOLOX":
        return YOLOXHead(**head_cfg)

def build_neck(cfg):
    neck_cfg = copy.deepcopy(cfg)
    name = neck_cfg.pop("name")
    if name == "PAFPN":
        return PAFPN(**neck_cfg)
    elif name == "GiraffeNeck":
        return GiraffeNeck(**neck_cfg)