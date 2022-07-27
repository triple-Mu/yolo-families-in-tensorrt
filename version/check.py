import contextlib
import re
import sys
from enum import IntEnum

import torch
import torch.nn as nn


class TRT_NMS(torch.autograd.Function):
    """TensorRT NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0,
                                max_output_boxes, (batch_size, 1),
                                dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0,
                                    num_classes,
                                    (batch_size, max_output_boxes),
                                    dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        out = g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4,
        )
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_TRT(nn.Module):
    """onnx module with TensorRT NMS operation."""

    def __init__(self,
                 max_obj=100,
                 iou_thres=0.45,
                 score_thres=0.25,
                 device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.background_class = (-1, )
        self.box_coding = (1, )
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, x):
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(
            box,
            score,
            self.background_class,
            self.box_coding,
            self.iou_threshold,
            self.max_obj,
            self.plugin_version,
            self.score_activation,
            self.score_threshold,
        )
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    """export onnx or tensorrt model with NMS operation."""

    def __init__(self,
                 model,
                 max_obj=100,
                 iou_thres=0.45,
                 score_thres=0.25,
                 device=None):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.model = model.to(device)
        self.end2end = ONNX_TRT(max_obj, iou_thres, score_thres, device)
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x


def check(t, sub='Conv'):
    flag = False
    try:
        s = re.findall("['](.*)[']", str(t))[0]
        s = s.split('.')
        s = set(s)
    except Exception as e:
        print(f'Something wrong with {e}')
    else:
        if isinstance(sub, str):
            if sub in s:
                flag = True
        elif isinstance(sub, list):
            if set(sub) & s:
                flag = True
    return flag


def get_root(t):
    s = re.findall("['](.*)[']", str(type(t)))[0]
    s = s.split('.')
    return s


@contextlib.contextmanager
def set_env(version=5):
    assert version in (5, 6, 7), 'Only support yolov-5/6/7'
    path = 'version' + ('' if version == 6 else f'/yolov{version}')
    try:
        sys.path.insert(0, path)
        yield
    finally:
        sys.path.remove(path)


class MODEL(IntEnum):
    yolov5 = 5
    yolov6 = 6
    yolov7 = 7
    airdet = 8
    yolox = 10
    ppyoloe = -1
