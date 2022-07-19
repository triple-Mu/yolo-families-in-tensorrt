import re
import sys
import contextlib
from enum import IntEnum
from ver.yolov6.models.effidehead import RepVGGBlock,Detect as V6Detect
from ver.yolov6.base import fuse_model, SiLU
import torch
import torch.nn as nn
from pathlib import Path
from ver.check import check


class MODEL(IntEnum):
    yolov5  = 5
    yolov6  = 6
    yolov7  = 7
    airdet  = 8

class TRT_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_TRT(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
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
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(box, score, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.model = model.to(device)
        self.end2end = ONNX_TRT(max_obj, iou_thres, score_thres, device)
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = [module(x)[0] for module in self]
        y = torch.cat(y, 1)
        return y


class DetModel:
    def __init__(self,weights,version,end2end=True,device=None):
        mod = MODEL(version)
        assert mod.name in ('yolov5','yolov6','yolov7','airdet') , 'Your model not support!'
        self.device = device if device else torch.device('cpu')
        self.weights = Path(weights)
        assert self.weights.exists(), 'Weights is not exists'
        self.model = None
        self.end2end = end2end
        with set_env(mod.value):
            # self.model = eval(f'self.{mod.name}()')
            self.model = self.yolov6()
    def yolov5(self):

        pass

    def yolov6(self):
        ckpt = torch.load(self.weights, map_location=self.device)
        model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
        model = fuse_model(model).eval()
        for layer in model.modules():
            if check(type(layer),'RepVGGBlock'):
                layer.__class__ = RepVGGBlock
                layer.switch_to_deploy()
        for k, m in model.named_modules():
            if check(type(m), 'Conv'):
                if isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            elif check(type(m), 'Detect'):
                m.__class__ = V6Detect
                m.inplace = False
        return model

    def yolov7(self):
        model = Ensemble()
        ckpt = torch.load(self.weights, map_location=self.device)
        ckpt = (ckpt.get('ema') or ckpt['model']).to(self.device).float()
        model.append(ckpt.fuse().eval())

        # Compatibility updates
        for m in model.modules():
            t = type(m)
            if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU):
                m.inplace = True
            elif check(t, 'Detect'):
                if not isinstance(m.anchor_grid, list):
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl) # , Detect, Model
            elif check(t, 'Conv'):
                m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
            elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        if len(model) == 1:
            return model[-1]  # return model
        return model

    def airdet(self):
        pass

    def export(self,inputs = torch.randn(1,3,640,640), max_obj=100, iou_thres=0.45, score_thres=0.25):
        if self.end2end:
            self.model = End2End(self.model,max_obj,iou_thres,score_thres,self.device)
        self.model.eval()
        inputs = inputs.to(self.device)
        torch.onnx.export(
            self.model,
            inputs,
            self.weights.with_suffix('.onnx'),
            verbose=False,
            opset_version=12,
            input_names=['images'],
            output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes'] if self.end2end else ['output'],
            dynamic_axes=None)



@contextlib.contextmanager
def set_env(version=5):
    assert version in (5,6,7), 'Only support yolov-5/6/7'
    path = 'ver' + ('' if version == 6 else f'/yolov{version}')
    try:
        sys.path.insert(0, path)
        yield
    finally:
        sys.path.remove(path)




if __name__ == '__main__':

    mod = MODEL(5)
    print(mod.value)



