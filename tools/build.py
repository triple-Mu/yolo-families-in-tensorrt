from collections import OrderedDict, namedtuple

import numpy as np
import tensorrt as trt
import torch
import cv2
import time
import random


names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}



class YoLov5TRT(object):

    def __init__(self, engine,trtonnx,device = None,workspace=8,fp16=True):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine = engine
        self.trtonnx = trtonnx
        self.device = device if device else torch.device("cuda:0")
        self.workspace = workspace
        self.fp16 = fp16
        self.trt,self.bindings, self.binding_addrs, self.context = None,None,None,None
        if not self.engine.exists():
            assert self._build()
        else:
            self._load()
        self.shape = self.bindings['images'].shape
        self.warmup(self.shape)

    def _build(self):
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace * 1 << 30
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse_from_file(str(self.trtonnx)):
            raise RuntimeError(f'failed to load ONNX file: {str(self.trtonnx)}')
        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        with builder.build_serialized_network(network, config) as engine, open(self.engine, 'wb') as t:
            t.write(engine)
        self.trt = engine
        return self.engine.stat().st_size >= 1024

    def _load(self):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(str(self.engine), 'rb') as f, trt.Runtime(logger) as runtime:
            self.trt = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        for index in range(self.trt.num_bindings):
            name = self.trt.get_binding_name(index)
            dtype = trt.nptype(self.trt.get_binding_dtype(index))
            shape = tuple(self.trt.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = self.trt.create_execution_context()
        self.bindings = bindings
        self.binding_addrs = binding_addrs
        self.context = context

    def warmup(self,shape= (1,3,640,640),times=10):
        for _ in range(times):
            tmp = torch.randn(shape).to(self.device)
            self.binding_addrs['images'] = int(tmp.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))

    def infer(self, im,show=False):
        img_copy = self.letterbox(im.copy(),self.shape[2:])[0]
        if self.shape[0] == 1:
            im, ratio, dwdh = self.preprocess(im)
            start = time.perf_counter()
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            end = time.perf_counter()
            print(f'Infer one image for {end-start}s')
            img_copy = self.draw(img_copy)
            cv2.imwrite('detect.jpg',img_copy)
            if show:
                cv2.imshow('result',cv2.cvtColor(img_copy,cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)

        else:
            print(f'For batch > 1, please write your own infer script')




    def letterbox(self,im,
                  new_shape=(640, 640),
                  color=(114, 114, 114),
                  auto=False,
                  scaleFill=False,
                  scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
            1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[
                0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im,top,bottom,left,right,cv2.BORDER_CONSTANT,value=color)  # add border
        return im, ratio, (dw, dh)

    def preprocess(self,img):
        img, ratio, dwdh = self.letterbox(img, self.shape[2:])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(self.device)
        im = im.float()
        im /= 255
        return im, ratio, dwdh

    def draw(self,img):
        num_detections = self.bindings['num_detections'].data
        detection_boxes = self.bindings['detection_boxes'].data
        detection_scores = self.bindings['detection_scores'].data
        detection_classes = self.bindings['detection_classes'].data
        for i,num in enumerate(num_detections):
            num = int(num.squeeze())
            box_img = detection_boxes[i, :num].round().int()
            box_img = self.clip_coords(box_img,self.shape[2:])
            score_img = detection_scores[i, :num]
            clss_img = detection_classes[i, :num]
            for i, (box, score, clss) in enumerate(zip(box_img, score_img, clss_img)):
                name = names[clss]
                color = colors[name]
                cv2.rectangle(img, box[:2].tolist(), box[2:].tolist(), color, 2)
                cv2.putText(img, name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            [225, 255, 255], thickness=2)
            return img


    def clip_coords(self,boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes


    def destroy(self):
        pass

class YoLov5ORT(object):
    def __init__(self, trtonnx, device=None):
        pass







