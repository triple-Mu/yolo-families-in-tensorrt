from itertools import repeat
from functools import partial
import collections.abc
import io
import cv2
import torch
import onnx
import random
import numpy as np
import onnx_graphsurgeon as gs
from .common import PostORT


torch2onnx = partial(torch.onnx.export,verbose=False,
                     training=torch.onnx.TrainingMode.EVAL,
                     do_constant_folding=True)

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def export(model,im,input_names=['images'],output_names=['outputs'],opset=12,dynamic_axes = None):
    model.eval()
    with io.BytesIO() as f:
        torch2onnx(model=model,args=im,f=f,
                   opset_version=opset,input_names=input_names,
                   output_names=output_names,dynamic_axes=dynamic_axes)
        f.seek(0)
        onnx_graph = onnx.load(f)
    return onnx_graph

def pre(batch=1,anchors=25200,num_classes=80):
    net = PostORT()
    net.eval()

    dynamic_axes = {
        'selected_indices': {0: 'num'},
        'boxes': {1: 'anchor'},
        'classes': {1: 'anchor'},
        'scores': {1: 'anchor'},
        'output': {0: 'num'}}

    num = random.randint(1, 100)
    a = np.random.randint(0, batch, (num,))
    a.sort()
    b = np.arange(1000, 1000 + num)
    c = np.zeros((num,), np.int64)
    selected_indices = np.concatenate([a[None], c[None], b[None]], 0).T
    boxes = np.random.randn(batch, anchors, 4).astype(np.float32)
    classes = np.random.randint(0, num_classes, (batch, anchors, 1))
    scores = np.random.randn(batch, anchors, 1).astype(np.float32)
    selected_indices, boxes, classes, scores = \
        torch.from_numpy(selected_indices), torch.from_numpy(boxes), \
        torch.from_numpy(classes), torch.from_numpy(scores)
    for _ in range(2):
        out = net(selected_indices, boxes, classes, scores)
    onnx_graph  = export(model=net,im=(selected_indices, boxes, classes, scores),
               input_names=['selected_indices', 'boxes', 'classes', 'scores'],
               opset=11,dynamic_axes=dynamic_axes)
    return onnx_graph



def trtnms(onnx_graph,detections_per_img,score_thresh,nms_thresh,batch=1,fp16=False):
    gs_graph = gs.import_onnx(onnx_graph)
    op_inputs = gs_graph.outputs
    op = "EfficientNMS_TRT"
    attrs = {"plugin_version": "1",
             "background_class": -1,  # no background class
             "max_output_boxes": detections_per_img,
             "score_threshold": score_thresh,
             "iou_threshold": nms_thresh,
             "score_activation": False,
             "box_coding": 0,}
    # NMS Outputs
    output_num_detections = gs.Variable(
        name="num_detections",
        dtype=np.int64,
        shape=[batch, 1],
    )  # A scalar indicating the number of valid detections per batch image.
    output_boxes = gs.Variable(
        name="detection_boxes",
        dtype=np.float16 if fp16 else np.float32,
        shape=[batch, detections_per_img, 4],
    )
    output_scores = gs.Variable(
        name="detection_scores",
        dtype=np.float16 if fp16 else np.float32,
        shape=[batch, detections_per_img],
    )
    output_labels = gs.Variable(
        name="detection_classes",
        dtype=np.int64,
        shape=[batch, detections_per_img],
    )
    op_outputs = [
        output_num_detections, output_boxes, output_scores, output_labels
    ]
    gs_graph.layer(op=op,
                   name="batched_nms",
                   inputs=op_inputs,
                   outputs=op_outputs,
                   attrs=attrs)
    gs_graph.outputs = op_outputs
    gs_graph.cleanup().toposort()
    return gs.export_onnx(gs_graph)



def ortnms(onnx_graph,detections_per_img,score_thresh,nms_thresh,num_classes=80,batch=1):
    graph = gs.import_onnx(pre(batch,num_classes=num_classes))
    gs_graph = gs.import_onnx(onnx_graph)
    nmsboxes,boxes,score0,classes,score = gs_graph.outputs
    max_output_boxes_per_class = gs.Constant(name='max_output_boxes_per_class',
                                             values=np.array([detections_per_img]))
    iou_threshold = gs.Constant(name='iou_threshold',
                                values=np.array([nms_thresh]).astype(np.float32))
    score_threshold = gs.Constant(name='score_threshold',
                                  values=np.array([score_thresh]).astype(np.float32))
    selected_indices = gs.Variable(name='selected_indices',
                                   dtype=np.int64,
                                   shape=['num_detections', 3])
    NonMaxSuppression = gs.Node(name='NonMaxSuppression',
                                     op='NonMaxSuppression',
                                     inputs=[nmsboxes, score0, max_output_boxes_per_class,
                                         iou_threshold, score_threshold],
                                     outputs=[selected_indices])
    gs_graph.nodes.append(NonMaxSuppression)

    names = {'selected_indices': [], 'boxes': [], 'classes': [], 'scores': []}
    for node in graph.nodes:
        try:
            inp = node.inputs[0]
            if inp.name in names.keys():
                names[inp.name].append(node)
        except:
            pass

    s = "Add_"
    for node in graph.nodes:
        node.name = s + node.name
        try:
            inp = node.inputs
            if inp:
                for i in inp:
                    if "Add" in i.name:
                        continue
                    i.name = s + i.name
        except:
            pass
        try:
            out = node.outputs
            if out:
                for i in out:
                    if "Add" in i.name:
                        continue
                    i.name = s + i.name
        except:
            pass
    names_ori = {'selected_indices': [], 'boxes': [], 'classes': [], 'scores': []}
    for node in gs_graph.nodes:
        try:
            out = node.outputs
            if out:
                for i in out:
                    if i.name in names_ori.keys():
                        names_ori[i.name].append(node)
        except:
            pass
    for node in graph.nodes:
        gs_graph.nodes.append(node)
    for name, nodes in names_ori.items():
        output = nodes[0].outputs
        tmp_node = names[name]
        for node in tmp_node:
            node.inputs[0] = output[0]

    gs_graph.outputs.clear()
    gs_graph.outputs = graph.outputs
    graph.inputs.clear()
    graph.outputs.clear()
    gs_graph.cleanup().toposort()
    return gs.export_onnx(gs_graph)







