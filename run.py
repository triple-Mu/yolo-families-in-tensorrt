import argparse
from pathlib import Path

import cv2
import onnx
import torch
from tools.load import yolov5
from tools.common import End2EndTRT,End2EndORT
from tools.func import to_2tuple,export,trtnms,ortnms
from tools.build import YoLov5TRT



def main(opt):
    # export config
    path = Path(opt.yolov5)
    weights = Path(opt.weights)
    imgsz = to_2tuple(opt.imgsz)
    image = Path(opt.image)
    batch = opt.batch
    trt = Path(opt.trt) if opt.trt else opt.trt
    ort = Path(opt.ort) if opt.ort else opt.ort
    export_device = torch.device('cpu')
    opset = opt.opset
    detections_per_img = opt.maxObj
    score_thresh = opt.score
    nms_thresh = opt.nms_score

    # build config
    device = torch.device(opt.device)
    fp16 = opt.fp16
    workspace = opt.GiB

    # export process
    im = torch.randn((batch,3,*imgsz)).to(export_device)
    V5 = yolov5(path,weights,export_device)
    for _ in range(2):
        _ = V5(im) # run 2 times

    trtonnx ,ortonnx = None,None
    if trt:
        if not trt.exists():
            trt.mkdir(parents=True,exist_ok=True)
        trtonnx = (trt/(weights.stem+"_trtnms")).with_suffix(".onnx")
        end2endTRT = End2EndTRT(V5,export_device)
        input_names=['images']
        output_names=['boxes','scorees']
        onnx_graph = export(end2endTRT,im,input_names,output_names,opset)
        nms_graph = trtnms(onnx_graph,detections_per_img,score_thresh,nms_thresh)
        onnx.save(nms_graph,str(trtonnx))
    if ort:
        if not ort.exists():
            ort.mkdir(parents=True,exist_ok=True)
        ortonnx = (ort/(weights.stem+"_ortnms")).with_suffix(".onnx")
        end2endORT = End2EndORT(V5,export_device,opt.imgsz)
        input_names=['images']
        output_names=['nmsboxes','boxes','scores0','classes','scores']
        onnx_graph = export(end2endORT,im,input_names,output_names,opset)
        nms_graph = ortnms(onnx_graph, detections_per_img, score_thresh, nms_thresh)
        onnx.save(nms_graph,str(ortonnx))

    end2endTRT,end2endORT = None,None # del torch model

    # build process
    if trt:
        img = cv2.imread(str(image))
        name = (trt/weights.stem).with_suffix('.engine')
        yoloTRT = YoLov5TRT(name,trtonnx,device = device,workspace=workspace,fp16=fp16)
        yoloTRT.warmup(im.shape)
        yoloTRT.infer(img,show=True)

    if ort:
        pass






def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5',
                        type=str,
                        default='/home/ubuntu/work/yolo/yolov5',
                        help='origin yolov5 repo path')
    parser.add_argument('--weights',
                        type=str,
                        default='/home/ubuntu/work/yolo/yolov5/yolov5s.pt',
                        help='origin yolov5 weights path')
    parser.add_argument('--image',
                        type=str,
                        default='images/bus.jpg',
                        help='test image')
    parser.add_argument('--imgsz',
                        type=int,
                        default=640,
                        help='image size default [640,640]')
    parser.add_argument('--batch',
                        type=int,
                        default=1,
                        help='batch size default 1')
    parser.add_argument('--opset',
                        type=int,
                        default=12,
                        help='opset version default 12')
    parser.add_argument('--onnx',
                        type=str,
                        default='./onnx/',
                        help='onnx save to path')
    parser.add_argument('--trt',
                        nargs='?',
                        const='./engine/',
                        default='',
                        help='save engine path')
    parser.add_argument('--ort',
                        nargs='?',
                        const='./ort/',
                        default='',
                        help='save onnx path')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device cpu or cuda:0')
    parser.add_argument('--GiB',
                        type=int,
                        default=8,
                        help='workspace for engine builder')
    parser.add_argument('--maxObj',
                        type=int,
                        default=100,
                        help='max object in one image')
    parser.add_argument('--score',
                        type=float,
                        default=0.25,
                        help='min score for nms attrs')
    parser.add_argument('--nms_score',
                        type=float,
                        default=0.45,
                        help='min mns_score for nms attrs')
    parser.add_argument('--fp16', action='store_true', help='fp16 exporter')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)