import argparse
from pathlib import Path

import onnx
import onnxsim
import torch

from version.check import MODEL, set_env


def main(opt):
    mod = MODEL(opt.version)
    device = torch.device(opt.device.lower())
    weights = Path(opt.weights)
    imgsz = opt.imgsz * 2 if len(opt.imgsz) == 1 else opt.imgsz * 1  # expand
    assert weights.exists()
    deploy_func = None
    if mod.value == 5:
        from version.yolov5.surgeon import model2deploy

        deploy_func = model2deploy
    elif mod.value == 6:
        from version.yolov6.surgeon import model2deploy

        deploy_func = model2deploy
    elif mod.value == 7:
        from version.yolov7.surgeon import model2deploy

        deploy_func = model2deploy

    else:
        raise ValueError

    end2endconfig = {
        'max_obj': opt.max_obj,
        'iou_thres': opt.iou_thres,
        'score_thres': opt.score_thres,
    }

    params = dict(weights=opt.weights,
                  device=device,
                  end2end=opt.end2end,
                  config=end2endconfig)

    with set_env(mod.value):
        model = deploy_func(**params)
    model.eval()
    inputs = torch.randn(opt.batch_size, 3, *imgsz)

    save_to = (weights.parent / (weights.stem + '_end2end')
               if opt.end2end else weights).with_suffix('.onnx')
    torch.onnx.export(
        model,
        inputs,
        save_to,
        verbose=False,
        opset_version=opt.opset,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        if opt.end2end else ['outputs'],
    )

    model_onnx = onnx.load(str(save_to))  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    if opt.end2end:
        shapes = [
            opt.batch_size,
            1,
            opt.batch_size,
            opt.max_obj,
            4,
            opt.batch_size,
            opt.max_obj,
            opt.batch_size,
            opt.max_obj,
        ]
        for i in model_onnx.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

    if opt.simplify:
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, 'assert check failed'
        onnx.save(model_onnx, str(save_to))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        required=True,
                        help='model.pt path(s)')
    parser.add_argument(
        '--version',
        type=int,
        required=True,
        help='model version yolov(ver):ver or airdet:8 ppyoloe -1',
    )
    parser.add_argument(
        '--imgsz',
        '--img',
        '--img-size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='image (h, w)',
    )
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu',
    )
    parser.add_argument('--simplify',
                        action='store_true',
                        help='ONNX: simplify model')
    parser.add_argument('--opset',
                        type=int,
                        default=12,
                        help='ONNX: opset version')
    parser.add_argument('--end2end',
                        action='store_true',
                        help='TRT: add EfficientNMS_TRT to model')
    parser.add_argument('--max-obj',
                        type=int,
                        default=100,
                        help='TRT: topk for every image to keep')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.45,
                        help='TRT: IoU threshold')
    parser.add_argument('--score-thres',
                        type=float,
                        default=0.25,
                        help='TRT: confidence threshold')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
