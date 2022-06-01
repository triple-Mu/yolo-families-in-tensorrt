import subprocess
import sys
import torch
from pathlib import Path

# def yolov5(path='/home/ubuntu/work/yolo/yolov5',weights='./yolov5s.pt',device=torch.device("cuda:0")):
#     path = Path(path)
#     weights = Path(weights)
#     assert path.exists() and weights.exists()
#     jitModel = weights.with_suffix('.torchscript')
#     if not jitModel.exists():
#         cmd = f'cd {str(path.absolute())} && python3 export.py --weights {str(weights.absolute())} --include torchscript'
#
#         cmd = f'ln -s {str(path.absolute())} {str(Path.cwd().parent.absolute())}'
#
#         subprocess.check_output(cmd,shell=True)
#         print(f"{weights.name} export jit success!")
#     model = torch.jit.load(str(jitModel),map_location=device)
#     return model

def yolov5(path='/home/ubuntu/work/yolo/yolov5',weights='./yolov5s.pt',device=None):
    path = Path(path)
    weights = Path(weights)
    assert path.exists() and weights.exists()
    cwd = Path.cwd().parent/'yolov5'
    model = None
    if not cwd.exists():
        cmd = f'ln -s {str(path.absolute())} {str(cwd.absolute())}'
        subprocess.check_output(cmd,shell=True)
    try:
        sys.path.append("../yolov5/")
        from models.experimental import attempt_load
        from models.yolo import Detect
    except Exception as e:
        print(f"error message is {e}")
    else:
        model = attempt_load(str(weights), device=device, inplace=True, fuse=True)  # load FP32 model
        model.eval()
        for k, m in model.named_modules():
            if isinstance(m, Detect):
                m.inplace = False
                m.onnx_dynamic = False
                m.export = True
    return model











if __name__ == '__main__':
    model = yolov5(weights='../yolov5s.pt')
    print(model)